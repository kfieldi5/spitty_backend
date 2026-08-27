import {initializeApp} from "firebase-admin/app";
import {FieldValue, getFirestore} from "firebase-admin/firestore";
import {onCall, onRequest, HttpsError} from "firebase-functions/v2/https";
import {onObjectFinalized} from "firebase-functions/v2/storage";
import {logger} from "firebase-functions";

initializeApp();

const db = getFirestore();

type DrumSound = "kick" | "snare" | "closedHat" | "openHat" | "rest";

type GridEvent = {
  sound: DrumSound;
  step: number;
};

type TrainingExamplePayload = {
  kind: "ftue" | "correction";
  storagePath: string;
  bpm: number;
  bars: number;
  quantization: "1/2" | "1/4" | "1/8" | "1/16";
  events: GridEvent[];
  appVersion?: string;
  localModelVersion?: string;
  notes?: string;
};

const sounds = new Set<DrumSound>([
  "kick",
  "snare",
  "closedHat",
  "openHat",
  "rest",
]);

function assertTrainingPayload(data: unknown): TrainingExamplePayload {
  if (!data || typeof data !== "object") {
    throw new HttpsError("invalid-argument", "Expected an object payload.");
  }
  const payload = data as Partial<TrainingExamplePayload>;
  if (payload.kind !== "ftue" && payload.kind !== "correction") {
    throw new HttpsError("invalid-argument", "kind must be ftue or correction.");
  }
  if (
    typeof payload.storagePath !== "string" ||
    !payload.storagePath.startsWith("training_uploads/")
  ) {
    throw new HttpsError(
      "invalid-argument",
      "storagePath must point to training_uploads/."
    );
  }
  if (typeof payload.bpm !== "number" || payload.bpm < 40 || payload.bpm > 220) {
    throw new HttpsError("invalid-argument", "bpm must be between 40 and 220.");
  }
  if (typeof payload.bars !== "number" || payload.bars < 1 || payload.bars > 32) {
    throw new HttpsError("invalid-argument", "bars must be between 1 and 32.");
  }
  if (!["1/2", "1/4", "1/8", "1/16"].includes(payload.quantization ?? "")) {
    throw new HttpsError("invalid-argument", "Invalid quantization.");
  }
  if (!Array.isArray(payload.events)) {
    throw new HttpsError("invalid-argument", "events must be an array.");
  }
  for (const event of payload.events) {
    if (!sounds.has(event.sound)) {
      throw new HttpsError("invalid-argument", `Invalid sound: ${event.sound}`);
    }
    if (!Number.isInteger(event.step) || event.step < 0) {
      throw new HttpsError("invalid-argument", "Event steps must be positive integers.");
    }
  }
  return payload as TrainingExamplePayload;
}

export const submitTrainingExample = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Sign in before uploading examples.");
  }
  const payload = assertTrainingPayload(request.data);
  const userPrefix = `training_uploads/${request.auth.uid}/`;
  if (!payload.storagePath.startsWith(userPrefix)) {
    throw new HttpsError(
      "permission-denied",
      "storagePath must live under the current user's upload folder."
    );
  }

  const doc = await db.collection("training_examples").add({
    ...payload,
    uid: request.auth.uid,
    createdAt: FieldValue.serverTimestamp(),
    schemaVersion: 1,
    reviewStatus: "pending",
  });

  return {id: doc.id};
});

export const indexTrainingUpload = onObjectFinalized(async (event) => {
  const object = event.data;
  const name = object.name;
  if (!name || !name.startsWith("training_uploads/")) return;

  await db.collection("training_uploads").add({
    bucket: object.bucket,
    storagePath: name,
    contentType: object.contentType ?? null,
    size: Number(object.size ?? 0),
    md5Hash: object.md5Hash ?? null,
    createdAt: FieldValue.serverTimestamp(),
  });

  logger.info("Indexed Spitty training upload", {storagePath: name});
});

export const getActiveModelManifest = onRequest(async (_request, response) => {
  const snapshot = await db.collection("model_releases").doc("active").get();
  if (!snapshot.exists) {
    response.status(404).json({error: "No active model release has been published."});
    return;
  }
  response.set("Cache-Control", "public, max-age=300");
  response.json(snapshot.data());
});
