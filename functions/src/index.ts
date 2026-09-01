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
  predictedBars?: number;
  rawPredictedBars?: number;
  quantization: "1/2" | "1/4" | "1/8" | "1/16";
  events: GridEvent[];
  appVersion?: string;
  localModelVersion?: string;
  notes?: string;
  samplesStoragePath?: string;
  sampleCount?: number;
};

const defaultCorrectionsPerTrainingRun = 5;

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
  for (const [name, value] of [
    ["predictedBars", payload.predictedBars],
    ["rawPredictedBars", payload.rawPredictedBars],
  ] as const) {
    if (
      value !== undefined &&
      (!Number.isInteger(value) || value < 1 || value > 32)
    ) {
      throw new HttpsError(
        "invalid-argument",
        `${name} must be between 1 and 32.`
      );
    }
  }
  if (
    payload.rawPredictedBars !== undefined &&
    payload.predictedBars === undefined
  ) {
    throw new HttpsError(
      "invalid-argument",
      "rawPredictedBars requires predictedBars."
    );
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
  if (payload.samplesStoragePath !== undefined) {
    if (
      typeof payload.samplesStoragePath !== "string" ||
      !payload.samplesStoragePath.startsWith("training_uploads/") ||
      !payload.samplesStoragePath.endsWith(".samples.json.gz")
    ) {
      throw new HttpsError(
        "invalid-argument",
        "samplesStoragePath must point to a compressed correction sample pack."
      );
    }
    if (
      !Number.isInteger(payload.sampleCount) ||
      (payload.sampleCount ?? 0) < 1 ||
      (payload.sampleCount ?? 0) > 256
    ) {
      throw new HttpsError(
        "invalid-argument",
        "sampleCount must be between 1 and 256 when samplesStoragePath is set."
      );
    }
  } else if (payload.sampleCount !== undefined) {
    throw new HttpsError(
      "invalid-argument",
      "sampleCount requires samplesStoragePath."
    );
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
  if (
    payload.samplesStoragePath &&
    !payload.samplesStoragePath.startsWith(userPrefix)
  ) {
    throw new HttpsError(
      "permission-denied",
      "samplesStoragePath must live under the current user's upload folder."
    );
  }

  const example = db.collection("training_examples").doc();
  const config = db.collection("model_training").doc("config");
  const state = db.collection("model_training").doc("state");

  await db.runTransaction(async (transaction) => {
    const configSnapshot = await transaction.get(config);
    const stateSnapshot = await transaction.get(state);
    const configuredThreshold = configSnapshot.get("correctionsPerRun");
    const correctionsPerRun =
      Number.isInteger(configuredThreshold) && configuredThreshold > 0
        ? configuredThreshold
        : defaultCorrectionsPerTrainingRun;
    const classifierTrainingEligible =
      payload.kind === "correction" && Boolean(payload.samplesStoragePath);
    const barLengthTrainingEligible =
      payload.kind === "correction" &&
      Number.isInteger(payload.predictedBars) &&
      Number.isInteger(payload.rawPredictedBars);
    const eligible = classifierTrainingEligible || barLengthTrainingEligible;
    const previousTotal = stateSnapshot.get("totalEligibleCorrections");
    const previousQueued = stateSnapshot.get("lastQueuedAtCount");
    const totalEligibleCorrections =
      (Number.isInteger(previousTotal) ? previousTotal : 0) + (eligible ? 1 : 0);
    let lastQueuedAtCount = Number.isInteger(previousQueued) ? previousQueued : 0;

    transaction.set(example, {
      ...payload,
      uid: request.auth!.uid,
      createdAt: FieldValue.serverTimestamp(),
      schemaVersion: 3,
      reviewStatus: "pending",
      trainingEligible: eligible,
      classifierTrainingEligible,
      barLengthTrainingEligible,
      ...(eligible && {trainingSequence: totalEligibleCorrections}),
    });

    if (!configSnapshot.exists) {
      transaction.set(config, {
        correctionsPerRun,
        updatedAt: FieldValue.serverTimestamp(),
      });
    }

    if (eligible && totalEligibleCorrections - lastQueuedAtCount >= correctionsPerRun) {
      const job = db.collection("model_training_jobs").doc();
      transaction.set(job, {
        status: "queued",
        correctionStartExclusive: lastQueuedAtCount,
        correctionEndInclusive: totalEligibleCorrections,
        correctionsPerRun,
        throughExampleId: example.id,
        createdAt: FieldValue.serverTimestamp(),
        attempts: 0,
      });
      lastQueuedAtCount = totalEligibleCorrections;
    }

    transaction.set(
      state,
      {
        totalEligibleCorrections,
        lastQueuedAtCount,
        updatedAt: FieldValue.serverTimestamp(),
      },
      {merge: true}
    );
  });

  return {id: example.id};
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
