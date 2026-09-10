import {initializeApp} from "firebase-admin/app";
import {FieldValue, getFirestore} from "firebase-admin/firestore";
import {onCall, onRequest, HttpsError} from "firebase-functions/v2/https";
import {onObjectFinalized} from "firebase-functions/v2/storage";
import {logger} from "firebase-functions";

initializeApp();

const db = getFirestore();

type DrumSound = "kick" | "snare" | "closedHat" | "openHat" | "rest";
type Quantization = "1/2" | "1/4" | "1/8" | "1/16";
type VerificationOutcome = "perfect" | "corrected" | "ftue";

type GridEvent = {
  sound: DrumSound;
  step: number;
  confidence?: number;
};

type PatternSnapshot = {
  bpm: number;
  bars: number;
  quantization: Quantization;
  events: GridEvent[];
};

type TrainingExamplePayload = {
  kind: "ftue" | "correction";
  storagePath: string;
  bpm: number;
  bars: number;
  predictedBars?: number;
  rawPredictedBars?: number;
  quantization: Quantization;
  events: GridEvent[];
  verificationOutcome?: VerificationOutcome;
  originalPattern?: PatternSnapshot;
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
const quantizations = new Set<Quantization>(["1/2", "1/4", "1/8", "1/16"]);
const verificationOutcomes = new Set<VerificationOutcome>([
  "perfect",
  "corrected",
  "ftue",
]);

function validatedEvents(value: unknown, field: string): GridEvent[] {
  if (!Array.isArray(value)) {
    throw new HttpsError("invalid-argument", `${field} must be an array.`);
  }
  for (const rawEvent of value) {
    if (!rawEvent || typeof rawEvent !== "object") {
      throw new HttpsError("invalid-argument", `${field} contains an invalid event.`);
    }
    const event = rawEvent as Partial<GridEvent>;
    if (!sounds.has(event.sound as DrumSound)) {
      throw new HttpsError("invalid-argument", `Invalid sound: ${event.sound}`);
    }
    if (!Number.isInteger(event.step) || (event.step ?? -1) < 0) {
      throw new HttpsError(
        "invalid-argument",
        `${field} event steps must be non-negative integers.`
      );
    }
    if (
      event.confidence !== undefined &&
      (typeof event.confidence !== "number" ||
        !Number.isFinite(event.confidence) ||
        event.confidence < 0 ||
        event.confidence > 1)
    ) {
      throw new HttpsError(
        "invalid-argument",
        `${field} event confidence must be between zero and one.`
      );
    }
  }
  return value as GridEvent[];
}

function validatedPatternSnapshot(value: unknown): PatternSnapshot {
  if (!value || typeof value !== "object") {
    throw new HttpsError("invalid-argument", "originalPattern must be an object.");
  }
  const pattern = value as Partial<PatternSnapshot>;
  if (
    !Number.isInteger(pattern.bpm) ||
    (pattern.bpm ?? 0) < 40 ||
    (pattern.bpm ?? 0) > 220
  ) {
    throw new HttpsError(
      "invalid-argument",
      "originalPattern.bpm must be between 40 and 220."
    );
  }
  if (
    !Number.isInteger(pattern.bars) ||
    (pattern.bars ?? 0) < 1 ||
    (pattern.bars ?? 0) > 32
  ) {
    throw new HttpsError(
      "invalid-argument",
      "originalPattern.bars must be between 1 and 32."
    );
  }
  if (!quantizations.has(pattern.quantization as Quantization)) {
    throw new HttpsError(
      "invalid-argument",
      "originalPattern.quantization is invalid."
    );
  }
  return {
    bpm: pattern.bpm as number,
    bars: pattern.bars as number,
    quantization: pattern.quantization as Quantization,
    events: validatedEvents(pattern.events, "originalPattern.events"),
  };
}

function summarizePatternEdit(original: PatternSnapshot, final: PatternSnapshot) {
  const remainingOriginal = original.events.map((event) => ({...event}));
  const remainingFinal = final.events.map((event) => ({...event}));
  const operations: Record<string, unknown>[] = [];

  // Remove exact matches first. They are useful confirmed positives but not
  // correction operations.
  for (let index = remainingOriginal.length - 1; index >= 0; index--) {
    const event = remainingOriginal[index];
    const match = remainingFinal.findIndex(
      (candidate) => candidate.step === event.step && candidate.sound === event.sound
    );
    if (match >= 0) {
      remainingOriginal.splice(index, 1);
      remainingFinal.splice(match, 1);
    }
  }

  // A different sound in the same cell is a class-label correction.
  for (let index = remainingOriginal.length - 1; index >= 0; index--) {
    const event = remainingOriginal[index];
    const match = remainingFinal.findIndex((candidate) => candidate.step === event.step);
    if (match >= 0) {
      const replacement = remainingFinal[match];
      operations.push({
        type: "relabel",
        step: event.step,
        fromSound: event.sound,
        toSound: replacement.sound,
      });
      remainingOriginal.splice(index, 1);
      remainingFinal.splice(match, 1);
    }
  }

  // Match the nearest remaining occurrence of the same sound as a timing move.
  while (true) {
    let best: {original: number; final: number; distance: number} | undefined;
    for (let originalIndex = 0; originalIndex < remainingOriginal.length; originalIndex++) {
      for (let finalIndex = 0; finalIndex < remainingFinal.length; finalIndex++) {
        if (remainingOriginal[originalIndex].sound !== remainingFinal[finalIndex].sound) {
          continue;
        }
        const distance = Math.abs(
          remainingOriginal[originalIndex].step - remainingFinal[finalIndex].step
        );
        if (!best || distance < best.distance) {
          best = {original: originalIndex, final: finalIndex, distance};
        }
      }
    }
    if (!best) break;
    const event = remainingOriginal[best.original];
    const replacement = remainingFinal[best.final];
    operations.push({
      type: "move",
      sound: event.sound,
      fromStep: event.step,
      toStep: replacement.step,
    });
    remainingOriginal.splice(best.original, 1);
    remainingFinal.splice(best.final, 1);
  }

  for (const event of remainingOriginal) {
    operations.push({type: "delete", step: event.step, sound: event.sound});
  }
  for (const event of remainingFinal) {
    operations.push({type: "add", step: event.step, sound: event.sound});
  }

  const addedEvents = remainingFinal.length;
  const removedEvents = remainingOriginal.length;
  const relabeledEvents = operations.filter((operation) => operation.type === "relabel").length;
  const movedEvents = operations.filter((operation) => operation.type === "move").length;
  const bpmChanged = original.bpm !== final.bpm;
  const barsChanged = original.bars !== final.bars;
  const quantizationChanged = original.quantization !== final.quantization;
  return {
    exactMatch:
      addedEvents === 0 &&
      removedEvents === 0 &&
      relabeledEvents === 0 &&
      movedEvents === 0 &&
      !bpmChanged &&
      !barsChanged &&
      !quantizationChanged,
    addedEvents,
    removedEvents,
    relabeledEvents,
    movedEvents,
    bpmChanged,
    barsChanged,
    quantizationChanged,
    operations,
  };
}

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
  if (
    !Number.isInteger(payload.bpm) ||
    (payload.bpm ?? 0) < 40 ||
    (payload.bpm ?? 0) > 220
  ) {
    throw new HttpsError("invalid-argument", "bpm must be between 40 and 220.");
  }
  if (
    !Number.isInteger(payload.bars) ||
    (payload.bars ?? 0) < 1 ||
    (payload.bars ?? 0) > 32
  ) {
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
  if (!quantizations.has(payload.quantization as Quantization)) {
    throw new HttpsError("invalid-argument", "Invalid quantization.");
  }
  payload.events = validatedEvents(payload.events, "events");
  if (
    payload.verificationOutcome !== undefined &&
    !verificationOutcomes.has(payload.verificationOutcome)
  ) {
    throw new HttpsError("invalid-argument", "Invalid verificationOutcome.");
  }
  if (payload.kind === "ftue") {
    if (
      (payload.verificationOutcome !== undefined &&
        payload.verificationOutcome !== "ftue") ||
      payload.originalPattern !== undefined
    ) {
      throw new HttpsError(
        "invalid-argument",
        "FTUE examples may only use the ftue verification outcome."
      );
    }
    payload.verificationOutcome = "ftue";
  } else {
    payload.verificationOutcome ??= "corrected";
    if (payload.originalPattern !== undefined) {
      payload.originalPattern = validatedPatternSnapshot(
        payload.originalPattern
      );
    }
    if (
      payload.verificationOutcome === "ftue"
    ) {
      throw new HttpsError(
        "invalid-argument",
        "Correction examples cannot use the ftue verification outcome."
      );
    }
    if (
      payload.verificationOutcome === "perfect" &&
      payload.originalPattern === undefined
    ) {
      throw new HttpsError(
        "invalid-argument",
        "Perfect confirmations require originalPattern."
      );
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
        "samplesStoragePath must point to a compressed training sample pack."
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
  const finalPattern: PatternSnapshot = {
    bpm: payload.bpm,
    bars: payload.bars,
    quantization: payload.quantization,
    events: payload.events,
  };
  const editSummary = payload.originalPattern
    ? summarizePatternEdit(payload.originalPattern, finalPattern)
    : undefined;
  if (payload.verificationOutcome === "perfect" && !editSummary?.exactMatch) {
    throw new HttpsError(
      "invalid-argument",
      "A perfect confirmation must exactly match the original prediction."
    );
  }
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
      Boolean(payload.samplesStoragePath);
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
      ...(editSummary && {editSummary}),
      uid: request.auth!.uid,
      createdAt: FieldValue.serverTimestamp(),
      schemaVersion: 4,
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
