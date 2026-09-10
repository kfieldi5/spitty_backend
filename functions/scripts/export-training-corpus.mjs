#!/usr/bin/env node

import {mkdir, writeFile} from "node:fs/promises";
import {resolve, relative} from "node:path";
import process from "node:process";

import {applicationDefault, initializeApp} from "firebase-admin/app";
import {getFirestore} from "firebase-admin/firestore";
import {getStorage} from "firebase-admin/storage";

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const projectId = argument("--project", "spitty-backend");
const bucketName = argument(
  "--bucket",
  "spitty-backend.firebasestorage.app"
);
const output = resolve(argument("--output", "./spitty-training-corpus"));
const throughValue = argument("--through", undefined);
const through = throughValue === undefined ? Number.MAX_SAFE_INTEGER : Number(throughValue);
if (!Number.isInteger(through) || through < 1) {
  throw new Error("--through must be a positive training sequence.");
}

initializeApp({credential: applicationDefault(), projectId, storageBucket: bucketName});
const db = getFirestore();
const bucket = getStorage().bucket();
await mkdir(output, {recursive: true});

const entries = [];
const snapshots = await db.collection("training_examples").get();
for (const snapshot of snapshots.docs) {
  const value = snapshot.data();
  if (
    value.kind !== "correction" ||
    value.trainingEligible !== true ||
    value.reviewStatus === "rejected" ||
    !Number.isInteger(value.trainingSequence) ||
    value.trainingSequence > through ||
    typeof value.uid !== "string" ||
    typeof value.storagePath !== "string" ||
    typeof value.samplesStoragePath !== "string"
  ) {
    continue;
  }
  const directory = resolve(output, "examples", value.uid, snapshot.id);
  await mkdir(directory, {recursive: true});
  const audio = resolve(directory, "recording.wav");
  const samples = resolve(directory, "samples.json.gz");
  await Promise.all([
    bucket.file(value.storagePath).download({destination: audio}),
    bucket.file(value.samplesStoragePath).download({destination: samples}),
  ]);
  entries.push({
    id: snapshot.id,
    uid: value.uid,
    trainingSequence: value.trainingSequence,
    verificationOutcome: value.verificationOutcome ?? "corrected",
    audio: relative(output, audio),
    samples: relative(output, samples),
    bpm: value.bpm,
    bars: value.bars,
    quantization: value.quantization,
    events: value.events,
    originalPattern: value.originalPattern,
    editSummary: value.editSummary,
  });
}
entries.sort((left, right) => left.trainingSequence - right.trainingSequence);
await writeFile(
  resolve(output, "manifest.json"),
  `${JSON.stringify({schemaVersion: 1, projectId, entries}, null, 2)}\n`
);
console.log(`Exported ${entries.length} consented examples to ${output}`);

