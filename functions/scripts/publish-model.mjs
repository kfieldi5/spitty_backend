import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import {basename, resolve} from "node:path";

import {applicationDefault, initializeApp} from "firebase-admin/app";
import {getFirestore} from "firebase-admin/firestore";
import {getStorage} from "firebase-admin/storage";

const expectedLabels = [
  "kick",
  "snare",
  "hihat_closed",
  "hihat_open",
  "rest",
];
const expectedShape = [1, 1, 64, 36];

function usage() {
  console.error(
    "Usage: npm run model:publish -- --model /path/model.pte " +
      "--manifest /path/manifest.json [--project spitty-backend] " +
      "[--bucket bucket-name] [--dry-run]"
  );
}

function parseArguments(values) {
  const parsed = {dryRun: false};
  for (let index = 0; index < values.length; index++) {
    const argument = values[index];
    if (argument === "--dry-run") {
      parsed.dryRun = true;
      continue;
    }
    if (!argument.startsWith("--") || index + 1 >= values.length) {
      usage();
      process.exit(2);
    }
    parsed[argument.slice(2)] = values[++index];
  }
  return parsed;
}

function parseVersion(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) throw new Error(`Invalid model_version: ${version}`);
  return match.slice(1).map(Number);
}

function compareVersions(left, right) {
  const a = parseVersion(left);
  const b = parseVersion(right);
  for (let index = 0; index < 3; index++) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return 0;
}

function sameArray(left, right) {
  return (
    Array.isArray(left) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function validateManifest(manifest, actualChecksum) {
  if (manifest.schema_version !== 1) {
    throw new Error("Only manifest schema_version 1 is supported.");
  }
  parseVersion(manifest.model_version);
  if (manifest.format !== "executorch_xnnpack") {
    throw new Error("format must be executorch_xnnpack.");
  }
  if (manifest.artifact !== "beatbox_classifier.pte") {
    throw new Error("artifact must be beatbox_classifier.pte.");
  }
  if (manifest.sha256?.toLowerCase() !== actualChecksum) {
    throw new Error(
      `Manifest SHA-256 ${manifest.sha256} does not match ${actualChecksum}.`
    );
  }
  if (!sameArray(manifest.labels, expectedLabels)) {
    throw new Error("Manifest labels are incompatible with the current app.");
  }
  if (
    manifest.input?.preprocessing !== "log_mel_v1_16khz_5600_samples" ||
    manifest.input?.dtype !== "float32" ||
    !sameArray(manifest.input?.shape, expectedShape)
  ) {
    throw new Error("Manifest input is incompatible with the current app.");
  }
  if (manifest.rhythm_bar_model !== undefined) {
    const rhythm = manifest.rhythm_bar_model;
    if (
      !rhythm ||
      typeof rhythm !== "object" ||
      Array.isArray(rhythm) ||
      rhythm.schema_version !== 1 ||
      !rhythm.overrides ||
      typeof rhythm.overrides !== "object" ||
      Array.isArray(rhythm.overrides)
    ) {
      throw new Error("rhythm_bar_model has an invalid schema.");
    }
    for (const [raw, corrected] of Object.entries(rhythm.overrides)) {
      const predicted = Number(raw);
      if (
        !Number.isInteger(predicted) ||
        predicted < 1 ||
        predicted > 32 ||
        !Number.isInteger(corrected) ||
        corrected < 1 ||
        corrected > 32
      ) {
        throw new Error("rhythm_bar_model contains an invalid override.");
      }
    }
  }
}

function primitiveMetrics(metrics) {
  if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
    return undefined;
  }
  const entries = Object.entries(metrics).filter(([, value]) =>
    ["string", "number", "boolean"].includes(typeof value)
  );
  return entries.length === 0 ? undefined : Object.fromEntries(entries);
}

function buildRuntimeRelease(manifest, storagePath) {
  const metrics = primitiveMetrics(manifest.metrics);
  return {
    schema_version: manifest.schema_version,
    model_version: manifest.model_version,
    format: manifest.format,
    artifact: manifest.artifact,
    sha256: manifest.sha256,
    input: manifest.input,
    labels: manifest.labels,
    storage_path: storagePath,
    published_at: new Date().toISOString(),
    ...(typeof manifest.created_at === "string" && {
      created_at: manifest.created_at,
    }),
    ...(typeof manifest.dataset === "string" && {dataset: manifest.dataset}),
    ...(Array.isArray(manifest.limitations) && {
      limitations: manifest.limitations,
    }),
    ...(metrics && {metrics}),
    ...(manifest.rhythm_bar_model && {
      rhythm_bar_model: manifest.rhythm_bar_model,
    }),
  };
}

function assertFirestoreCompatible(value, path = "document", insideArray = false) {
  if (Array.isArray(value)) {
    if (insideArray) {
      throw new Error(`${path} contains a nested array unsupported by Firestore.`);
    }
    value.forEach((item, index) =>
      assertFirestoreCompatible(item, `${path}[${index}]`, true)
    );
    return;
  }
  if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      assertFirestoreCompatible(child, `${path}.${key}`, false);
    }
  }
}

const args = parseArguments(process.argv.slice(2));
if (!args.model || !args.manifest) {
  usage();
  process.exit(2);
}

const projectId =
  args.project ??
  process.env.GOOGLE_CLOUD_PROJECT ??
  process.env.GCLOUD_PROJECT ??
  "spitty-backend";
const bucketName = args.bucket ?? `${projectId}.firebasestorage.app`;
const modelPath = resolve(args.model);
const manifestPath = resolve(args.manifest);
if (basename(modelPath) !== "beatbox_classifier.pte") {
  throw new Error("The model filename must be beatbox_classifier.pte.");
}

const [modelBytes, manifestBytes] = await Promise.all([
  readFile(modelPath),
  readFile(manifestPath, "utf8"),
]);
const checksum = createHash("sha256").update(modelBytes).digest("hex");
const manifest = JSON.parse(manifestBytes);
validateManifest(manifest, checksum);
const storagePath =
  `models/beatbox_classifier/${manifest.model_version}/` +
  "beatbox_classifier.pte";
const release = buildRuntimeRelease(manifest, storagePath);
const archivedRelease = {
  ...release,
  source_manifest_json: manifestBytes,
};
assertFirestoreCompatible(release);
assertFirestoreCompatible(archivedRelease);

if (args.dryRun) {
  console.log(
    `Validated Spitty model ${manifest.model_version} ` +
      `(${modelBytes.length} bytes, ${checksum}); ` +
      "runtime manifest is Firestore-compatible."
  );
  process.exit(0);
}

initializeApp({
  credential: applicationDefault(),
  projectId,
  storageBucket: bucketName,
});
const firestore = getFirestore();
const activeReference = firestore.collection("model_releases").doc("active");
const active = await activeReference.get();
const currentVersion = active.data()?.model_version;
if (
  currentVersion &&
  compareVersions(manifest.model_version, currentVersion) <= 0
) {
  throw new Error(
    `Active model is ${currentVersion}; refusing to publish ` +
      `${manifest.model_version}. Releases must always move forward.`
  );
}

await getStorage().bucket().upload(modelPath, {
  destination: storagePath,
  metadata: {
    contentType: "application/octet-stream",
    cacheControl: "public,max-age=31536000,immutable",
    metadata: {
      modelVersion: manifest.model_version,
      sha256: checksum,
    },
  },
});

const batch = firestore.batch();
batch.set(
  firestore.collection("model_releases").doc(manifest.model_version),
  archivedRelease
);
batch.set(activeReference, release);
await batch.commit();

console.log(
  `Published Spitty model ${manifest.model_version} to gs://${bucketName}/${storagePath}`
);
