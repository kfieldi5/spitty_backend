import {applicationDefault, initializeApp} from "firebase-admin/app";
import {FieldValue, getFirestore} from "firebase-admin/firestore";

function parseCount(values) {
  const countIndex = values.indexOf("--count");
  const value = countIndex === -1 ? undefined : Number(values[countIndex + 1]);
  if (!Number.isInteger(value) || value < 1 || value > 10000) {
    throw new Error("Usage: npm run model:threshold -- --count <1-10000>");
  }
  return value;
}

const count = parseCount(process.argv.slice(2));
const projectId =
  process.env.GOOGLE_CLOUD_PROJECT ??
  process.env.GCLOUD_PROJECT ??
  "spitty-backend";

initializeApp({credential: applicationDefault(), projectId});
await getFirestore().collection("model_training").doc("config").set(
  {
    correctionsPerRun: count,
    updatedAt: FieldValue.serverTimestamp(),
  },
  {merge: true}
);

console.log(`Spitty will queue a model-training run every ${count} corrections.`);
