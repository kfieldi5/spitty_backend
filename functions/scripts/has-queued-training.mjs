import {applicationDefault, initializeApp} from "firebase-admin/app";
import {getFirestore} from "firebase-admin/firestore";

const projectId =
  process.env.GOOGLE_CLOUD_PROJECT ??
  process.env.GCLOUD_PROJECT ??
  "spitty-backend";

initializeApp({credential: applicationDefault(), projectId});
const queued = await getFirestore()
  .collection("model_training_jobs")
  .where("status", "==", "queued")
  .limit(1)
  .get();

const value = queued.empty ? "false" : "true";
if (process.env.GITHUB_OUTPUT) {
  const {appendFile} = await import("node:fs/promises");
  await appendFile(process.env.GITHUB_OUTPUT, `queued=${value}\n`);
}
console.log(value);
