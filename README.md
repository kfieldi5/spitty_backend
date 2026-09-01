# Spitty Firebase Backend

This is a Firebase-first scaffold for the cloud side of Spitty's model-learning loop.

The `public/` directory contains the static Spitty landing, support, and privacy
pages deployed through Firebase Hosting. The default hosted URL is
`https://spitty-backend.web.app`; `spitty.app` can be attached as a custom domain
in the Firebase console.

The intended flow:

1. The app records FTUE calibration audio or a custom beatbox recording.
2. The user verifies/corrects the generated grid.
3. The app uploads the original audio to Firebase Storage.
4. The app calls `submitTrainingExample` with the Storage path plus verified labels/grid metadata.
5. Cloud Functions writes a training manifest document into Firestore.
6. Every configurable batch of eligible corrections queues a training job.
7. GitHub Actions fine-tunes the classifier head and learns loop-length
   corrections from the raw bar estimate, the estimate shown to the user, and
   the final verified bar count.
8. A verified ExecuTorch release is published to Firebase Storage and promoted
   through the `model_releases/active` Firestore document.
9. The app checks that active release, installs it only when it is newer and
   checksum-valid, then continues applying its user-specific personalization.

Why this shape:

- Local inference keeps the app immediate and musical.
- Local personalization can adapt to a single beatboxer's mouth sounds.
- Cloud collection improves the universal base model over time.
- Cloud-only inference remains possible later, but it adds latency, connectivity requirements, privacy questions, and removes some per-user feel unless we add user-specific server profiles.

## Functions

- `submitTrainingExample` — callable function for user-approved FTUE/correction metadata.
- `indexTrainingUpload` — Storage finalize trigger that records uploaded files.
- `getActiveModelManifest` — public HTTP endpoint for active dynamic
  model-release metadata.

## Automatic retraining

Only correction uploads containing an exact labeled feature pack are eligible
for classifier training. New app builds create that pack from the same aligned
transients used for local personalization, so the cloud worker never has to
guess which transient belongs to which grid cell. Corrections also include
`rawPredictedBars`, `predictedBars`, and the final `bars`. Those labels are
eligible for rhythm training even when no transient feature pack can be made.

The rhythm trainer learns conservative bar-count overrides only after at least
three examples agree by a two-thirds majority. For example, repeated verified
`16 -> 4` corrections teach future releases to turn the same raw 16-bar
heuristic result into four bars. The learned `rhythm_bar_model` is published in
the remote model manifest and applied before the grid is generated. A rhythm
improvement can publish a new release even when the transient classifier does
not change.

The default threshold is five eligible corrected beats. `submitTrainingExample`
increments the counter transactionally and creates a `model_training_jobs`
document whenever the threshold is reached. The `Retrain remote model` GitHub
Actions workflow checks for queued jobs every 15 minutes. It:

1. Claims one queued correction batch.
2. Downloads every eligible feature pack through that batch.
3. Holds back a deterministic subset of complete beats for validation.
4. Fine-tunes only the final classifier layer with an anchor penalty to limit
   catastrophic drift.
5. Rejects candidates that do not improve validation accuracy or loss, or that
   regress too far on a represented class.
6. Exports a checksum-verified ExecuTorch model, increments the active patch
   version, uploads its trainable checkpoint, and atomically promotes it.

Tune the threshold without changing or redeploying the app:

```sh
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
cd functions
npm run model:threshold -- --count 5
```

The same `FIREBASE_SERVICE_ACCOUNT_SPITTY_BACKEND` GitHub secret used by backend
deployment is used by the training workflow. Its service account needs read and
write access to Firestore and Firebase Storage. Scheduled workflows run only
after `.github/workflows/retrain_model.yml` exists on the repository's default
branch. The workflow can also be started immediately from GitHub Actions with
`Run workflow`.

Training jobs finish as `completed`, `completed_no_promotion`, or `failed` in
Firestore. A completed job's `publishedModelVersion` identifies the promoted
release. A no-promotion result is expected when five corrections do not yet
provide enough class coverage or measurable held-out improvement.

## Publish a base model

The app accepts only manifests that match its current ExecuTorch input and label
contract. The publish command validates that contract and the SHA-256 before it
uploads or changes the active release.

The active Firestore document contains only runtime-compatible fields. Complete
training metadata—including confusion matrices that Firestore cannot represent
as nested arrays—is retained as JSON on the versioned release document.

Authenticate with a service account that can write Firebase Storage and
Firestore, then run:

```sh
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
cd functions
npm run model:publish -- \
  --model /absolute/path/to/beatbox_classifier.pte \
  --manifest /absolute/path/to/manifest.json
```

For the current project the command defaults to project `spitty-backend` and
bucket `spitty-backend.firebasestorage.app`. Pass `--project` or `--bucket` to
override either value.

Publishing creates these resources:

```txt
Storage:  models/beatbox_classifier/<version>/beatbox_classifier.pte
Firestore: model_releases/<version>
Firestore: model_releases/active
```

Versions use `major.minor.patch`. A release must be strictly newer than the
current active version. To roll back bad weights, republish the last known-good
weights under a new, higher patch version so every installed app moves forward.

To validate files without connecting to Firebase, append `--dry-run`.

On-device behavior:

- A remote check runs after startup and is throttled to once every 12 hours.
- A failed request, incompatible manifest, or bad checksum leaves the current
  downloaded model (or bundled model) active and will be retried later.
- Valid downloaded releases are retained side by side, preserving rollback.
- FTUE and correction prototypes remain local and are applied after whichever
  verified base model is active.

## Setup

Install dependencies:

```sh
cd functions
npm install
```

Build:

```sh
npm run build
```

Deploy everything, including the static website:

```sh
firebase deploy
```

Deploy only the website:

```sh
firebase deploy --project spitty-backend --only hosting
```

Copy `.firebaserc.example` to `.firebaserc` and set your Firebase project id first.

## GitHub Actions deployment

Pushes to `main` deploy the backend automatically via:

```txt
.github/workflows/deploy_backend.yml
```

Pull requests and pushes to `main` also run:

```txt
.github/workflows/backend_ci.yml
```

Add this GitHub secret:

```txt
FIREBASE_SERVICE_ACCOUNT_SPITTY_BACKEND
```

Its value should be the raw JSON for a Google Cloud service account that can
deploy Firebase Functions, Firestore rules, Storage rules, and Hosting for
project `spitty-backend`.

Minimum practical roles:

- Firebase Admin
- Cloud Functions Admin
- Cloud Run Admin
- Cloud Build Editor
- Service Account User
- Artifact Registry Admin
- Eventarc Admin
- Pub/Sub Admin
- Firebase Rules Admin
- Storage Admin

You can tighten these later, but this gets CI deployment unblocked.
