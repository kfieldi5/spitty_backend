# Spitty Firebase Backend

This is a Firebase-first scaffold for the cloud side of Spitty's model-learning loop.

The intended flow:

1. The app records FTUE calibration audio or a custom beatbox recording.
2. The user verifies/corrects the generated grid.
3. The app uploads the original audio to Firebase Storage.
4. The app calls `submitTrainingExample` with the Storage path plus verified labels/grid metadata.
5. Cloud Functions writes a training manifest document into Firestore.
6. Offline training jobs can export these examples, retrain the universal model, and publish a new app-bundled model in a future release.

Why this shape:

- Local inference keeps the app immediate and musical.
- Local personalization can adapt to a single beatboxer's mouth sounds.
- Cloud collection improves the universal base model over time.
- Cloud-only inference remains possible later, but it adds latency, connectivity requirements, privacy questions, and removes some per-user feel unless we add user-specific server profiles.

## Functions

- `submitTrainingExample` — callable function for user-approved FTUE/correction metadata.
- `indexTrainingUpload` — Storage finalize trigger that records uploaded files.
- `getActiveModelManifest` — HTTP endpoint for future dynamic model-release metadata.

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

Deploy:

```sh
firebase deploy
```

Copy `.firebaserc.example` to `.firebaserc` and set your Firebase project id first.

