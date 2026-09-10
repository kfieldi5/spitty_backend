#!/usr/bin/env python3
"""Claim, train, validate, and publish one queued Spitty model release."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import tempfile
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import firebase_admin
import torch
import torch.nn.functional as F
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from firebase_admin import credentials, firestore, storage
from torch.utils.data import DataLoader, TensorDataset

from model import BeatboxClassifier, Config, FeatureClassifier, LABELS
from rhythm import (
    evaluate_rhythm_bar_model,
    load_bar_examples,
    normalize_rhythm_bar_model,
    train_rhythm_bar_model,
)
from validation import distinct_users, stable_user_partition


FEATURE_SHAPE = (1, 64, 36)
FEATURE_COUNT = math.prod(FEATURE_SHAPE)
MAX_PACK_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_PACK_BYTES = 64 * 1024 * 1024
VERIFICATION_WEIGHTS = {"perfect": 1.0, "corrected": 1.5, "ftue": 1.0}
CORRECTION_OPERATION_WEIGHTS = {
    "confirmed": 1.0,
    "relabeled": 1.6,
    "moved": 1.3,
    "deleted": 1.6,
    # An added cell is valuable evidence that onset detection missed a sound,
    # but its synthesized clip center is less certain than a detected onset.
    "added": 0.7,
}
MIN_DISTINCT_USERS = int(os.environ.get("SPITTY_MIN_DISTINCT_USERS", "3"))
MIN_VALIDATION_EXAMPLES_PER_SOUND = int(
    os.environ.get("SPITTY_MIN_VALIDATION_EXAMPLES_PER_SOUND", "1")
)


class NoQueuedJob(RuntimeError):
    pass


class NoPromotion(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def initialize_firebase(project: str, bucket: str) -> None:
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(
        credentials.ApplicationDefault(),
        {"projectId": project, "storageBucket": bucket},
    )


def claim_job(db: Any) -> tuple[Any, dict[str, Any]]:
    candidates = list(
        db.collection("model_training_jobs")
        .where("status", "==", "queued")
        .limit(100)
        .stream()
    )
    if not candidates:
        raise NoQueuedJob("No queued model-training job.")
    candidate = min(
        candidates,
        key=lambda snapshot: int(
            (snapshot.to_dict() or {}).get("correctionEndInclusive", 0)
        ),
    )
    reference = candidate.reference
    transaction = db.transaction()

    @firestore.transactional
    def claim(current_transaction: Any) -> dict[str, Any] | None:
        snapshot = reference.get(transaction=current_transaction)
        data = snapshot.to_dict() or {}
        if data.get("status") != "queued":
            return None
        attempts = int(data.get("attempts", 0)) + 1
        current_transaction.update(
            reference,
            {
                "status": "running",
                "attempts": attempts,
                "startedAt": firestore.SERVER_TIMESTAMP,
                "runner": os.environ.get("GITHUB_RUN_ID", "local"),
            },
        )
        data["attempts"] = attempts
        return data

    data = claim(transaction)
    if data is None:
        raise NoQueuedJob("The queued job was claimed by another worker.")
    return reference, data


def load_sample_packs(
    db: Any,
    bucket: Any,
    through_sequence: int,
) -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []
    for snapshot in db.collection("training_examples").stream():
        document = snapshot.to_dict() or {}
        sequence = document.get("trainingSequence")
        path = document.get("samplesStoragePath")
        verification_outcome = document.get("verificationOutcome", "corrected")
        if (
            document.get("trainingEligible") is not True
            or document.get("reviewStatus") == "rejected"
            or not isinstance(sequence, int)
            or sequence > through_sequence
            or not isinstance(path, str)
            or verification_outcome not in VERIFICATION_WEIGHTS
        ):
            continue
        blob = bucket.blob(path)
        compressed = blob.download_as_bytes()
        if len(compressed) > MAX_PACK_BYTES:
            raise ValueError(f"Training sample pack is too large: {path}")
        decoded = gzip.decompress(compressed)
        if len(decoded) > MAX_UNCOMPRESSED_PACK_BYTES:
            raise ValueError(f"Expanded training sample pack is too large: {path}")
        payload = json.loads(decoded)
        if (
            payload.get("schema_version") not in (1, 2)
            or payload.get("preprocessing")
            != "log_mel_v1_16khz_5600_samples"
            or payload.get("shape") != [1, 1, 64, 36]
            or not isinstance(payload.get("samples"), list)
        ):
            raise ValueError(f"Training sample pack has an incompatible schema: {path}")
        pack_outcome = payload.get("verification_outcome")
        if pack_outcome is not None and pack_outcome != verification_outcome:
            raise ValueError(
                f"Training sample pack verification outcome does not match: {path}"
            )
        samples: list[tuple[list[float], int, str]] = []
        for sample in payload["samples"]:
            label_name = sample.get("label")
            features = sample.get("features")
            if label_name not in LABELS or not isinstance(features, list):
                raise ValueError(f"Training sample has an invalid label: {path}")
            if len(features) != FEATURE_COUNT:
                raise ValueError(f"Training sample has an invalid feature count: {path}")
            values = [float(value) for value in features]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Training sample contains non-finite features: {path}")
            operation = sample.get("operation", "confirmed")
            if operation not in CORRECTION_OPERATION_WEIGHTS:
                raise ValueError(
                    f"Training sample contains an invalid correction operation: {path}"
                )
            samples.append((values, LABELS.index(label_name), operation))
        if not samples:
            continue
        packs.append(
            {
                "id": snapshot.id,
                "uid": document.get("uid", "unknown"),
                "sequence": sequence,
                "verification_outcome": verification_outcome,
                "samples": samples,
            }
        )
    packs.sort(key=lambda pack: (pack["sequence"], pack["id"]))
    return packs


def stable_partition(packs: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    try:
        return stable_user_partition(
            packs,
            minimum_users=MIN_DISTINCT_USERS,
        )
    except ValueError as error:
        raise NoPromotion(str(error)) from error


def flatten_samples(
    packs: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [
        (
            sample[0],
            sample[1],
            VERIFICATION_WEIGHTS[pack["verification_outcome"]]
            * CORRECTION_OPERATION_WEIGHTS[sample[2]],
        )
        for pack in packs
        for sample in pack["samples"]
    ]
    if not examples:
        raise NoPromotion("No usable verified samples were found.")
    features = torch.tensor([item[0] for item in examples], dtype=torch.float32)
    labels = torch.tensor([item[1] for item in examples], dtype=torch.long)
    verification_weights = torch.tensor(
        [item[2] for item in examples], dtype=torch.float32
    )
    return features.reshape((-1, *FEATURE_SHAPE)), labels, verification_weights


def load_checkpoint(path: Path) -> tuple[BeatboxClassifier, Config, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    source = BeatboxClassifier(config)
    source.load_state_dict(checkpoint["state_dict"])
    return source, config, checkpoint


@torch.no_grad()
def evaluate(
    model: FeatureClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    logits = model(features)
    loss = F.cross_entropy(logits, labels).item()
    predictions = logits.argmax(1)
    accuracy = (predictions == labels).float().mean().item()
    per_label: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for index, name in enumerate(LABELS):
        mask = labels == index
        support = int(mask.sum())
        if support > 0:
            true_positives = int(((predictions == index) & mask).sum())
            false_positives = int(((predictions == index) & ~mask).sum())
            false_negatives = support - true_positives
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / support
            f1 = (
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            per_label[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
            f1_scores.append(f1)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_label": per_label,
    }


def train_candidate(
    baseline: FeatureClassifier,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_verification_weights: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    seed: int,
) -> tuple[FeatureClassifier, dict[str, Any], dict[str, Any]]:
    sound_classes = set(train_labels.tolist()) - {LABELS.index("rest")}
    if len(sound_classes) < 2:
        raise NoPromotion("Corrections must cover at least two drum classes.")
    torch.manual_seed(seed)
    random.seed(seed)
    candidate = copy.deepcopy(baseline)
    for parameter in candidate.parameters():
        parameter.requires_grad = False
    final_layer = candidate.classifier[-1]
    for parameter in final_layer.parameters():
        parameter.requires_grad = True
    original_weight = final_layer.weight.detach().clone()
    original_bias = final_layer.bias.detach().clone()
    counts = Counter(train_labels.tolist())
    weights = torch.tensor(
        [
            len(train_labels) / (len(LABELS) * max(counts.get(index, 0), 1))
            for index in range(len(LABELS))
        ],
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(final_layer.parameters(), lr=7e-4, weight_decay=1e-4)
    dataset = TensorDataset(
        train_features,
        train_labels,
        train_verification_weights,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(64, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    baseline_metrics = evaluate(
        baseline,
        validation_features,
        validation_labels,
    )
    for label in LABELS[:4]:
        support = int(baseline_metrics["per_label"].get(label, {}).get("support", 0))
        if support < MIN_VALIDATION_EXAMPLES_PER_SOUND:
            raise NoPromotion(
                f"Validation needs at least {MIN_VALIDATION_EXAMPLES_PER_SOUND} "
                f"{label} examples; found {support}."
            )
    best_state = copy.deepcopy(candidate.state_dict())
    best_metrics = evaluate(candidate, validation_features, validation_labels)
    for _ in range(48):
        candidate.train()
        for batch_features, batch_labels, batch_verification_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = candidate(batch_features)
            per_sample_loss = F.cross_entropy(
                logits,
                batch_labels,
                weight=weights,
                label_smoothing=0.02,
                reduction="none",
            )
            effective_weights = (
                weights[batch_labels] * batch_verification_weights
            )
            classification = (
                per_sample_loss * batch_verification_weights
            ).sum() / effective_weights.sum().clamp_min(1e-6)
            anchor = (
                (final_layer.weight - original_weight).pow(2).mean()
                + (final_layer.bias - original_bias).pow(2).mean()
            )
            (classification + anchor * 0.08).backward()
            optimizer.step()
        metrics = evaluate(candidate, validation_features, validation_labels)
        if (
            metrics["macro_f1"] > best_metrics["macro_f1"]
            or (
                metrics["macro_f1"] == best_metrics["macro_f1"]
                and (
                    metrics["accuracy"] > best_metrics["accuracy"]
                    or (
                        metrics["accuracy"] == best_metrics["accuracy"]
                        and metrics["loss"] < best_metrics["loss"]
                    )
                )
            )
        ):
            best_state = copy.deepcopy(candidate.state_dict())
            best_metrics = metrics
    candidate.load_state_dict(best_state)
    if best_metrics["macro_f1"] + 1e-6 < baseline_metrics["macro_f1"]:
        raise NoPromotion("Candidate validation macro-F1 regressed.")
    if (
        best_metrics["accuracy"] <= baseline_metrics["accuracy"] + 1e-6
        and best_metrics["loss"] >= baseline_metrics["loss"] * 0.995
    ):
        raise NoPromotion("Candidate did not improve validation accuracy or loss.")
    for label, baseline_label_metrics in baseline_metrics["per_label"].items():
        baseline_recall = float(baseline_label_metrics["recall"])
        candidate_recall = float(
            best_metrics["per_label"].get(label, {}).get("recall", 0.0)
        )
        if candidate_recall + 0.15 < baseline_recall:
            raise NoPromotion(f"Candidate regressed too far on {label}.")
    return candidate, baseline_metrics, best_metrics


def increment_patch(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Active model has an invalid version: {version}")
    return f"{int(parts[0])}.{int(parts[1])}.{int(parts[2]) + 1}"


def export_executorch(model: FeatureClassifier, output: Path) -> str:
    model = model.cpu().eval()
    exported = torch.export.export(model, (torch.zeros(1, 1, 64, 36),))
    program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    output.write_bytes(program.buffer)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def mark_job(reference: Any, status: str, **values: Any) -> None:
    reference.update(
        {
            "status": status,
            "finishedAt": firestore.SERVER_TIMESTAMP,
            **values,
        }
    )


def publish_candidate(
    db: Any,
    bucket: Any,
    job_reference: Any,
    job: dict[str, Any],
    source: BeatboxClassifier,
    config: Config,
    checkpoint_metadata: dict[str, Any],
    model: FeatureClassifier,
    baseline_metrics: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None,
    rhythm_bar_model: dict[str, Any],
    baseline_bar_metrics: dict[str, Any],
    candidate_bar_metrics: dict[str, Any],
    correction_count: int,
    included_example_count: int,
    verification_counts: dict[str, int],
    correction_operation_counts: dict[str, int],
    distinct_users_in_release: set[str],
    workdir: Path,
) -> str:
    active_reference = db.collection("model_releases").document("active")
    active = active_reference.get().to_dict() or {}
    base_version = active.get("model_version", "2.0.1")
    version = increment_patch(base_version)
    model_path = workdir / "beatbox_classifier.pte"
    checkpoint_path = workdir / "beatbox_classifier.pth"
    checksum = export_executorch(model, model_path)
    source.features.load_state_dict(model.features.state_dict())
    source.classifier.load_state_dict(model.classifier.state_dict())
    torch.save(
        {
            **checkpoint_metadata,
            "state_dict": source.state_dict(),
            "config": asdict(config),
            "labels": LABELS,
        },
        checkpoint_path,
    )
    prefix = f"models/beatbox_classifier/{version}"
    storage_model_path = f"{prefix}/beatbox_classifier.pte"
    storage_checkpoint_path = f"{prefix}/beatbox_classifier.pth"
    bucket.blob(storage_model_path).upload_from_filename(
        model_path,
        content_type="application/octet-stream",
    )
    bucket.blob(storage_checkpoint_path).upload_from_filename(
        checkpoint_path,
        content_type="application/octet-stream",
    )
    metrics = {
        "classifier_training_samples": correction_count,
        "distinct_contributing_users": len(distinct_users_in_release),
        "perfect_classifier_beats": verification_counts.get("perfect", 0),
        "corrected_classifier_beats": verification_counts.get("corrected", 0),
        "ftue_classifier_beats": verification_counts.get("ftue", 0),
        "correction_operations": correction_operation_counts,
        "rhythm_bar_examples": candidate_bar_metrics["example_count"],
        "rhythm_bar_accuracy": candidate_bar_metrics["accuracy"],
        "rhythm_bar_override_count": candidate_bar_metrics["override_count"],
    }
    if baseline_metrics is not None and candidate_metrics is not None:
        metrics.update(
            {
                "baseline_validation_accuracy": baseline_metrics["accuracy"],
                "candidate_validation_accuracy": candidate_metrics["accuracy"],
                "baseline_validation_macro_f1": baseline_metrics["macro_f1"],
                "candidate_validation_macro_f1": candidate_metrics["macro_f1"],
                "baseline_validation_loss": baseline_metrics["loss"],
                "candidate_validation_loss": candidate_metrics["loss"],
                "candidate_validation_per_label": candidate_metrics["per_label"],
            }
        )
    runtime_release = {
        "schema_version": 1,
        "model_version": version,
        "format": "executorch_xnnpack",
        "artifact": "beatbox_classifier.pte",
        "sha256": checksum,
        "input": {
            "shape": [1, 1, 64, 36],
            "preprocessing": "log_mel_v1_16khz_5600_samples",
            "dtype": "float32",
        },
        "labels": list(LABELS),
        "storage_path": storage_model_path,
        "training_checkpoint_path": storage_checkpoint_path,
        "published_at": utc_now().isoformat(),
        "dataset": "AVP_Dataset + consented verified Spitty beats",
        "metrics": metrics,
        "rhythm_bar_model": rhythm_bar_model,
        "limitations": [
            "Automatic continual training is validation-gated but early correction data may not represent all beatboxers.",
            "Crash is unsupported.",
        ],
    }
    archived_release = {
        **runtime_release,
        "training_job_id": job_reference.id,
        "included_example_count": included_example_count,
        "through_training_sequence": int(job["correctionEndInclusive"]),
    }
    transaction = db.transaction()

    @firestore.transactional
    def promote(current_transaction: Any) -> None:
        latest = active_reference.get(transaction=current_transaction).to_dict() or {}
        if latest.get("model_version") != base_version:
            raise RuntimeError("The active model changed while this candidate trained.")
        current_transaction.set(
            db.collection("model_releases").document(version),
            archived_release,
        )
        current_transaction.set(active_reference, runtime_release)
        current_transaction.update(
            job_reference,
            {
                "status": "completed",
                "finishedAt": firestore.SERVER_TIMESTAMP,
                "publishedModelVersion": version,
                "baselineMetrics": baseline_metrics,
                "candidateMetrics": candidate_metrics,
                "baselineBarMetrics": baseline_bar_metrics,
                "candidateBarMetrics": candidate_bar_metrics,
            },
        )

    promote(transaction)
    return version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="spitty-backend")
    parser.add_argument(
        "--bucket",
        default="spitty-backend.firebasestorage.app",
    )
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=Path(__file__).parent / "seed" / "beatbox_classifier.pth",
    )
    args = parser.parse_args()
    initialize_firebase(args.project, args.bucket)
    db = firestore.client()
    bucket = storage.bucket()
    try:
        job_reference, job = claim_job(db)
    except NoQueuedJob as error:
        print(error)
        return 0
    try:
        through_sequence = int(job["correctionEndInclusive"])
        packs = load_sample_packs(db, bucket, through_sequence)
        bar_examples = load_bar_examples(db, through_sequence)
        active = db.collection("model_releases").document("active").get().to_dict() or {}
        baseline_bar_model = normalize_rhythm_bar_model(
            active.get("rhythm_bar_model")
        )
        baseline_bar_metrics = evaluate_rhythm_bar_model(baseline_bar_model, [])
        candidate_bar_model = baseline_bar_model
        candidate_bar_metrics = baseline_bar_metrics
        bar_model_improved = False
        try:
            bar_training, bar_validation = stable_user_partition(
                bar_examples,
                minimum_users=MIN_DISTINCT_USERS,
            )
            candidate_bar_model = train_rhythm_bar_model(bar_training)
            baseline_bar_metrics = evaluate_rhythm_bar_model(
                baseline_bar_model, bar_validation
            )
            candidate_bar_metrics = evaluate_rhythm_bar_model(
                candidate_bar_model, bar_validation
            )
            bar_model_improved = (
                candidate_bar_model["overrides"]
                != baseline_bar_model["overrides"]
                and candidate_bar_metrics["accuracy"]
                > baseline_bar_metrics["accuracy"] + 1e-9
            )
        except ValueError:
            # Rhythm overrides need the same user-disjoint evidence as the
            # classifier. A small single-user batch remains collected but is
            # not permitted to change the universal model.
            pass
        selected_bar_model = (
            candidate_bar_model if bar_model_improved else baseline_bar_model
        )
        selected_bar_metrics = (
            candidate_bar_metrics if bar_model_improved else baseline_bar_metrics
        )
        checkpoint_storage_path = active.get("training_checkpoint_path")
        with tempfile.TemporaryDirectory(prefix="spitty-training-") as directory:
            workdir = Path(directory)
            checkpoint_path = workdir / "base.pth"
            if isinstance(checkpoint_storage_path, str):
                bucket.blob(checkpoint_storage_path).download_to_filename(checkpoint_path)
            else:
                checkpoint_path.write_bytes(args.seed_checkpoint.read_bytes())
            source, config, checkpoint_metadata = load_checkpoint(checkpoint_path)
            baseline = FeatureClassifier(source).eval()
            candidate = baseline
            baseline_metrics = None
            candidate_metrics = None
            correction_count = 0
            sound_model_improved = False
            sound_no_promotion_reason = "Not enough verified classifier packs."
            try:
                training_packs, validation_packs = stable_partition(packs)
                (
                    training_features,
                    training_labels,
                    training_verification_weights,
                ) = flatten_samples(training_packs)
                validation_features, validation_labels, _ = flatten_samples(
                    validation_packs
                )
                candidate, baseline_metrics, candidate_metrics = train_candidate(
                    baseline,
                    training_features,
                    training_labels,
                    training_verification_weights,
                    validation_features,
                    validation_labels,
                    seed=config.seed + through_sequence,
                )
                correction_count = len(training_labels)
                sound_model_improved = True
            except NoPromotion as error:
                sound_no_promotion_reason = str(error)
            if not sound_model_improved and not bar_model_improved:
                raise NoPromotion(
                    f"Classifier: {sound_no_promotion_reason} "
                    "Rhythm bar model did not improve."
                )
            version = publish_candidate(
                db,
                bucket,
                job_reference,
                job,
                source,
                config,
                checkpoint_metadata,
                candidate,
                baseline_metrics,
                candidate_metrics,
                selected_bar_model,
                baseline_bar_metrics,
                selected_bar_metrics,
                correction_count=correction_count,
                included_example_count=max(len(packs), len(bar_examples)),
                verification_counts=dict(
                    Counter(pack["verification_outcome"] for pack in packs)
                ),
                correction_operation_counts=dict(
                    Counter(
                        operation
                        for pack in packs
                        for _, _, operation in pack["samples"]
                    )
                ),
                distinct_users_in_release=distinct_users([*packs, *bar_examples]),
                workdir=workdir,
            )
        print(f"Published validation-gated Spitty model {version}.")
        return 0
    except NoPromotion as error:
        mark_job(job_reference, "completed_no_promotion", reason=str(error))
        print(f"Training completed without promotion: {error}")
        return 0
    except Exception as error:
        attempts = int(job.get("attempts", 1))
        status = "queued" if attempts < 3 else "failed"
        job_reference.update(
            {
                "status": status,
                "lastError": str(error)[:2000],
                "lastTraceback": traceback.format_exc()[-8000:],
                "finishedAt": firestore.SERVER_TIMESTAMP,
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
