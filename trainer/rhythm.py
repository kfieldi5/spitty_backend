from __future__ import annotations

from collections import Counter
from typing import Any


MIN_BAR_OVERRIDE_EXAMPLES = 3
MIN_BAR_OVERRIDE_CONSENSUS = 2 / 3


def load_bar_examples(db: Any, through_sequence: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for snapshot in db.collection("training_examples").stream():
        document = snapshot.to_dict() or {}
        sequence = document.get("trainingSequence")
        raw_predicted = document.get("rawPredictedBars")
        predicted = document.get("predictedBars")
        corrected = document.get("bars")
        if (
            document.get("kind") != "correction"
            or document.get("reviewStatus") == "rejected"
            or not isinstance(sequence, int)
            or sequence > through_sequence
            or not isinstance(raw_predicted, int)
            or not isinstance(predicted, int)
            or not isinstance(corrected, int)
            or not all(
                1 <= value <= 32
                for value in (raw_predicted, predicted, corrected)
            )
        ):
            continue
        examples.append(
            {
                "id": snapshot.id,
                "uid": document.get("uid", "unknown"),
                "raw_predicted": raw_predicted,
                "predicted": predicted,
                "corrected": corrected,
            }
        )
    return examples


def normalize_rhythm_bar_model(value: Any) -> dict[str, Any]:
    overrides: dict[str, int] = {}
    if isinstance(value, dict) and isinstance(value.get("overrides"), dict):
        for raw, corrected in value["overrides"].items():
            try:
                raw_bars = int(raw)
            except (TypeError, ValueError):
                continue
            if (
                1 <= raw_bars <= 32
                and isinstance(corrected, int)
                and 1 <= corrected <= 32
            ):
                overrides[str(raw_bars)] = corrected
    return {
        "schema_version": 1,
        "overrides": overrides,
        "trained_example_count": int(
            value.get("trained_example_count", 0)
            if isinstance(value, dict)
            else 0
        ),
    }


def train_rhythm_bar_model(examples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, Counter[int]] = {}
    for example in examples:
        grouped.setdefault(example["raw_predicted"], Counter()).update(
            [example["corrected"]]
        )
    overrides: dict[str, int] = {}
    for raw_predicted, counts in grouped.items():
        total = sum(counts.values())
        corrected, votes = max(
            counts.items(),
            key=lambda item: (item[1], item[0] == raw_predicted, -item[0]),
        )
        if (
            total >= MIN_BAR_OVERRIDE_EXAMPLES
            and votes / total >= MIN_BAR_OVERRIDE_CONSENSUS
            and corrected != raw_predicted
        ):
            overrides[str(raw_predicted)] = corrected
    return {
        "schema_version": 1,
        "overrides": overrides,
        "trained_example_count": len(examples),
    }


def evaluate_rhythm_bar_model(
    model: dict[str, Any], examples: list[dict[str, Any]]
) -> dict[str, Any]:
    overrides = model.get("overrides", {})
    correct = 0
    initial_correct = 0
    for example in examples:
        raw = example["raw_predicted"]
        corrected = example["corrected"]
        prediction = overrides.get(str(raw), raw)
        correct += int(prediction == corrected)
        initial_correct += int(example["predicted"] == corrected)
    count = len(examples)
    return {
        "example_count": count,
        "accuracy": correct / count if count else 0.0,
        "displayed_prediction_accuracy": initial_correct / count if count else 0.0,
        "override_count": len(overrides),
    }
