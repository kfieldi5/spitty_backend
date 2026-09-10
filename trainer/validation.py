from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def stable_user_partition(
    examples: list[dict[str, Any]],
    *,
    minimum_users: int = 2,
    validation_modulo: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic holdout without leaking one user across splits."""

    by_uid: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        uid = example.get("uid")
        if not isinstance(uid, str) or not uid or uid == "unknown":
            raise ValueError("Every training example must have a stable user id.")
        by_uid.setdefault(uid, []).append(example)
    if len(by_uid) < minimum_users:
        raise ValueError(
            f"At least {minimum_users} distinct users are required; "
            f"found {len(by_uid)}."
        )

    def bucket(uid: str) -> int:
        return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)

    validation_uids = {
        uid for uid in by_uid if bucket(uid) % validation_modulo == 0
    }
    if not validation_uids:
        validation_uids = {min(by_uid, key=lambda uid: (bucket(uid), uid))}
    if len(validation_uids) == len(by_uid):
        validation_uids.remove(max(validation_uids, key=lambda uid: (bucket(uid), uid)))

    training = [
        example for uid, values in by_uid.items() if uid not in validation_uids
        for example in values
    ]
    validation = [
        example for uid, values in by_uid.items() if uid in validation_uids
        for example in values
    ]
    if not training or not validation:
        raise ValueError("Both user-disjoint partitions must contain examples.")
    return training, validation


def distinct_users(examples: Iterable[dict[str, Any]]) -> set[str]:
    return {
        uid
        for example in examples
        if isinstance((uid := example.get("uid")), str) and uid and uid != "unknown"
    }
