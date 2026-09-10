from __future__ import annotations

import unittest

from validation import stable_user_partition


class StableUserPartitionTest(unittest.TestCase):
    def test_never_splits_one_user_between_training_and_validation(self) -> None:
        examples = [
            {"id": f"{uid}-{index}", "uid": uid}
            for uid in ("a", "b", "c", "d", "e", "f")
            for index in range(3)
        ]

        training, validation = stable_user_partition(examples)

        training_users = {example["uid"] for example in training}
        validation_users = {example["uid"] for example in validation}
        self.assertTrue(training_users)
        self.assertTrue(validation_users)
        self.assertTrue(training_users.isdisjoint(validation_users))

    def test_requires_stable_user_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "stable user id"):
            stable_user_partition([{"id": "1", "uid": "unknown"}, {"id": "2"}])

    def test_enforces_minimum_distinct_users(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 distinct users"):
            stable_user_partition(
                [{"id": "1", "uid": "a"}, {"id": "2", "uid": "b"}],
                minimum_users=3,
            )


if __name__ == "__main__":
    unittest.main()
