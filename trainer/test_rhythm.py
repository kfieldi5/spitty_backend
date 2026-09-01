import unittest

from rhythm import evaluate_rhythm_bar_model, train_rhythm_bar_model


class RhythmBarModelTest(unittest.TestCase):
    def test_learns_consistent_sixteen_to_four_bar_corrections(self) -> None:
        examples = [
            {
                "id": f"example-{index}",
                "raw_predicted": 16,
                "predicted": 16,
                "corrected": 4,
            }
            for index in range(5)
        ]

        model = train_rhythm_bar_model(examples)
        metrics = evaluate_rhythm_bar_model(model, examples)

        self.assertEqual(model["overrides"], {"16": 4})
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["displayed_prediction_accuracy"], 0.0)

    def test_requires_repeated_consistent_corrections(self) -> None:
        examples = [
            {
                "id": "example-1",
                "raw_predicted": 16,
                "predicted": 16,
                "corrected": 4,
            },
            {
                "id": "example-2",
                "raw_predicted": 16,
                "predicted": 16,
                "corrected": 8,
            },
        ]

        model = train_rhythm_bar_model(examples)

        self.assertEqual(model["overrides"], {})


if __name__ == "__main__":
    unittest.main()
