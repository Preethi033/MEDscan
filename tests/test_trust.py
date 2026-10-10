import unittest

from scanexplain.trust import (
    EmbeddingReference,
    PredictionRecord,
    analyze_predictions,
    check_ood,
)


class AnalyzePredictionsTests(unittest.TestCase):
    def test_builds_failure_gallery_for_errors_uncertainty_and_shortcut_cues(self):
        report = analyze_predictions(
            [
                PredictionRecord("correct", "clear", "clear", 0.95),
                PredictionRecord("wrong", "clear", "finding", 0.81),
                PredictionRecord(
                    "border",
                    "clear",
                    "clear",
                    0.92,
                    shortcut_cues=("bright image border",),
                ),
                PredictionRecord("uncertain", "finding", "finding", 0.42),
            ]
        )

        self.assertEqual(report.evaluated_count, 4)
        self.assertEqual(report.misclassified_count, 1)
        self.assertEqual(report.low_confidence_count, 1)
        self.assertEqual(report.shortcut_cue_count, 1)
        self.assertEqual(
            [case.sample_id for case in report.failure_gallery],
            ["wrong", "border", "uncertain"],
        )
        self.assertIn("Expected 'clear'", report.failure_gallery[0].note)
        self.assertIn("does not establish", report.failure_gallery[1].note)
        self.assertEqual(report.to_dict()["evaluated_count"], 4)

    def test_rejects_duplicate_ids_and_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            analyze_predictions(
                [
                    PredictionRecord("same", "a", "a", 0.8),
                    PredictionRecord("same", "a", "b", 0.4),
                ]
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            analyze_predictions([PredictionRecord("bad", "a", "a", 1.1)])


class OODCheckTests(unittest.TestCase):
    def setUp(self):
        self.reference = EmbeddingReference.fit(
            [(1.0, 0.0), (0.99, 0.01), (0.98, 0.02)],
            quantile=0.95,
        )

    def test_flags_embedding_far_from_supported_reference(self):
        result = check_ood(
            [0.99, 0.01],
            reference=self.reference,
            embedding=(0.0, 1.0),
        )

        self.assertEqual(result.status, "ood_suspected")
        self.assertIn("embedding_far_from_supported_reference", result.reasons)

    def test_reports_uncertainty_separately_from_embedding_ood(self):
        result = check_ood(
            [0.5, 0.5],
            reference=self.reference,
            embedding=(1.0, 0.0),
        )

        self.assertEqual(result.status, "uncertain")
        self.assertIn("high_prediction_entropy", result.reasons)

    def test_does_not_claim_ood_without_reference_embeddings(self):
        result = check_ood([0.9, 0.1])

        self.assertEqual(result.status, "cannot_assess")
        self.assertEqual(result.reasons, ("no_supported_embedding_reference",))

    def test_validates_embedding_pair_and_probability_distribution(self):
        with self.assertRaisesRegex(ValueError, "must either both"):
            check_ood([1.0], reference=self.reference)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            check_ood([0.7, 0.4])

    def test_reference_requires_multiple_nonzero_equal_dimension_vectors(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            EmbeddingReference.fit([(1.0, 0.0)])
        with self.assertRaisesRegex(ValueError, "zero vector"):
            EmbeddingReference.fit([(1.0, 0.0), (0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
