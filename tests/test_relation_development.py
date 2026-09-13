import copy
import tempfile
from pathlib import Path
import unittest

from experiments.stage1.evaluate_relation import COORDINATES, SLOTS, evaluate_sample, run_evaluation


def fixture(points):
    annotations = {"sample_id": "fixture", "position_definition": "test_center",
                   "coordinate_system": COORDINATES, "frame_width": 101, "frame_height": 101,
                   "rows": [{"sample_index": i, "time_seconds": i / 5, "p": p,
                             "p_pixel": [100 * v for v in p], "uncertainty_pixels": 1,
                             "reason": "VISIBLE"} for i, p in enumerate(points)]}
    observations = []
    for i, p in enumerate(points):
        m = [(a + b) / 2 for a, b in zip(points[i - 1], p)] if i else None
        q = [0.7 * m[0] + 0.1 * m[1] + 0.1, -0.1 * m[0] + 0.6 * m[1] + 0.2] if m else None
        observations.append({"sample_index": i, "time_seconds": i / 5, "q": q,
                             "valid": bool(i), "reason": "VALID" if i else "INITIAL_FRAME"})
    return observations, annotations


def evaluate(observations, annotations):
    return evaluate_sample(observations, annotations, sample_id="fixture", position_definition="test_center")


class RelationDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0.2 + 0.05 * i, 0.2 + 0.03 * ((i * i) % 11)] for i in range(12)]

    def test_known_affine_recovers_heldout(self):
        result = evaluate(*fixture(self.points))
        self.assertEqual(result["fit_indices"], [2, 4, 6, 8, 10])
        self.assertEqual(result["heldout_indices"], [1, 3, 5, 7, 9, 11])
        self.assertLess(result["heldout_metrics"]["rmse"], 1e-12)
        self.assertGreater(result["heldout_metrics"]["baseline_rmse"], 0.01)
        self.assertIsNotNone(result["annotation_uncertainty"]["fit_minor_sd_over_uncertainty"])

    def test_collinear_input_does_not_answer_2d(self):
        result = evaluate(*fixture([[0.2 + 0.03 * i, 0.5] for i in range(12)]))
        self.assertEqual(result["terminal"], "INPUT_RANK_INSUFFICIENT_FOR_2D_AFFINE")
        self.assertIsNone(result["model"])
        self.assertEqual(result["spread"]["fit_eligible_m"]["rank"], 1)
        self.assertIsNone(result["heldout_metrics"]["rmse"])

    def test_valid_fraction_and_longest_gap_include_initial(self):
        observations, annotations = fixture(self.points)
        for index in (4, 5, 6, 9):
            observations[index].update(q=None, valid=False, reason="NO_MOTION_SUPPORT")
        result = evaluate(observations, annotations)
        availability = result["observation_availability"]
        self.assertEqual(availability["sample_count_including_initial"], 12)
        self.assertEqual(availability["q_valid_count"], 7)
        self.assertEqual(availability["q_valid_fraction"], 7 / 12)
        self.assertEqual(availability["longest_consecutive_invalid_count"], 3)
        initial_only = evaluate(*fixture(self.points))["observation_availability"]
        self.assertEqual(initial_only["q_valid_fraction"], 11 / 12)
        self.assertEqual(initial_only["longest_consecutive_invalid_count"], 1)

    def test_heldout_values_never_enter_fit_baseline_or_scale(self):
        observations, annotations = fixture(self.points)
        before = evaluate(observations, annotations)
        for row in observations:
            if row["sample_index"] % 2:
                row["q"] = [0.95, 0.05]
        after = evaluate(observations, annotations)
        self.assertEqual(before["model"], after["model"])
        self.assertEqual(before["baseline_fit_q_mean"], after["baseline_fit_q_mean"])
        self.assertEqual(before["spread"]["fit_eligible_m"], after["spread"]["fit_eligible_m"])
        self.assertGreater(after["heldout_metrics"]["rmse"], before["heldout_metrics"]["rmse"])

    def test_missing_annotations_preserve_time_and_static_zero_scale(self):
        observations, annotations = fixture(self.points)
        annotations["rows"].pop(3)
        result = evaluate(observations, annotations)
        self.assertEqual(len(result["rows"]), 12)
        self.assertEqual(result["rows"][3]["annotation_reason"], "ANNOTATION_ROW_MISSING")
        self.assertFalse(result["rows"][4]["eligible"])
        self.assertEqual(result["rows"][4]["role"], "fit")
        observations, annotations = fixture([[0.5, 0.5]] * 12)
        for row in observations[1:]:
            row["q"] = [0.1 + row["sample_index"] * 0.03, 0.4]
        static = evaluate(observations, annotations)
        self.assertIsNone(static["heldout_metrics"]["normalized_baseline_rmse"])
        self.assertTrue(static["stationary_annotation_diagnostics"]["exact_zero_p_scale"])
        self.assertGreater(static["stationary_annotation_diagnostics"]["q_rms_drift"], 0)

    def test_annotation_pixel_mismatch_rejected_and_fixed_missing_slots(self):
        observations, annotations = fixture(self.points)
        bad = copy.deepcopy(annotations)
        bad["rows"][0]["p_pixel"][0] += 1
        with self.assertRaises(ValueError):
            evaluate(observations, bad)
        manifest = {"position_definition": "test_center",
                    "slots": [{"slot": slot, "status": "MISSING", "sample_id": None,
                               "reason": "fixed content slot unavailable"} for slot in SLOTS]}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_evaluation(manifest, Path(tmp) / "out")
            self.assertEqual(result["fixed_slot_denominator"], 4)
            self.assertEqual(len(result["rows"]), 4)
            self.assertTrue(all(r["terminal"] == "SAMPLE_MISSING" for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
