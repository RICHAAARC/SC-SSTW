import itertools
import json
import unittest
from pathlib import Path

from experiments.paper_results_v1.attack_matrix import expand_weighted_truth, fixed_attack_specs, path_error_rows
from experiments.paper_results_v1.attack_eval import build_plan, validate_config
from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import PUBLIC, decode_visible_span, solve_monotone
from runtime.wan.variable_rgb_media import commands


ROOT = Path(__file__).resolve().parents[1]


class AttackMatrixTests(unittest.TestCase):
    def test_fixed_matrix_lengths_positions_and_denominators(self):
        specs = fixed_attack_specs()
        self.assertEqual(len(specs), 15)
        lengths = [len(expand_weighted_truth(row)) for row in specs]
        self.assertEqual(lengths, [181, 89, 241, 145, 181, 181, 163, 163, 163, 199, 199, 199, 199, 199, 199])
        self.assertTrue(all(len(row.get("positions", [])) == 18 for row in specs[6:]))
        config = validate_config(json.loads((ROOT / "experiments/paper_results_v1/attack_eval.adopted.json").read_text()))
        plan = build_plan(config)
        self.assertEqual((len(plan["receiver_rows"]), len(plan["baseline_rows"]), len(plan["quality_rows"])), (2400, 300, 70))
        self.assertEqual(plan["fixed_denominator"]["attempt_main_receiver_rows"], 480)
        self.assertEqual(plan["fixed_denominator"]["attempt_main_receiver_bits"], 15360)
        self.assertEqual({row["status"] for row in plan["receiver_rows"] if row["case_id"].startswith("confirm_")}, {"NOT_EXECUTED_BY_NOTEBOOK"})

    def test_top_two_paths_match_small_exhaustive_embedding(self):
        # Four received frames; only source columns 0..3 are competitive.
        local = [[-100.0] * 181 for _ in range(4)]
        local[0][:4] = [4, 1, 0, -1]
        local[1][:4] = [0, 5, 3, 1]
        local[2][:4] = [-1, 2, 6, 4]
        local[3][:4] = [-2, 0, 3, 7]
        result = solve_monotone(local, [[1.0] * 181 for _ in range(4)])
        exhaustive = sorted(
            ((sum(local[t][s] for t, s in enumerate(path)), list(path)) for path in itertools.combinations_with_replacement(range(4), 4)),
            reverse=True,
        )
        self.assertEqual(result["best_path"], exhaustive[0][1])
        self.assertEqual(result["runner_up_path"], exhaustive[1][1])
        self.assertNotEqual(result["best_path"], result["runner_up_path"])
        self.assertEqual(result["score_gap"], exhaustive[0][0] - exhaustive[1][0])

    def test_exact_tie_is_unresolved_and_crop_operation_is_visible_span_only(self):
        numerator = [[0.0] * 181 for _ in range(2)]
        result = solve_monotone(numerator, [[1.0] * 181 for _ in range(2)])
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertTrue(result["exact_tie"])
        operation = decode_visible_span(list(range(37, 126)), PUBLIC)
        self.assertEqual((operation["anchor"], operation["front_boundary_copies"], operation["output_frames"]), (36, 1, 89))
        self.assertEqual((operation["tail_discarded_source_positions"], operation["wan_support"]), (1, 22))

    def test_weighted_truth_error_keeps_support_and_spread(self):
        metric = path_error_rows([5], [[{"source_index": 5, "weight": 0.5}, {"source_index": 6, "weight": 0.5}]])
        self.assertEqual(metric["minimum_support_absolute_error_sum"], 0)
        self.assertEqual(metric["weighted_absolute_error_sum"], 0.5)

    def test_variable_codec_keeps_parameters_and_requested_length(self):
        rows = commands(Path("published.mp4"), 241)
        self.assertIn("241", rows["save"])
        self.assertIn("libx264", rows["save"])
        self.assertIn("yuv420p", rows["save"])
        self.assertIn("8", rows["save"])


if __name__ == "__main__":
    unittest.main()
