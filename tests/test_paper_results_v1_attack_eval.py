import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.paper_results_v1.attack_eval import (
    ATTEMPT_CASES, AttackRunStore, phase_evaluate, phase_receiver_clock, validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class AttackEvalTests(unittest.TestCase):
    def config(self):
        return validate_config(json.loads((ROOT / "experiments/paper_results_v1/attack_eval.adopted.json").read_text()))

    def test_clock_setup_failure_keeps_raw_ready_and_fails_blind_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AttackRunStore(Path(directory) / "run", self.config(), create=True)
            for artifact in store.data["artifacts"]:
                if artifact["case_id"] == "pilot_01" and artifact["kind"] == "TEMPORAL_OBSERVATION" and any(
                    f"/{arm}/" in artifact["artifact_id"] for arm in ("OFF_NATIVE", "PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_FRAMEWISE_M05")
                ):
                    artifact.update(status="AVAILABLE", shape=[181, 320, 512, 3])
            store.save()
            with mock.patch(
                "experiments.paper_results_v1.real_backends.load_local_framewise_backend",
                side_effect=RuntimeError("synthetic framewise setup failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    phase_receiver_clock(store, self.config(), "pilot_01")
            raw = [row for row in store.data["receiver_rows"] if row["case_id"] == "pilot_01" and row["mode"] == "RAW"]
            blind = [row for row in store.data["receiver_rows"] if row["case_id"] == "pilot_01" and row["mode"] == "BLIND_PATH"]
            self.assertEqual({row["status"] for row in raw}, {"CLOCK_READY"})
            self.assertEqual({row["status"] for row in blind}, {"FAILED"})
            self.assertTrue(all(row["operation"]["source_coordinate_map"] == [None] * 181 for row in raw if row["attack_id"] == "full"))

    def test_fixed_report_retains_missing_and_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AttackRunStore(Path(directory) / "run", self.config(), create=True)
            rows = [row for row in store.data["receiver_rows"] if row["case_id"] == "pilot_01"]
            rows[0].update(status="READ", decoded_bits=self.config()["payload_bits"], missing_bits=0)
            rows[1].update(status="FAILED", reason="fixture", missing_bits=32)
            report = phase_evaluate(store, self.config())
            self.assertEqual(len(report["receiver_rows"]), 2400)
            self.assertEqual(len(report["baseline_rows"]), 300)
            self.assertEqual(len(report["quality_rows"]), 70)
            self.assertEqual(len(report["paired_m05_rows"]), 1200)
            self.assertEqual(report["source_summaries"]["pilot_01"]["main_fixed_summary"]["fixed_rows"], 240)
            self.assertEqual(report["source_summaries"]["pilot_01"]["main_fixed_summary"]["missing_bits"], 239 * 32)
            self.assertEqual(report["source_summaries"]["confirm_01"]["execution_scope"], "NOT_EXECUTED_BY_NOTEBOOK")
            self.assertEqual({row["status"] for row in report["receiver_rows"] if row["case_id"] == "confirm_01"}, {"NOT_EXECUTED_BY_NOTEBOOK"})
            confirmation = report["receiver_group_summaries"]["CONFIRMATION_NOT_EXECUTED|OFF_NATIVE|K0|RAW|FULL"]
            self.assertIsNone(confirmation["fixed_denominator_exact_recovery_rate"])
            self.assertEqual({row["status"] for row in report["paired_m05_rows"] if row["case_id"] == "confirm_01"}, {"NOT_EXECUTED_BY_NOTEBOOK"})


if __name__ == "__main__":
    unittest.main()
