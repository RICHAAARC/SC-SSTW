import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments.paper_results_v1 import attack_cli, attack_eval, confirmation_v1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/paper_results_v1/confirmation_v1.adopted.json"


class _TensorLike:
    def float(self): return self
    def div(self, _value): return self
    def detach(self): return self
    def cpu(self): return self


class ConfirmationTests(unittest.TestCase):
    def config(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_fixed_plan_and_confirmation_attack_scope(self):
        config = self.config()
        plan = confirmation_v1.build_plan(config)
        self.assertEqual(len(plan["rows"]), 3600)
        self.assertEqual(sum(row["status"] == "PLANNED" for row in plan["rows"]), 2880)
        self.assertEqual(sum(row["status"] == "NOT_EXECUTED_BY_NOTEBOOK" for row in plan["rows"]), 720)
        attack = attack_eval.build_plan(config)
        self.assertEqual(attack["fixed_denominator"]["attempt_main_receiver_rows"], 1920)
        self.assertEqual(attack["fixed_denominator"]["attempt_baseline_rows"], 240)
        self.assertEqual(attack["fixed_denominator"]["attempt_quality_rows"], 56)
        parsed = attack_cli.parser().parse_args([
            "--config", str(CONFIG), "--output", "/tmp/unused", "--phase", "import-main",
            "--case-id", "confirm_08",
        ])
        self.assertEqual(parsed.case_id, "confirm_08")
        pilot = json.loads((ROOT / "experiments/paper_results_v1/attack_eval.adopted.json").read_text())
        self.assertEqual(attack_eval.build_plan(pilot)["fixed_denominator"]["attempt_main_receiver_rows"], 480)

    def test_successful_confirmation_baseline_clears_initialized_missing_bits(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            store = attack_eval.AttackRunStore(root / "attack", config, create=True)
            row = next(item for item in store.data["baseline_rows"] if (
                item["case_id"], item["method"], item["attack_id"]
            ) == ("confirm_01", "videoseal", "full"))
            row.update(status="EXTRACTED", missing_bits=32)
            result = {
                "status": "EVALUATED", "reason": None, "decoded_bits": config["payload_bits"],
                "bit_errors": 0, "exact_recovery": True, "tie_count": 0,
            }
            with mock.patch.object(attack_eval, "_evaluate_baseline", return_value=result):
                attack_eval.phase_evaluate(store, config)
            self.assertEqual(row["status"], "EVALUATED")
            self.assertEqual(row["missing_bits"], 0)
            second = next(item for item in store.data["baseline_rows"] if (
                item["case_id"], item["method"], item["attack_id"]
            ) == ("confirm_01", "rivagan", "full"))
            second.update(status="EXTRACTED", missing_bits=32)
            quality = next(item for item in store.data["quality_rows"] if item["case_id"] == "confirm_01")
            quality.update(status="EVALUATED", metrics={"PSNR_RGB": {"status": "EVALUATED", "value": 40.0}})
            store.save()
            with mock.patch.object(attack_eval, "_evaluate_baseline", return_value=result):
                code = attack_cli.main([
                    "--config", str(config_path), "--output", str(store.output), "--phase", "evaluate",
                ])
            self.assertEqual(code, 0)
            refreshed = attack_eval.AttackRunStore(store.output, config)
            report = json.loads((store.output / "evaluation_report.json").read_text())
            self.assertEqual(next(item for item in refreshed.data["baseline_rows"] if item["slot_id"] == second["slot_id"])["missing_bits"], 0)
            self.assertEqual(next(item for item in report["quality_rows"] if item["quality_id"] == quality["quality_id"])["status"], "EVALUATED")
            self.assertIn("eight-source confirmation", report["claim_ceiling"])
            self.assertEqual(len(refreshed.data["phases"]["evaluate"]["aggregate_attempts"]), 2)

    def test_one_physical_map_keeps_keys_and_source_annotations_separate(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attack = root / "attack"
            attack.mkdir()
            artifact_id = "confirm_01/OFF_NATIVE/full/RECEIVED"
            (attack / "attack_run_state.json").write_text(json.dumps({
                "artifacts": [{"artifact_id": artifact_id, "status": "AVAILABLE", "shape": [5, 320, 512, 3]}],
                "receiver_rows": [], "baseline_rows": [], "quality_rows": [], "records": {}, "phases": {},
            }), encoding="utf-8")
            store = confirmation_v1.initialize(attack, root / "confirmation", config)
            maps = ([0, 1, 2, 3, 4], [4, 5, 6, 7, 8])
            rows = []
            for index, (key, receiver, source_map) in enumerate((
                ("K0", "RAW_U", maps[0]), ("K0", "CENTERED_U", maps[1]),
                ("K1", "RAW_U", maps[0]),
            )):
                rows.append({
                    "slot_id": f"confirm_01/OFF_NATIVE/full/{key}/{receiver}",
                    "case_id": "confirm_01", "cohort": "confirmation", "arm": "OFF_NATIVE",
                    "attack_id": "full", "key_label": key, "receiver": receiver,
                    "status": "PENDING_READ", "planned_bits": 32, "missing_bits": 32,
                    "physical_id": "map_0000", "operation": {
                        "received_index_map": [0, 1, 2, 3, 4], "output_frames": 5,
                        "source_coordinate_map": list(source_map),
                    },
                })
            store.data["rows"] = rows
            store.data["physical_inputs"] = [{
                "physical_id": "map_0000", "artifact_id": artifact_id,
                "received_index_map": [0, 1, 2, 3, 4], "output_frames": 5,
                "status": "PLANNED", "logical_slots": [row["slot_id"] for row in rows],
                "key_reads": {},
            }]
            store.save()
            calls = {"encode": 0, "keys": []}

            def encode(_vae, value, encode_observer):
                calls["encode"] += 1; encode_observer("STARTED", None); encode_observer("RETURNED", None)
                return value

            def payload_read(_value, key, output_frames, source_map):
                calls["keys"].append(key)
                self.assertEqual(source_map, [None] * output_frames)
                bit = int(key == config["keys"]["K0"])
                return {
                    "status": "READ", "R": 1, "decoded_bits": [bit] * 32,
                    "votes": [{"ones": bit, "zeros": 1-bit} for _ in range(32)],
                    "signed_votes": [[[bit] * 8] for _ in range(4)],
                    "zero_mask": [[[False] * 8] for _ in range(4)],
                    "bit_rows": [{"bit_index": i, "ones": bit, "zeros": 1-bit, "count": 1, "decoded": bit} for i in range(32)],
                    "time_bit_rows": [{"output_stride_coordinate": 4, "estimated_source_coordinate": None}],
                    "original_reader_match": True,
                }

            fake_numpy = types.SimpleNamespace(
                savez_compressed=lambda stream, **_arrays: stream.write(b"synthetic-npz"),
            )
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch}):
                confirmation_v1.read_case(
                    store, "confirm_01", load_vae=lambda: object(), encode=encode,
                    payload_read=payload_read, load_rgb=lambda _group, _artifact: object(),
                    operate=lambda _received, _mapping: _TensorLike(),
                    save_latent=lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(b"stub-latent")),
                    load_latent=lambda path: _TensorLike(),
                )
            self.assertEqual(calls["encode"], 1)
            self.assertEqual(calls["keys"], [config["keys"]["K0"], config["keys"]["K1"]])
            self.assertEqual(set(store.data["physical_inputs"][0]["key_reads"]), {"K0", "K1"})
            self.assertNotEqual(rows[0]["physical_read_record"]["path"], rows[2]["physical_read_record"]["path"])
            annotation_a = json.loads(Path(rows[0]["detail_record"]["path"]).read_text())
            annotation_b = json.loads(Path(rows[1]["detail_record"]["path"]).read_text())
            self.assertEqual(annotation_a["source_coordinate_map"], list(maps[0]))
            self.assertEqual(annotation_b["source_coordinate_map"], list(maps[1]))
            physical = json.loads(Path(rows[0]["physical_read_record"]["path"]).read_text())
            self.assertIsNone(physical["time_bit_rows"][0]["estimated_source_coordinate"])


if __name__ == "__main__":
    unittest.main()
