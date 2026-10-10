import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # Initialize before the clock's intentionally minimal numpy stub.

from experiments.paper_results_v1.attack_eval import (
    ATTEMPT_CASES, AttackRunStore, build_plan, phase_evaluate, phase_receiver_clock,
    phase_receiver_read, record_external_failure, validate_config,
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

    def test_resource_plan_allows_raw_and_two_distinct_blind_maps(self):
        plan = build_plan(self.config())["fixed_denominator"]
        self.assertEqual(plan["attempt_logical_wan_reads"], 480)
        self.assertEqual(plan["attempt_physical_wan_encode_upper_bound"], 360)

    def _clock_fixture(self, store, available_attacks):
        for artifact in store.data["artifacts"]:
            if artifact["case_id"] != "pilot_01" or artifact["kind"] != "TEMPORAL_OBSERVATION":
                continue
            parts = artifact["artifact_id"].split("/")
            if parts[1] == "OFF_NATIVE" and parts[2] in available_attacks:
                artifact.update(status="AVAILABLE", shape=[181, 320, 512, 3])
            else:
                artifact.update(status="MISSING", reason="fixture")
        store.save()

    def _fake_clock_modules(self, config, score_calls):
        fake_numpy = types.SimpleNamespace(
            savez_compressed=lambda path, **arrays: Path(path).write_bytes(b"npz"),
        )
        public = object()

        def score_framewise(latent, key, supplied_public):
            self.assertIs(supplied_public, public)
            score_calls.append(key)
            if key == config["keys"]["K0"]:
                raise RuntimeError("synthetic K0 score failure")
            return {
                "signed_projection": [[1.0]], "rho": [[1.0]],
                "received_frames": 181, "source_frames": 181,
            }

        fake_method = types.SimpleNamespace(
            PUBLIC=public,
            score_framewise=score_framewise,
            solve_monotone=lambda *args: {
                "status": "ESTIMATED", "path": [0] * 181, "reason": None,
                "score_gap": 1.0,
            },
            decode_visible_span=lambda *args: {
                "status": "SUPPORTED", "reason": None, "output_frames": 181,
                "received_index_map": list(range(181)),
                "source_coordinate_map": list(range(181)),
            },
        )
        fake_method.receive_scores = lambda *args: {
            "estimate": fake_method.solve_monotone(),
            "operation": fake_method.decode_visible_span(),
        }
        return fake_numpy, fake_method

    def test_clock_key_failure_isolated_and_actual_encode_persisted(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            store = AttackRunStore(Path(directory) / "run", config, create=True)
            self._clock_fixture(store, {"full"})
            score_calls = []
            fake_numpy, fake_method = self._fake_clock_modules(config, score_calls)

            class Backend:
                def encode(self, rgb):
                    return types.SimpleNamespace(shape=(181,))

                def close(self):
                    pass

            import main.tube_state as tube_state
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy,
                "main.tube_state.receiver_controls_v1": types.SimpleNamespace(receive_scores=fake_method.receive_scores),
            }), mock.patch.object(
                tube_state, "video_trajectory_temporal_edit_receiver_v1", fake_method, create=True,
            ), mock.patch(
                "experiments.paper_results_v1.real_backends.load_local_framewise_backend",
                return_value=Backend(),
            ), mock.patch("experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object()):
                phase_receiver_clock(store, config, "pilot_01")

            k0 = next(row for row in store.data["receiver_rows"] if row["slot_id"] == "pilot_01/OFF_NATIVE/full/K0/BLIND_PATH")
            k1 = next(row for row in store.data["receiver_rows"] if row["slot_id"] == "pilot_01/OFF_NATIVE/full/K1/BLIND_PATH")
            self.assertEqual(k0["status"], "FAILED")
            self.assertEqual(k1["status"], "CLOCK_READY")
            self.assertEqual(score_calls, [config["keys"]["K0"], config["keys"]["K1"]])
            actual = store.data["records"]["pilot_01"]["framewise_clock_encodes"]
            self.assertEqual((actual["attempted"], actual["completed"], actual["failed"], actual["unfinished"]), (1, 1, 0, 0))

    def test_clock_interrupt_retains_completed_and_unfinished_call_records(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            store = AttackRunStore(run, config, create=True)
            self._clock_fixture(store, {"full", "crop37_126"})
            fake_numpy, fake_method = self._fake_clock_modules(config, [])
            fake_method.score_framewise = lambda *args: {
                "signed_projection": [[1.0]], "rho": [[1.0]],
                "received_frames": 181, "source_frames": 181,
            }

            class Backend:
                calls = 0

                def encode(self, rgb):
                    self.calls += 1
                    if self.calls == 2:
                        raise KeyboardInterrupt()
                    return types.SimpleNamespace(shape=(181,))

                def close(self):
                    pass

            import main.tube_state as tube_state
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy,
                "main.tube_state.receiver_controls_v1": types.SimpleNamespace(receive_scores=fake_method.receive_scores),
            }), mock.patch.object(
                tube_state, "video_trajectory_temporal_edit_receiver_v1", fake_method, create=True,
            ), mock.patch(
                "experiments.paper_results_v1.real_backends.load_local_framewise_backend",
                return_value=Backend(),
            ), mock.patch("experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object()):
                with self.assertRaises(KeyboardInterrupt):
                    phase_receiver_clock(store, config, "pilot_01")

            reopened = AttackRunStore(run, config)
            actual = reopened.data["records"]["pilot_01"]["framewise_clock_encodes"]
            self.assertEqual((actual["attempted"], actual["completed"], actual["failed"], actual["unfinished"]), (2, 1, 0, 1))
            self.assertEqual([row["status"] for row in actual["records"]], ["COMPLETE", "RUNNING"])

    def test_external_failure_closes_running_only_and_preserves_internal_failure(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run = root / "run"
            store = AttackRunStore(run, config, create=True)
            store.phase_start("receiver-read", "pilot_01")
            completed, unfinished = [
                row for row in store.data["receiver_rows"] if row["case_id"] == "pilot_01"
            ][:2]
            completed.update(status="READ", reason=None)
            unfinished.update(status="CLOCK_READY", reason=None)
            store.save()
            record_external_failure(config_path, run, "receiver-read", "pilot_01", "external SIGTERM")
            reopened = AttackRunStore(run, config)
            self.assertEqual(reopened.data["phases"]["receiver-read"]["cases"]["pilot_01"]["status"], "FAILED")
            self.assertEqual(reopened.data["receiver_rows"][0]["status"], "READ")
            self.assertEqual(reopened.data["receiver_rows"][1]["status"], "FAILED")

            record_external_failure(config_path, run, "attack-media", "pilot_02", "external before launch")
            self.assertEqual(
                AttackRunStore(run, config).data["phases"]["attack-media"]["cases"]["pilot_02"]["status"],
                "FAILED",
            )

            reopened = AttackRunStore(run, config)
            reopened.phase_start("import-main", "pilot_02")
            reopened.phase_finish("import-main", "pilot_02")
            record_external_failure(config_path, run, "import-main", "pilot_02", "late generic error")
            self.assertEqual(
                AttackRunStore(run, config).data["phases"]["import-main"]["cases"]["pilot_02"]["status"],
                "COMPLETE",
            )

            reopened.phase_failure("receiver-clock", ValueError("more precise internal error"), "pilot_02")
            before = list(reopened.data["phases"]["receiver-clock"]["cases"]["pilot_02"]["failures"])
            record_external_failure(config_path, run, "receiver-clock", "pilot_02", "generic external error")
            after = AttackRunStore(run, config).data["phases"]["receiver-clock"]["cases"]["pilot_02"]["failures"]
            self.assertEqual(after, before)

    def test_wan_call_return_stays_complete_if_cpu_detach_fails(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            store = AttackRunStore(run, config, create=True)
            source = store.artifact("pilot_01/OFF_NATIVE/full/RECEIVED")
            source.update(status="AVAILABLE", shape=[181, 320, 512, 3])
            row = next(item for item in store.data["receiver_rows"] if item["slot_id"] == "pilot_01/OFF_NATIVE/full/K0/RAW")
            row.update(status="CLOCK_READY", operation={
                "received_index_map": list(range(181)), "source_coordinate_map": [None] * 181,
                "output_frames": 181,
            })
            store.save()

            class Media:
                def float(self): return self
                def div(self, value): return self

            class Returned:
                def detach(self): raise RuntimeError("synthetic CPU transfer failure")

            fake_generation = types.SimpleNamespace(load_frozen_vae=lambda *args, **kwargs: object())
            fake_vae = types.SimpleNamespace(reencode_rgb24_readback=lambda *args: Returned())
            fake_receiver = types.SimpleNamespace(
                operate_map=lambda *args: Media(), read_payload_general=lambda *args: None,
            )
            with mock.patch.dict(sys.modules, {
                "runtime.wan.generation": fake_generation,
                "runtime.wan.vae": fake_vae,
                "runtime.wan.video_trajectory_temporal_edit_receiver_v1": fake_receiver,
                "numpy": types.SimpleNamespace(),
            }), mock.patch("experiments.paper_results_v1.attack_eval._load_rgb8", return_value=Media()):
                phase_receiver_read(store, config, "pilot_01")

            reopened = AttackRunStore(run, config)
            actual = reopened.data["records"]["pilot_01"]["physical_wan_encodes"]
            self.assertEqual((actual["attempted"], actual["completed"], actual["failed"]), (1, 1, 0))
            self.assertEqual(actual["records"][0]["status"], "COMPLETE")
            final_row = next(item for item in reopened.data["receiver_rows"] if item["slot_id"] == row["slot_id"])
            self.assertEqual(final_row["status"], "FAILED")

    def test_wan_interrupt_retains_calls_and_preprocess_failure_is_not_counted(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            store = AttackRunStore(run, config, create=True)
            selected = (
                ("full", [99]), ("crop37_126", [1]), ("speed075", [2]),
            )
            for attack_id, index_map in selected:
                store.artifact(f"pilot_01/OFF_NATIVE/{attack_id}/RECEIVED").update(
                    status="AVAILABLE", shape=[181, 320, 512, 3],
                )
                row = next(item for item in store.data["receiver_rows"] if item["slot_id"] == f"pilot_01/OFF_NATIVE/{attack_id}/K0/RAW")
                row.update(status="CLOCK_READY", operation={
                    "received_index_map": index_map, "source_coordinate_map": [None],
                    "output_frames": 1,
                })
            store.save()

            class Media:
                def float(self): return self
                def div(self, value): return self

            class Returned:
                def detach(self): return self
                def cpu(self): return self

            encode_calls = []

            def reencode(*args):
                encode_calls.append(1)
                if len(encode_calls) == 2:
                    raise KeyboardInterrupt()
                return Returned()

            def operate(received, index_map):
                if index_map == [99]:
                    raise RuntimeError("synthetic preprocessing failure")
                return Media()

            result = {
                "signed_votes": [], "zero_mask": [], "decoded_bits": [0] * 32,
                "votes": [], "bit_rows": [], "R": 0, "original_reader_match": True,
            }
            fake_generation = types.SimpleNamespace(load_frozen_vae=lambda *args, **kwargs: object())
            fake_vae = types.SimpleNamespace(reencode_rgb24_readback=reencode)
            fake_receiver = types.SimpleNamespace(
                operate_map=operate, read_payload_general=lambda *args: dict(result),
            )
            fake_numpy = types.SimpleNamespace(
                savez_compressed=lambda path, **arrays: Path(path).write_bytes(b"npz"),
            )
            with mock.patch.dict(sys.modules, {
                "runtime.wan.generation": fake_generation,
                "runtime.wan.vae": fake_vae,
                "runtime.wan.video_trajectory_temporal_edit_receiver_v1": fake_receiver,
                "numpy": fake_numpy,
            }), mock.patch("experiments.paper_results_v1.attack_eval._load_rgb8", return_value=Media()):
                with self.assertRaises(KeyboardInterrupt):
                    phase_receiver_read(store, config, "pilot_01")

            reopened = AttackRunStore(run, config)
            actual = reopened.data["records"]["pilot_01"]["physical_wan_encodes"]
            self.assertEqual((actual["attempted"], actual["completed"], actual["failed"], actual["unfinished"]), (2, 1, 0, 1))
            self.assertEqual([row["status"] for row in actual["records"]], ["COMPLETE", "RUNNING"])
            self.assertEqual(len(encode_calls), 2)


if __name__ == "__main__":
    unittest.main()
