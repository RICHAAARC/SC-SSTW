import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments.paper_results_v1 import attack_eval, attack_recovery


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/paper_results_v1/attack_eval.adopted.json"


def receipt(path, frames=181):
    return {"status": "SAVED", "path": str(path), "bytes": frames * 320 * 512 * 3, "shape": [frames, 320, 512, 3], "dtype": "uint8"}


class AttackRecoveryTests(unittest.TestCase):
    def config(self):
        return attack_eval.read_json(CONFIG)

    def source_fixture(self, root):
        config = self.config()
        source = Path(root) / "source"
        state_root = source / "run_state"
        store = attack_eval.AttackRunStore(state_root, config, create=True)
        store.data["evaluation_report"] = {"path": str(state_root / "evaluation_report.json"), "bytes": 1}
        for artifact in store.data["artifacts"]:
            if artifact["case_id"] in attack_eval.ATTEMPT_CASES:
                artifact.update(status="FAILED", reason="source status collision")
        for row in store.data["receiver_rows"]:
            if row["case_id"] in attack_eval.ATTEMPT_CASES:
                row.update(status="FAILED", reason="source status collision")
        # Preserve the actual FULL denominator shape: 30 evaluated + 2 unsupported.
        full = [row for row in store.data["receiver_rows"] if row["case_id"] in attack_eval.ATTEMPT_CASES and row["attack_id"] == "full"]
        for row in full[:30]:
            row.update(status="EVALUATED", decoded_bits=[0] * 32, bit_errors=32, exact_recovery=False)
        for row in full[30:32]:
            row.update(status="UNSUPPORTED", missing_bits=32)
        for row in store.data["baseline_rows"]:
            if row["case_id"] in attack_eval.ATTEMPT_CASES:
                row.update(status="FAILED", reason="source collision")
        for row in store.data["quality_rows"]:
            if row["case_id"] in attack_eval.ATTEMPT_CASES:
                row.update(status="EVALUATED" if row["candidate"] not in attack_eval.BASELINES else "FAILED")
        store.save()
        return source, store

    def test_saved_codec_receipt_becomes_available_without_keyword_collision(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            store = attack_eval.AttackRunStore(Path(directory) / "run", config, create=True)
            for method in attack_eval.BASELINES:
                store.artifact(f"pilot_01/{method}/NATIVE_PRE").update(status="AVAILABLE")
            store.save()
            fake_module = types.SimpleNamespace(roundtrip=lambda *a, **k: (object(), {"received_rgb": receipt("received.rgb8"), "mp4": {"status": "SAVED"}}))
            with mock.patch.dict(sys.modules, {"runtime.wan.variable_rgb_media": fake_module}), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object(),
            ):
                attack_eval.phase_baseline_codec(store, config, "pilot_01")
            for method in attack_eval.BASELINES:
                row = store.artifact(f"pilot_01/{method}/NATIVE_POST")
                self.assertEqual(row["status"], "AVAILABLE")
                self.assertEqual(row["media_receipt_status"], "SAVED")

    def test_saved_attack_receipt_registers_truth_and_reaches_receiver_clock(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            store = attack_eval.AttackRunStore(Path(directory) / "run", config, create=True)
            for arm in attack_eval.MAIN_ARMS:
                store.artifact(f"pilot_01/{arm}/POST").update(status="FAILED")
            store.artifact("pilot_01/OFF_NATIVE/POST").update(
                status="AVAILABLE", path="source.rgb8", shape=[181, 320, 512, 3], dtype="uint8", bytes=1,
            )
            store.save()

            def roundtrip(value, output, *, media_output):
                attack_id = Path(output).name
                frames = attack_eval.recipe_receipt(next(row for row in config["attacks"] if row["attack_id"] == attack_id))["output_frames"]
                return object(), {
                    "received_rgb": receipt(Path(output) / "received.rgb8", frames),
                    "edited_rgb": receipt(Path(output) / "edited.rgb8", frames),
                }

            fake_media = types.SimpleNamespace(roundtrip=roundtrip)
            with mock.patch.dict(sys.modules, {"runtime.wan.variable_rgb_media": fake_media}), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object(),
            ), mock.patch(
                "experiments.paper_results_v1.attack_eval.apply_edit_rgb8",
                side_effect=lambda value, spec: (object(), attack_eval.recipe_receipt(spec)),
            ):
                attack_eval.phase_attack_media(store, config, "pilot_01")
            target = store.artifact("pilot_01/OFF_NATIVE/crop37_126/RECEIVED")
            self.assertEqual(target["status"], "AVAILABLE")
            self.assertEqual(target["media_receipt_status"], "SAVED")
            self.assertTrue(target["truth_hidden_from_receiver"])
            self.assertTrue(Path(target["weighted_truth_record"]["path"]).is_file())

            class Backend:
                def encode(self, rgb): return types.SimpleNamespace(shape=(89,))
                def close(self): pass

            public = object()
            method = types.SimpleNamespace(
                PUBLIC=public,
                score_framewise=lambda latent, key, p: {"signed_projection": [[1.0]], "rho": [[1.0]], "received_frames": 89, "source_frames": 181},
                solve_monotone=lambda *a: {"status": "UNRESOLVED", "reason": "synthetic", "path": None, "score_gap": 0.0},
            )
            fake_numpy = types.SimpleNamespace(savez_compressed=lambda path, **arrays: Path(path).write_bytes(b"npz"))
            import main.tube_state as tube_state
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy}), mock.patch.object(
                tube_state, "video_trajectory_temporal_edit_receiver_v1", method, create=True,
            ), mock.patch(
                "experiments.paper_results_v1.real_backends.load_local_framewise_backend", return_value=Backend(),
            ), mock.patch("experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object()):
                attack_eval.phase_receiver_clock(store, config, "pilot_01")
            raw = next(row for row in store.data["receiver_rows"] if row["slot_id"] == "pilot_01/OFF_NATIVE/crop37_126/K0/RAW")
            self.assertEqual(raw["status"], "CLOCK_READY")

    def test_initialize_preserves_full_results_and_fixed_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            state = attack_recovery.initialize(CONFIG, source, output)
            full = [row for row in state["receiver_rows"] if row["case_id"] in attack_eval.ATTEMPT_CASES and row["attack_id"] == "full"]
            self.assertEqual(sum(row["status"] == "EVALUATED" for row in full), 30)
            self.assertEqual(sum(row["status"] == "UNSUPPORTED" for row in full), 2)
            edited = [row for row in state["receiver_rows"] if row["case_id"] in attack_eval.ATTEMPT_CASES and row["attack_id"] != "full"]
            self.assertEqual({row["status"] for row in edited}, {"PLANNED"})
            self.assertEqual(state["fixed_denominator"]["attempt_main_receiver_rows"], 480)
            self.assertEqual(state["recovery"]["budget"]["new_baseline_temporal_codec_roundtrips"], 56)
            self.assertTrue((output / "source_attack_state.json").is_file())

    def test_media_pipeline_isolates_unreadable_mp4_and_resume_does_not_repeat_codec(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            store = attack_eval.AttackRunStore(output, config, temp_root=Path(directory) / "temp")
            decode_calls = []
            codec_calls = []

            def fake_decode(mp4, target, frames):
                decode_calls.append(str(mp4))
                if "pilot_01/OFF_NATIVE/crop37_126" in str(mp4):
                    raise ValueError("synthetic unreadable MP4")
                return attack_eval._available_media(receipt(target, frames))

            def fake_roundtrip(value, output_path, *, media_output):
                codec_calls.append(str(media_output))
                if len(codec_calls) == 2:
                    raise KeyboardInterrupt()
                return object(), {"received_rgb": receipt(Path(output_path) / "received.rgb8", 163)}

            with mock.patch.object(attack_recovery, "_decode", side_effect=fake_decode), mock.patch.object(
                attack_recovery, "_validate_reused_rows"
            ), mock.patch.object(attack_recovery, "apply_edit_rgb8", return_value=(object(), {})), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object()
            ), mock.patch("runtime.wan.variable_rgb_media.roundtrip", side_effect=fake_roundtrip):
                with self.assertRaises(KeyboardInterrupt):
                    attack_recovery.phase_index_saved_media(store, config, "pilot_01")

            first = copy.deepcopy(store.data["recovery"]["actual_calls"]["baseline_edit_codecs"])
            self.assertEqual([row["status"] for row in first], ["COMPLETE", "RUNNING"])
            self.assertEqual(store.artifact("pilot_01/OFF_NATIVE/crop37_126/RECEIVED")["status"], "FAILED")

            # The interrupted call is retained and not executed again; other
            # fixed rows continue independently on resume.
            def successful_roundtrip(value, output_path, *, media_output):
                codec_calls.append(str(media_output))
                return object(), {"received_rgb": receipt(Path(output_path) / "received.rgb8", 163)}

            with mock.patch.object(attack_recovery, "_decode", side_effect=fake_decode), mock.patch.object(
                attack_recovery, "_validate_reused_rows"
            ), mock.patch.object(attack_recovery, "apply_edit_rgb8", return_value=(object(), {})), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object()
            ), mock.patch("runtime.wan.variable_rgb_media.roundtrip", side_effect=successful_roundtrip):
                attack_recovery.phase_index_saved_media(store, config, "pilot_01")
            after = store.data["recovery"]["actual_calls"]["baseline_edit_codecs"]
            self.assertEqual(sum(row["status"] == "RUNNING" for row in after), 1)
            self.assertEqual(len({row["call_id"] for row in after}), len(after))
            self.assertEqual(store.data["recovery"]["phases"]["index-saved-media/pilot_01"]["status"], "COMPLETE")

    def test_complete_index_rematerializes_deleted_temp_from_saved_mp4_without_codec(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            temp = Path(directory) / "temp"
            store = attack_eval.AttackRunStore(output, config, temp_root=temp)
            shared = Path(directory) / "still-readable.rgb8"; shared.write_bytes(b"x")
            for artifact in store.data["artifacts"]:
                if artifact["case_id"] == "pilot_01" and artifact["kind"] in ("IMPORTED_MAIN", "NEW_BASELINE", "TEMPORAL_OBSERVATION"):
                    artifact.update(status="AVAILABLE", path=str(shared), shape=[181, 320, 512, 3], dtype="uint8", bytes=1)
            missing = store.artifact("pilot_01/OFF_NATIVE/crop37_126/RECEIVED")
            missing_path = temp / "old-kernel.rgb8"
            missing.update(path=str(missing_path), shape=[89, 320, 512, 3])
            store.data["recovery"]["phases"]["index-saved-media/pilot_01"] = {"status": "COMPLETE", "failures": []}
            store.save()
            calls = []

            def fake_decode(mp4, target, frames):
                calls.append((str(mp4), frames)); Path(target).parent.mkdir(parents=True, exist_ok=True); Path(target).write_bytes(b"rgb")
                return attack_eval._available_media(receipt(target, frames))

            with mock.patch.object(attack_recovery, "_decode", side_effect=fake_decode), mock.patch.object(
                attack_recovery, "_validate_reused_rows"
            ), mock.patch("runtime.wan.variable_rgb_media.roundtrip") as codec:
                attack_recovery.phase_index_saved_media(store, config, "pilot_01")
            codec.assert_not_called()
            self.assertEqual(len(calls), 1)
            self.assertIn("/run_state/media/pilot_01/OFF_NATIVE/crop37_126/published.mp4", calls[0][0])
            self.assertEqual(store.artifact("pilot_01/OFF_NATIVE/crop37_126/RECEIVED")["status"], "AVAILABLE")

    def test_reused_nested_sidecar_path_is_relocated_and_opened(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, source_store = self.source_fixture(directory)
            recorded_root = "/content/drive/MyDrive/Video-WM/old-attack-run"
            source_store.data["evaluation_report"]["path"] = recorded_root + "/run_state/evaluation_report.json"
            source_store.save()
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            store = attack_eval.AttackRunStore(output, config)
            old = store.data["recovery"]["source_recorded_root"]
            sidecar = source / "run_state/records/evidence.npz"; sidecar.parent.mkdir(parents=True, exist_ok=True); sidecar.write_bytes(b"npz-evidence")
            votes = source / "run_state/records/votes.npz"; votes.write_bytes(b"npz-votes")
            record = source / "run_state/records/clock.json"
            record.write_text(json.dumps({
                "evidence": {"path": "evidence.npz"},
                "vote_sidecar": {"path": old + "/run_state/records/votes.npz"},
            }), encoding="utf-8")
            row = next(item for item in store.data["receiver_rows"] if item["slot_id"] == "pilot_01/OFF_NATIVE/full/K0/BLIND_PATH")
            row.update(status="EVALUATED", clock_record={"path": str(record), "bytes": record.stat().st_size})
            opened = []
            class Loaded:
                files = ["value"]
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def __getitem__(self, key): return types.SimpleNamespace(shape=(1,))
            fake_numpy = types.SimpleNamespace(load=lambda path, allow_pickle=False: (opened.append(str(path)), Path(path).read_bytes(), Loaded())[-1])
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
                attack_recovery._validate_reused_rows(store, "pilot_01")
            copied = json.loads(Path(row["clock_record"]["path"]).read_text())
            self.assertEqual(copied["evidence"]["path"], str(sidecar))
            self.assertEqual(copied["vote_sidecar"]["path"], str(votes))
            self.assertIn(str(sidecar), opened); self.assertIn(str(votes), opened)

    def test_recovery_phase_dispatch_runs_only_pending_pipeline(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            store = attack_eval.AttackRunStore(output, config)
            called = []
            patches = {
                "receiver-clock": "phase_receiver_clock", "receiver-read": "phase_receiver_read",
                "baseline-extract-videoseal": "phase_baseline_extract",
                "baseline-extract-rivagan": "phase_baseline_extract", "quality": "phase_quality",
            }
            for phase, function in patches.items():
                with mock.patch.object(attack_eval, function, side_effect=lambda *a, p=phase, **k: called.append(p)):
                    attack_recovery.run_case_phase(store, config, phase, "pilot_01")
            self.assertEqual(called, list(patches))

    def test_reused_quality_video_is_probed_and_fully_decoded(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            store = attack_eval.AttackRunStore(output, config)
            row = next(item for item in store.data["quality_rows"] if item["quality_id"] == "pilot_01/PAYLOAD_NATIVE_vs_OFF_NATIVE")
            record = source / "run_state/quality.json"; record.write_text("{}")
            video = source / "run_state/quality.mp4"; video.write_bytes(b"mp4")
            row.update(status="EVALUATED", record={"path": str(record), "bytes": 2}, side_by_side={"path": str(video)})
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                stdout = json.dumps({"streams": [{"width": 1024, "height": 320, "nb_frames": 181}]}).encode() if command[0] == "ffprobe" else b""
                return types.SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
            with mock.patch.object(attack_recovery.subprocess, "run", side_effect=run):
                attack_recovery._validate_reused_rows(store, "pilot_01")
            self.assertTrue(any(command[0] == "ffprobe" for command in commands))
            self.assertTrue(any(command[0] == "ffmpeg" and "null" in command for command in commands))

    def test_evaluate_recomputes_after_prior_complete_marker(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source_fixture(directory)
            output = Path(directory) / "recovery"
            attack_recovery.initialize(CONFIG, source, output)
            store = attack_eval.AttackRunStore(output, config)
            store.data["recovery"]["phases"]["evaluate"] = {"status": "COMPLETE", "failures": []}
            store.save(); calls = []
            with mock.patch.object(attack_eval, "phase_evaluate", side_effect=lambda *a: calls.append("evaluate") or {}):
                attack_recovery.evaluate(store, config)
                attack_recovery.evaluate(store, config)
            self.assertEqual(calls, ["evaluate", "evaluate"])


if __name__ == "__main__":
    unittest.main()
