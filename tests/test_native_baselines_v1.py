"""CPU engineering fixtures only; never download weights or invoke real codec/model."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from main.baselines.native_paper_v1 import denominators, operations, read_tasks, validate_config
from runtime.wan.native_baseline_colab_v1 import safe_extract
from runtime.wan.native_baseline_storage_v1 import Journal, read_json, recover, reserve_local
from runtime.wan.native_baselines_v1 import DEFAULT_CONFIG, run


class FakeMedia:
    def __init__(self, missing=False):
        self.saved = {}
        self.calls = []
        self.missing = missing

    def read_source(self, root, source, shape, fps):
        self.calls.append("input")
        if self.missing:
            raise FileNotFoundError("fixture missing OFF")
        values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        return values

    def encode(self, rgb, path, fps):
        self.calls.append("encode")
        self.saved[str(path)] = rgb.copy()
        return {"fixture": True}

    def probe(self, path, shape, fps):
        self.calls.append("probe")
        return {"fixture": True}

    def decode(self, path, shape):
        self.calls.append("decode")
        return self.saved[str(path)].copy()


class FakeAdapter:
    def __init__(self, method, fail=None):
        self.method = method
        self.fail = fail
        self.counts = dict(embedder_calls=0, detector_calls=0)
        self.received = []

    def load(self):
        return dict(fixture=True, method=self.method)

    def embed(self, rgb, message, rectangle):
        self.counts["embedder_calls"] += 1
        return rgb.copy()

    def detect(self, rgb):
        self.counts["detector_calls"] += 1
        self.received.append(rgb.copy())
        if self.fail == self.counts["detector_calls"]:
            raise RuntimeError("intentional fixture detector failure")
        if self.method == "wam":
            return dict(bits=None, selected_pixels=0, sums=np.zeros(32), selected_means=None, mask=np.zeros(8192, dtype=np.uint8), mask_shape=[256, 256])
        logits = np.zeros((len(rgb), 257), dtype=np.float32)
        logits[:, 1:] = 1
        return dict(bits=[[1]*256 for _ in rgb], video_bits=[1]*256, logits=logits)

    def close(self):
        pass


class NativeBaselineChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = read_json(DEFAULT_CONFIG)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture_config(self):
        cfg = deepcopy(self.config)
        cfg["shape"] = [3, 8, 12, 3]
        cfg["crop_yxyx"] = [2, 3, 6, 9]
        return cfg

    def test_fixed_native_budget_and_no_old_path(self):
        validate_config(self.config)
        ds = denominators(self.config)
        self.assertEqual(ds["frame_rows"], 5792)
        self.assertEqual(ds["frame_bits"], dict(wam=92672, pixelseal=741376))
        ops = operations(self.config)
        for kind, count in [("load", 2), ("encode", 12), ("decode", 12), ("probe", 12),
                            ("embed", 728), ("detect", 2912), ("quality_pre", 8), ("quality_decoded", 8)]:
            self.assertEqual(sum(r["kind"] == kind for r in ops), count)
        self.assertTrue(all(s["relative_rgb"].startswith("m4_inputs/") for s in self.config["sources"]))

    def test_success_full_denominator_real_crop_boundary_and_read_only(self):
        cfg = self.fixture_config()
        adapters = {}
        def factory(method):
            adapters[method] = FakeAdapter(method)
            return adapters[method]
        output = self.root / "attempt01"
        with patch("runtime.wan.native_baselines_v1.validate_config", side_effect=lambda x: x):
            result = run(cfg, output, self.root / "work", self.root, adapter_factory=factory, media=FakeMedia())
        self.assertEqual(result["status"], "COMPLETED")
        ev = read_json(output / "evaluation.json")
        self.assertEqual(sum(r["planned_frames"] for r in ev["frame_summaries"]), 96)
        self.assertTrue(all(r["correct"]+r["wrong"]+r["unknown"] == r["planned_bits"] for r in ev["frame_summaries"]))
        self.assertEqual(len(ev["video_messages"]), 16)
        # The second view passed to the detector has actual cropped pixel dimensions/values.
        full, crop = adapters["pixelseal"].received[:2]
        np.testing.assert_array_equal(crop, full[:, 2:6, 3:9])
        self.assertEqual(crop.shape, (3, 4, 6, 3))
        before = {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        rerun = run(cfg, output, self.root / "work", self.root,
                    adapter_factory=lambda m: self.fail("model retry"), media=None)
        self.assertEqual(rerun["mode"], "READ_ONLY")
        self.assertEqual(before, {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()})
        self.assertLessEqual((output / "blind_terminated.json").stat().st_mtime_ns,
                             (output / "post_truth_access.json").stat().st_mtime_ns)

    def test_failed_detect_retains_partial_unknown_and_no_retry(self):
        cfg = self.fixture_config()
        output = self.root / "attempt"
        with patch("runtime.wan.native_baselines_v1.validate_config", side_effect=lambda x: x):
            result = run(cfg, output, self.root / "work", self.root,
                         adapter_factory=lambda m: FakeAdapter(m, fail=2), media=FakeMedia())
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(result["status_counts"]["FAILED"], 1)
        ev = read_json(output / "evaluation.json")
        self.assertEqual(sum(r["planned_frames"] for r in ev["frame_summaries"]), 96)
        self.assertTrue(all(r["correct"]+r["wrong"]+r["unknown"] == r["planned_bits"] for r in ev["frame_summaries"]))
        self.assertTrue(any(r["statuses"].get("NOT_STARTED") for r in ev["frame_summaries"]))
        self.assertEqual(sum(r["statuses"].get("FAILED", 0) for r in ev["frame_summaries"]), 1)

    def test_missing_off_does_not_load_model(self):
        cfg = self.fixture_config()
        with patch("runtime.wan.native_baselines_v1.validate_config", side_effect=lambda x: x):
            result = run(cfg, self.root / "attempt", self.root / "work", self.root,
                         adapter_factory=lambda m: self.fail("must not load model without inputs"), media=FakeMedia(missing=True))
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertIn("fixture missing OFF", result["failure"]["message"])
        self.assertEqual(result["physical_calls"]["load"], {"NOT_STARTED": 2})

    def test_interruption_accounting_and_unsafe_zip(self):
        output = self.root / "attempt"
        self.assertTrue(reserve_local(output))
        journal = Journal(output, [{"id": "only", "kind": "detect", "status": "NOT_STARTED"}])
        journal.event(dict(id="only", kind="detect", status="STARTED"))
        self.assertEqual(recover(output)["operations"][0]["status"], "INTERRUPTED_CONSUMED")
        self.assertFalse(reserve_local(output))
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "x")
        with self.assertRaises(ValueError):
            safe_extract(archive, self.root / "unpack")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_notebook_builder_sync_and_first_cell(self):
        from scripts.build_native_baselines_v1_notebook import build, ROOT
        (self.root / "notebooks").mkdir()
        built = json.loads(build(self.root).read_text())
        delivered = json.loads((ROOT / "notebooks/native_baselines_v1.ipynb").read_text())
        self.assertEqual(built, delivered)
        self.assertEqual("".join(built["cells"][0]["source"]), "from google.colab import drive\ndrive.mount('/content/drive')\n")
        for cell in built["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_native_adapter_single_detect_and_unequal_last_chunk(self):
        import torch
        from runtime.wan.native_baseline_models_v1 import NativeAdapter
        class PixelSpy:
            def __init__(self):
                self.calls = 0
            def detect(spy, x, **kwargs):
                spy.calls += 1
                self.assertTrue(kwargs["is_video"])
                self.assertEqual(kwargs["interpolation"], dict(mode="bilinear", align_corners=False, antialias=True))
                logits = torch.zeros(181, 257)
                logits[:160, 1:] = 0.15
                logits[160:, 1:] = -1
                return {"preds": logits}
        adapter = NativeAdapter("pixelseal", self.root, device="cpu")
        adapter.model = PixelSpy()
        result = adapter.detect(np.zeros((181, 2, 2, 3), dtype=np.uint8))
        self.assertEqual(adapter.model.calls, 1)
        self.assertEqual(result["video_bits"], [1]*256)
        self.assertEqual(result["bits"][-1], [0]*256)
        # An incorrect equal-chunk mean would be negative for this fixture.
        self.assertLess((5*0.15-1)/6, 0)

    def test_wam_raw_logits_native_callback_and_empty_unknown(self):
        import torch
        from runtime.wan.native_baseline_models_v1 import NativeAdapter
        class WamSpy:
            def __init__(self):
                self.empty = False
                self.near_boundary = False
            def detect(spy, x):
                self.assertEqual(list(x.shape), [1, 3, 2, 2])
                value = torch.full((1, 33, 256, 256), 0.25)
                value[:, :1] = -1
                if not spy.empty:
                    value[:, :1, 0, 0] = 1
                    value[:, 1:, 0, 0] = 0.5
                    value[:, 2::2, 0, 0] = 0.75
                    if spy.near_boundary:
                        value[:, :1, 0, 1] = 1
                        value[:, 1:, 0, 0] = 0.5
                        value[:, 1:, 0, 1] = torch.nextafter(torch.tensor(0.5), torch.tensor(1.0))
                return {"preds": value}
        adapter = NativeAdapter("wam", self.root, device="cpu")
        adapter.model = WamSpy()
        adapter.normalize = lambda x: x
        callbacks = []
        def native_callback(bits, masks, method):
            callbacks.append(method)
            self.assertEqual(method, "semihard")
            self.assertEqual(float(bits[0, 0, 0, 0]), 0.25 if adapter.model.empty else 0.5)
            selected = masks[0, 0] > 0.5
            if not selected.any():
                return torch.zeros((1, 32), dtype=torch.bool)
            return (bits[0, :, selected].mean(-1, dtype=float) > 0.5).unsqueeze(0)
        adapter.native_message = native_callback
        rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        result = adapter.detect(rgb)
        self.assertEqual(result["bits"], [0, 1]*16)
        self.assertEqual(result["selected_pixels"], 1)
        np.testing.assert_array_equal(result["sums"], [0.5, 0.75]*16)
        self.assertEqual(result["selected_means"], [0.5, 0.75]*16)
        adapter.model.empty = True
        result = adapter.detect(rgb)
        self.assertIsNone(result["bits"])
        self.assertEqual(result["selected_pixels"], 0)
        self.assertIsNone(result["selected_means"])
        adapter.model.empty = False
        adapter.model.near_boundary = True
        result = adapter.detect(rgb)
        self.assertEqual(result["bits"], [1]*32)
        self.assertTrue(all(x > 0.5 for x in result["selected_means"]))
        self.assertEqual([int(x > 0.5) for x in result["selected_means"]], result["bits"])
        self.assertEqual(len(callbacks), 3)

    def test_evaluation_write_failure_stops_cli_and_never_retries(self):
        from experiments.wan_state_clock.native_baselines_v1_run import main
        cfg = self.fixture_config()
        output = self.root / "attempt"
        with patch("runtime.wan.native_baselines_v1.validate_config", side_effect=lambda x: x), \
             patch("runtime.wan.native_baselines_v1.evaluate_saved", side_effect=OSError("fixture evaluation write failure")):
            result = run(cfg, output, self.root / "work", self.root,
                         adapter_factory=FakeAdapter, media=FakeMedia())
        self.assertEqual(result["physical_status"], "COMPLETED")
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(read_json(output / "result.json")["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(read_json(output / "attempt.json")["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertTrue((output / "blind_terminated.json").is_file())
        self.assertTrue((output / "evaluation_failure.json").is_file())
        with patch("experiments.wan_state_clock.native_baselines_v1_run.run", return_value=result), \
             patch("sys.argv", ["runner"]), patch("builtins.print"):
            self.assertEqual(main(), 1)
        with patch("runtime.wan.native_baselines_v1.evaluate_saved", side_effect=AssertionError("no re-evaluation")):
            self.assertEqual(run(cfg, output, self.root / "work", self.root)["mode"], "READ_ONLY")

    def test_colab_worker_exit_without_result_and_existing_specific_failure(self):
        from runtime.wan import native_baseline_colab_v1 as entry
        source = DEFAULT_CONFIG.parents[3]
        copies = []
        class CloudSpy:
            def directory(self, create=False):
                return "fixture-folder" if create else None
            def children(self, parent):
                return []
            def write_json(self, parent, name, value):
                copies.append((name, deepcopy(value)))
                return {"fixture": True}
        def fake_source(repo, destination, ref="main"):
            destination.mkdir(parents=True, exist_ok=True)
            if "watermark-anything" in repo:
                weight = destination / "checkpoints/wam_mit.pth"
                weight.parent.mkdir()
                weight.write_bytes(b"not-a-real-weight; no loader is invoked")
            return destination
        output = self.root / "attempt"
        class WorkerSpy:
            returncode = 7
            calls = 0
            existing_specific_failure = False
            result_status = "STOPPED_FIXED_ATTEMPT"
            malformed = False
            def __init__(self, *args, **kwargs):
                WorkerSpy.calls += 1
                if WorkerSpy.existing_specific_failure:
                    (output / "result.json").write_text(json.dumps(dict(status=WorkerSpy.result_status, failure={"message": "specific model failure"} if WorkerSpy.result_status != "COMPLETED" else None)))
                if WorkerSpy.malformed:
                    (output / "result.json").write_text('{"status":')
            def poll(self):
                return self.returncode
        with patch.object(entry, "Cloud", CloudSpy), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "tempwork"), patch.object(entry, "download_source", side_effect=fake_source), \
             patch.object(entry, "_install_dependencies", return_value={}), patch.object(entry.subprocess, "Popen", WorkerSpy), \
             patch("builtins.print"):
            result = entry.run_colab(source_directory=source)
            self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
            self.assertEqual(result["worker_returncode"], 7)
            self.assertEqual(read_json(output / "attempt.json")["status"], "STOPPED_FIXED_ATTEMPT")
            self.assertTrue((output / "planned_calls.json").is_file())
            self.assertTrue((output / "preparation_failure.json").is_file())
            self.assertTrue(any(name == "preparation_failure_cloud.json" for name, _ in copies))
            self.assertEqual(entry.run_colab(source_directory=source)["status"], "READ_ONLY")
            self.assertEqual(WorkerSpy.calls, 1)
        # Separate artificial attempt: an existing specific worker failure is preserved.
        output = self.root / "specific_attempt"
        WorkerSpy.existing_specific_failure = True
        with patch.object(entry, "Cloud", CloudSpy), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "tempwork2"), patch.object(entry, "download_source", side_effect=fake_source), \
             patch.object(entry, "_install_dependencies", return_value={}), patch.object(entry.subprocess, "Popen", WorkerSpy), \
             patch("builtins.print"):
            result = entry.run_colab(source_directory=source)
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(result["failure"]["message"], "specific model failure")
        self.assertFalse((output / "preparation_failure.json").exists())
        self.assertEqual(read_json(output / "attempt.json")["status"], "STOPPED_FIXED_ATTEMPT")
        # Display status comes from a readable final result, not merely an exit code.
        output = self.root / "completed_attempt"
        WorkerSpy.returncode = 0
        WorkerSpy.result_status = "COMPLETED"
        with patch.object(entry, "Cloud", CloudSpy), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "tempwork3"), patch.object(entry, "download_source", side_effect=fake_source), \
             patch.object(entry, "_install_dependencies", return_value={}), patch.object(entry.subprocess, "Popen", WorkerSpy), \
             patch("builtins.print"):
            result = entry.run_colab(source_directory=source)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(read_json(output / "attempt.json")["status"], "COMPLETED")
        # A truncated terminal JSON must not make the display hide the saved failure.
        output = self.root / "malformed_attempt"
        WorkerSpy.returncode = 7
        WorkerSpy.malformed = True
        with patch.object(entry, "Cloud", CloudSpy), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "tempwork4"), patch.object(entry, "download_source", side_effect=fake_source), \
             patch.object(entry, "_install_dependencies", return_value={}), patch.object(entry.subprocess, "Popen", WorkerSpy), \
             patch("builtins.print"):
            result = entry.run_colab(source_directory=source)
            entry.show_results()
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(read_json(output / "attempt.json")["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertTrue((output / "preparation_failure.json").is_file())
        self.assertEqual((output / "result.json").read_text(), '{"status":')


if __name__ == "__main__":
    unittest.main()
