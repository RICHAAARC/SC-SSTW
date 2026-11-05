"""Artificial CPU pixels/backends only. No network, real weights or codec."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main.baselines.native_paper_temporal_v2 import (CHANNELS, CONDITIONS, denominators, edit_maps,
    operations, physical_budget, read_tasks, validate_config)
from runtime.wan.native_baseline_storage_v1 import read_json, recover
from runtime.wan import native_baselines_temporal_v2 as runner


class MediaFixture:
    def __init__(self, fail=False):
        self.saved, self.calls, self.fail = {}, [], fail

    def read_source(self, root, source, shape, fps):
        self.calls.append(("input", source["id"]))
        if self.fail:
            raise FileNotFoundError("intentional missing artificial input")
        return np.broadcast_to(np.arange(shape[0], dtype=np.uint8)[:, None, None, None], shape).copy()

    def encode(self, rgb, path, fps):
        self.calls.append(("encode", str(path)))
        # Simulate changed received pixels; this is not a codec quality model.
        self.saved[str(path)] = rgb + np.uint8(1)
        return {"artificial": True}

    def probe(self, path, shape, fps):
        self.calls.append(("probe", str(path)))
        assert list(self.saved[str(path)].shape) == shape
        assert fps == 8
        return {"artificial": True, "frames": shape[0]}

    def decode(self, path, shape):
        self.calls.append(("decode", str(path)))
        assert list(self.saved[str(path)].shape) == shape
        return self.saved[str(path)].copy()


class AdapterFixture:
    def __init__(self, method, output, fail=None):
        self.method, self.output, self.fail = method, output, fail
        self.seen = []
        self.counts = dict(embedder_calls=0, embedder_frames=0, detector_calls=0, detector_frames=0)

    def load(self):
        assert (self.output / "edit_truth.json").is_file()
        return {"artificial": True}

    def embed(self, rgb, message, rectangle):
        self.counts["embedder_calls"] += 1
        self.counts["embedder_frames"] += len(rgb) if self.method == "wam" else (len(rgb)+7)//8
        return rgb + np.uint8(2)

    def detect(self, rgb):
        assert not (self.output / "post_truth_access.json").exists()
        self.seen.append(rgb[:, 0, 0, 0].tolist())
        self.counts["detector_calls"] += 1 if self.method == "wam" else (len(rgb)+31)//32
        self.counts["detector_frames"] += len(rgb)
        if self.fail and len(self.seen) == self.fail:
            raise RuntimeError("intentional artificial detector failure")
        if self.method == "wam":
            return dict(bits=None, selected_pixels=0, sums=np.zeros(32), selected_means=None,
                        mask=np.zeros(8192, dtype=np.uint8), mask_shape=[256, 256])
        logits = np.ones((len(rgb), 257), dtype=np.float32)
        return dict(bits=[[1]*256 for _ in rgb], video_bits=[1]*256, logits=logits)

    def close(self):
        pass


class TemporalChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read_json(runner.DEFAULT_CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def artificial_config(self):
        config = deepcopy(self.config)
        config["shape"] = [181, 2, 2, 3]
        config["crop_yxyx"] = [0, 0, 1, 1]
        return config

    def test_maps_and_fixed_budget(self):
        validate_config(self.config)
        maps = edit_maps(self.config)
        for host, spec in maps.items():
            m = spec["conditions"]
            self.assertEqual({k: len(v["indices"]) for k, v in m.items()}, CONDITIONS)
            d18 = set(m["DROP18"]["not_received"])
            d54 = set(m["DROP54"]["not_received"])
            self.assertTrue(d18 < d54)
            repeated = Counter(m["REPEAT_FRAME18"]["indices"])
            self.assertEqual({i for i, n in repeated.items() if n == 2}, d18)
            self.assertEqual(m["REPEAT_FRAME18"]["zero_deltas"], 18)
            self.assertEqual(m["REPLAY_BLOCK18"]["indices"][98:101], [98, 81, 82])
            self.assertEqual(m["REPLAY_BLOCK18"]["backward_deltas"], 1)
            self.assertEqual(m["BURST54"]["not_received"], list(range(63, 117)))
            self.assertEqual([m[f"TRIM_S{s}_L129"]["indices"][0] for s in range(26, 30)], [26, 27, 28, 29])
        tasks = read_tasks(self.config)
        self.assertEqual(len({t["id"] for t in tasks}), 352)
        self.assertEqual(sum(t["frames"] for t in tasks), 53600)
        ds = denominators(self.config)
        self.assertEqual(ds["frame_bits"], dict(wam=857600, pixelseal=6860800))
        self.assertEqual(ds["pixelseal_video_bits"], 45056)
        budget = physical_budget(self.config)
        self.assertEqual((budget["wam_embed_api"], budget["wam_detect_api"]), (724, 26800))
        self.assertEqual((budget["pixelseal_detect_api"], budget["pixelseal_detector_forward"]), (176, 960))
        counts = Counter(r["kind"] for r in operations(self.config))
        for key, n in {"load": 2, "encode": 144, "probe": 144, "decode": 144,
                       "embed": 728, "detect": 26976, "quality_precodec": 8,
                       "quality_G1_IDENTITY": 8, "quality_G2": 88}.items():
            self.assertEqual(counts[key], n)

    def test_full_matrix_artificial_calls_shared_off_truth_boundary_and_readonly(self):
        output = self.root / "attempt"
        adapters, media = {}, MediaFixture()
        def factory(method):
            adapters[method] = AdapterFixture(method, output)
            return adapters[method]
        with patch.object(runner, "validate_config", side_effect=lambda x: x), patch("builtins.print"):
            result = runner.run(self.artificial_config(), output, self.root / "work", self.root,
                                adapter_factory=factory, media=media)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["physical_calls"]["detect"], {"RETURNED": 26976})
        self.assertEqual(result["internal_counts"]["pixelseal"]["detector_calls"], 960)
        self.assertEqual(result["internal_counts"]["wam"]["detector_frames"], 26800)
        self.assertEqual(len([p for op, p in media.calls if op == "encode" and "/OFF/" in p]), 48)
        self.assertEqual(Counter(op for op, _ in media.calls), dict(input=4, encode=144, probe=144, decode=144))
        ev = read_json(output / "evaluation.json")
        details = ev["source_condition_channel_details"]
        self.assertEqual(len(details), 352)
        self.assertTrue(all(d["correct"]+d["wrong"]+d["unknown"] == d["planned_bits"] for d in details))
        self.assertEqual(len(ev["video_messages"]), 176)
        self.assertEqual(len(read_json(output / "quality.json")), 104)
        trim = next(r for r in ev["source_retention"] if r["condition"] == "TRIM_S26_L129")
        self.assertEqual((trim["not_received_source_frames"], trim["received_instances"]), (52, 129))
        self.assertTrue(all(d["unknown"] == d["planned_bits"] for d in details if d["method"] == "wam"))
        self.assertFalse(ev["read_errors"])
        # Ordering per condition: WM G1/G2 then shared OFF G1/G2; no reuse of logits.
        reads = adapters["pixelseal"].seen
        self.assertEqual([x[0] for x in reads[:4]], [3, 4, 1, 2])
        self.assertEqual([len(x) for x in reads[4:8]], [129]*4)
        self.assertEqual([x[0] for x in reads[4:8]], [29, 30, 27, 28])
        replay = reads[40]
        self.assertEqual(replay[98:101], [101, 84, 85])
        before = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()]
        with patch.object(runner, "evaluate_saved", side_effect=AssertionError("no evaluation retry")):
            result = runner.run(self.config, output, self.root / "work", self.root,
                                adapter_factory=lambda m: self.fail("no model retry"))
        self.assertEqual(result["mode"], "READ_ONLY")
        after = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()]
        self.assertEqual(before, after)

    def test_failure_retains_all_planned_slots_and_partial_observations(self):
        output = self.root / "failed"
        with patch.object(runner, "validate_config", side_effect=lambda x: x), patch("builtins.print"):
            result = runner.run(self.artificial_config(), output, self.root / "work", self.root,
                adapter_factory=lambda m: AdapterFixture(m, output, fail=2), media=MediaFixture())
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual(result["physical_calls"]["detect"], {"RETURNED": 1, "FAILED": 1, "NOT_STARTED": 26974})
        ev = read_json(output / "evaluation.json")
        self.assertEqual(sum(d["planned_frames"] for d in ev["source_condition_channel_details"]), 53600)
        self.assertEqual(sum(d["unknown"] for d in ev["source_condition_channel_details"]), 7718400)
        self.assertTrue((output / "blind_terminated.json").is_file())
        self.assertTrue((output / "post_truth_access.json").is_file())

    def test_evaluation_failure_stays_consumed_and_cli_nonzero(self):
        from experiments.wan_state_clock.native_baselines_temporal_v2_run import main
        output = self.root / "post_failed"
        with patch.object(runner, "validate_config", side_effect=lambda x: x), \
             patch.object(runner, "evaluate_saved", side_effect=OSError("artificial evaluation save failure")):
            result = runner.run(self.artificial_config(), output, self.root / "work", self.root,
                                media=MediaFixture(fail=True))
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertIn("evaluation_failure", result)
        self.assertTrue((output / "evaluation_failure.json").is_file())
        with patch("experiments.wan_state_clock.native_baselines_temporal_v2_run.run", return_value=result), \
             patch("sys.argv", ["runner"]), patch("builtins.print"):
            self.assertEqual(main(), 1)
        self.assertEqual(recover(output)["mode"], "READ_ONLY")

    def test_native_received_frame_mean_preserves_tail_and_repeats(self):
        import torch
        from runtime.wan.native_baseline_models_v1 import NativeAdapter
        class Backend:
            def detect(self, x, **kwargs):
                self.chunks = [len(x[i:i+32]) for i in range(0, len(x), 32)]
                preds = torch.zeros((len(x), 257))
                preds[:, 1:] = x[:, 0, 0, 0, None]*255-100
                return {"preds": preds}
        adapter = NativeAdapter("pixelseal", self.root, device="cpu")
        adapter.model = Backend()
        # 128 negative versus one positive: mean chunk means would give the wrong sign.
        rgb = np.full((129, 1, 1, 3), 98, dtype=np.uint8)
        rgb[-1] = 255
        pred = adapter.detect(rgb)
        self.assertEqual(adapter.model.chunks, [32, 32, 32, 32, 1])
        self.assertEqual(pred["video_bits"], [0]*256)
        # Duplicated positives retain their weight instead of being deduplicated.
        rgb = np.array([0, 160, 160], dtype=np.uint8)[:, None, None, None].repeat(3, axis=-1)
        self.assertEqual(adapter.detect(rgb)["video_bits"], [1]*256)

    def test_notebook_builder_and_standalone_reservation_counts(self):
        from scripts.build_native_baselines_temporal_v2_notebook import build
        from runtime.wan.native_baseline_temporal_colab_v2 import reservation_plan
        root = runner.DEFAULT_CONFIG.parents[3]
        (self.root / "notebooks").mkdir()
        built = build(self.root)
        self.assertEqual(built.read_bytes(), (root / "notebooks/native_baselines_temporal_v2.ipynb").read_bytes())
        nb = json.loads(built.read_text())
        self.assertEqual("".join(nb["cells"][0]["source"]), "from google.colab import drive\ndrive.mount('/content/drive')\n")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
        planned = reservation_plan()
        self.assertEqual([{k: v for k, v in t.items() if k != "status"} for t in planned], read_tasks(self.config))
        self.assertEqual(sum(t["frames"]*t["nbits"] for t in planned), 7718400)

    def test_bootstrap_setup_failure_full_slots_and_readonly(self):
        from runtime.wan import native_baseline_temporal_colab_v2 as entry
        output, copies = self.root / "colab", []
        class Cloud:
            def directory(self, create=False): return "folder" if create else None
            def children(self, parent): return []
            def write_json(self, parent, name, value): copies.append((name, deepcopy(value)))
        with patch.object(entry, "Cloud", Cloud), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "cache"), patch("builtins.print"), \
             patch.object(entry, "download_source", side_effect=OSError("artificial download failure")) as download:
            result = entry.run_colab()
            self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
            self.assertEqual(len(read_json(output / "planned_read_tasks.json")), 352)
            self.assertEqual(result["planned_frame_rows"], 53600)
            self.assertEqual(copies[0][0], "cloud_start.json")
            self.assertEqual(len(copies[0][1]["planned_read_tasks"]), 352)
            self.assertEqual(entry.run_colab()["status"], "READ_ONLY")
            self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
