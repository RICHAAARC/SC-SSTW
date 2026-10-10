import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import json
from experiments.paper_results_v1.attack_eval import (
    AttackRunStore, _reference_flow_fluctuation, phase_baseline_embed, phase_baseline_extract, validate_config,
)
from experiments.paper_results_v1.attack_matrix import apply_edit_rgb8, fixed_attack_specs


class AttackNumpyTests(unittest.TestCase):
    def test_numeric_rounding_and_fixed_lengths(self):
        source = np.empty((181, 320, 512, 3), np.uint8)
        source[:] = np.arange(181, dtype=np.uint8)[:, None, None, None]
        specs = {row["attack_id"]: row for row in fixed_attack_specs()}
        crop, _ = apply_edit_rgb8(source, specs["crop37_126"])
        self.assertEqual((len(crop), int(crop[0, 0, 0, 0]), int(crop[-1, 0, 0, 0])), (89, 37, 125))
        del crop
        speed, _ = apply_edit_rgb8(source, specs["speed075"])
        self.assertEqual((len(speed), int(speed[1, 0, 0, 0])), (241, 1))
        del speed
        mean, _ = apply_edit_rgb8(source, specs["mean3"])
        self.assertEqual((int(mean[0, 0, 0, 0]), int(mean[180, 0, 0, 0])), (0, 180))
        del mean
        interp, _ = apply_edit_rgb8(source, specs["interp_g0"])
        self.assertEqual(len(interp), 199)

    def test_videoseal_phase_keeps_unique_sidecar_per_attack(self):
        with tempfile.TemporaryDirectory() as directory:
            config = validate_config(json.loads((Path(__file__).resolve().parents[1] / "experiments/paper_results_v1/attack_eval.adopted.json").read_text()))
            store = AttackRunStore(Path(directory) / "run", config, create=True)
            artifacts = [row for row in store.data["artifacts"] if row["case_id"] == "pilot_01" and row["artifact_id"].startswith("pilot_01/videoseal/") and row["kind"] == "TEMPORAL_OBSERVATION"]
            for index, row in enumerate(artifacts): row.update(status="AVAILABLE", shape=[index + 1, 320, 512, 3])
            store.save()
            class Adapter:
                native_output_store = None
                count = 0
                def extract(self, media):
                    self.count += 1
                    values = {"preds": np.full((self.count, 2), float(self.count))}
                    ref = self.native_output_store("videoseal_detect_output", values, {})
                    return {"storage": "LOSSLESS_SIDECAR", "lossless_native_output": ref}
            adapter = Adapter()
            with mock.patch("experiments.paper_results_v1.attack_eval._adapter", return_value=adapter), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=object(),
            ):
                phase_baseline_extract(store, config, "pilot_01", "videoseal")
            rows = [row for row in store.data["baseline_rows"] if row["case_id"] == "pilot_01" and row["method"] == "videoseal"]
            first = json.loads(Path(rows[0]["extract_record"]["path"]).read_text())["lossless_native_output"]
            second = json.loads(Path(rows[1]["extract_record"]["path"]).read_text())["lossless_native_output"]
            self.assertNotEqual(first["uri"], second["uri"])
            with np.load(first["uri"], allow_pickle=False) as values:
                self.assertEqual(values["preds"].tolist(), [[1.0, 1.0]])
            with np.load(second["uri"], allow_pickle=False) as values:
                self.assertEqual(values["preds"].tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_rivagan_embed_phase_converts_torch_like_input_to_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            config = validate_config(json.loads((Path(__file__).resolve().parents[1] / "experiments/paper_results_v1/attack_eval.adopted.json").read_text()))
            store = AttackRunStore(Path(directory) / "run", config, create=True)
            source = store.artifact("pilot_01/OFF_NATIVE/PRE"); source.update(status="AVAILABLE")
            class Input:
                def numpy(self): return np.zeros((1, 2, 2, 3), np.uint8)
            class Adapter:
                def embed(self, media, bits):
                    if not isinstance(media, np.ndarray): raise AssertionError("RivaGAN did not receive ndarray")
                    return media, {"method": "rivagan", "native_message": bits}
            with mock.patch("experiments.paper_results_v1.attack_eval._adapter", return_value=Adapter()), mock.patch(
                "experiments.paper_results_v1.attack_eval._load_rgb8", return_value=Input(),
            ), mock.patch(
                "experiments.paper_results_v1.attack_eval._save_rgb8",
                return_value={"path": "fixture.rgb8", "shape": [1, 2, 2, 3], "dtype": "uint8", "bytes": 12},
            ):
                phase_baseline_embed(store, config, "pilot_01", "rivagan")
            self.assertEqual(store.artifact("pilot_01/rivagan/NATIVE_PRE")["status"], "AVAILABLE")

    def test_reference_flow_known_translation_flicker_and_invalid_coverage(self):
        # Patch OpenCV so this CPU fixture verifies warp direction/masking and
        # fluctuation arithmetic without depending on a system OpenCV wheel.
        import sys, types
        def remap(value, map_x, map_y, *args, **kwargs):
            x = np.rint(map_x).astype(int); y = np.rint(map_y).astype(int)
            output = np.zeros_like(value)
            valid = (x >= 0) & (x < value.shape[1]) & (y >= 0) & (y < value.shape[0])
            output[valid] = value[y[valid], x[valid]]
            return output
        fake = types.SimpleNamespace(
            COLOR_RGB2GRAY=1, INTER_LINEAR=1, BORDER_CONSTANT=0,
            cvtColor=lambda value, _: value[..., 0],
            calcOpticalFlowFarneback=lambda current, prior, *args: np.dstack((np.full(current.shape, -1.0, np.float32), np.zeros(current.shape, np.float32))),
            remap=remap,
        )
        old = sys.modules.get("cv2"); sys.modules["cv2"] = fake
        try:
            ref = np.zeros((9, 320, 512, 3), np.uint8)
            candidate = ref.copy()
            pattern = (np.arange(512, dtype=np.uint16) % 200).astype(np.uint8)
            for frame in range(9):
                candidate[frame, :, :, :] = np.roll(pattern, frame)[None, :, None]
            stable = _reference_flow_fluctuation(ref, candidate, 8)
            self.assertEqual(stable["status"], "EVALUATED")
            self.assertEqual(stable["mean"], 0.0)
            wrong = fake.calcOpticalFlowFarneback
            fake.calcOpticalFlowFarneback = lambda current, prior, *args: np.dstack((np.full(current.shape, 1.0, np.float32), np.zeros(current.shape, np.float32)))
            self.assertGreater(_reference_flow_fluctuation(ref, candidate, 8)["mean"], 0.0)
            fake.calcOpticalFlowFarneback = wrong
            candidate[4:] = np.clip(candidate[4:].astype(np.uint16) + 20, 0, 255).astype(np.uint8)
            flicker = _reference_flow_fluctuation(ref, candidate, 8)
            self.assertGreater(flicker["mean"], 0.0)
            self.assertEqual(flicker["transitions"][3]["mean_absolute_residual_delta"], 20.0)
            fake.calcOpticalFlowFarneback = lambda current, prior, *args: np.dstack((np.full(current.shape, -1000.0, np.float32), np.zeros(current.shape, np.float32)))
            invalid = _reference_flow_fluctuation(ref, candidate, 8)
            self.assertEqual(invalid["status"], "NO_VALID_COVERAGE")
        finally:
            if old is None: del sys.modules["cv2"]
            else: sys.modules["cv2"] = old


if __name__ == "__main__":
    unittest.main()
