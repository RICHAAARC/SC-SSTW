"""CPU array fixtures only: no generated video and no scientific denominator."""
import json
from pathlib import Path
import tempfile
import unittest

from main.sc_sstw.aisb import make_double_redundant_templates
from main.sc_sstw.public_candidates import PublicBurst, freeze_public_bursts, read_frozen_bursts, candidate_digest, canonical_bytes
from main.sc_sstw.public_scan import scan_public_q
from runtime.stage1.observation import ObserverConfig, Observation, observe_frames, repeat_error
from experiments.stage1.run_observation import run_manifest


class Stage1Engineering(unittest.TestCase):
    def test_manifest_array_adapter_to_public_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fixture"
            path.write_bytes(b"not a video; injected in-memory frames")
            template = make_double_redundant_templates()[0]
            manifest = {"samples": [{"sample_id": "array", "path": str(path), "split": "development", "content_category": "engineering-array"}],
                        "observer": {"pixel_delta": 1, "min_support_fraction": 0.01, "max_support_fraction": 0.5},
                        "decode": {"sample_hz": 2, "max_frames": 20, "max_pixels": 16, "timeout_seconds": 1},
                        "public_acquisition": {"templates": [{"template_id": template.template_id, "points": template.points}],
                                               "missing_sets": [[], [1]], "max_evaluations": 20, "max_retained": 1}}
            frames = [bytes([100 if k == i % 16 else 0 for k in range(16)]) for i in range(14)]
            result = run_manifest(manifest, Path(tmp) / "out", decoder=lambda *a, **kw: (4, 4, list(frames)))
            row = result["rows"][0]
            self.assertEqual(row["candidate_freeze"], "ENGINEERING_FROZEN_READBACK")
            self.assertEqual(row["public_acquisition"]["evaluated_count"], 7)
            self.assertEqual(row["public_acquisition"]["records"][0]["status"], "INVALID_OBSERVATION_WINDOW")
            frozen = (Path(tmp) / "out" / row["candidate_file"]).read_bytes()
            self.assertEqual(candidate_digest(frozen), row["candidate_sha256"])
            self.assertEqual(len(read_frozen_bursts(frozen)), 1)

    def test_freeze_ties_roundtrip_and_tampering(self):
        a = PublicBurst("a", 1, 6, 6, (), -0.0)
        b = PublicBurst("b", 0, 6, 6, (), 0)
        frozen = freeze_public_bursts([b, a], budget=1, enumeration_complete=True)
        self.assertEqual(frozen, freeze_public_bursts([a, b], budget=1, enumeration_complete=True))
        self.assertEqual(read_frozen_bursts(frozen), (a,))
        self.assertEqual(len(candidate_digest(frozen)), 64)
        bad = json.loads(frozen)
        bad["dropped_count"] = 7
        with self.assertRaises(ValueError):
            read_frozen_bursts(json.dumps(bad).encode())
        with self.assertRaises(ValueError):
            freeze_public_bursts([a, a], budget=2, enumeration_complete=True)
        with self.assertRaises(ValueError):
            freeze_public_bursts([a], budget=1, enumeration_complete=False)

    def test_frozen_metadata_rejects_bool_integer_substitution(self):
        candidates = [PublicBurst("a", 0, 6, 6, (), 0), PublicBurst("b", 0, 6, 6, (), 0)]
        frozen = freeze_public_bursts(candidates, budget=1, enumeration_complete=True)
        for field, replacement in (("after_count", True), ("dropped_count", True),
                                   ("enumeration_complete", 1)):
            with self.subTest(field=field):
                payload = json.loads(frozen)
                payload[field] = replacement
                with self.assertRaises(ValueError):
                    read_frozen_bursts(canonical_bytes(payload))

    def test_no_motion_nonfinite_and_repeat(self):
        config = ObserverConfig(1, 0.01, 0.5)
        frames = [bytes(16), bytes([0, 0, 0, 0, 0, 100] + [0] * 10)]
        rows = observe_frames(frames, width=4, height=4, sample_hz=2, config=config)
        self.assertFalse(rows[0].valid)
        self.assertEqual(rows[1].q, (1 / 3, 1 / 3))
        self.assertTrue(repeat_error(rows, observe_frames(list(frames), width=4, height=4, sample_hz=2, config=config))["all_observation_fields_equal"])
        static = observe_frames([bytes(16)] * 3, width=4, height=4, sample_hz=2, config=config)
        self.assertFalse(any(r.valid for r in static))
        cut = observe_frames([bytes(16), bytes([255] * 16)], width=4, height=4, sample_hz=2, config=config)
        self.assertEqual(cut[1].reason, "GLOBAL_CHANGE_OR_CUT")
        with self.assertRaises(ValueError):
            Observation(0, 0, (float("nan"), 0), True, "VALID", 1, 1)

    def test_scan_budget_invalid_and_constant_are_not_pass(self):
        templates = make_double_redundant_templates()
        args = dict(templates=templates, missing_sets=((),), max_retained=2)
        complete = scan_public_q([(0.0, 0.0)] * 13, max_evaluations=6, **args)
        self.assertTrue(complete["enumeration_complete"])
        self.assertEqual(complete["scored_count"], 6)
        self.assertEqual(complete["records"][0]["scatter"], 0)
        self.assertEqual(complete["records"][0]["aisb_validity"], "UNDETERMINED_NO_FROZEN_VALIDITY_RULE")
        stopped = scan_public_q([(0.0, 0.0)] * 13, max_evaluations=1, **args)
        self.assertFalse(stopped["enumeration_complete"])
        self.assertIsNone(stopped["frozen"])
        invalid = scan_public_q([None] * 12, max_evaluations=3, **args)
        self.assertEqual(invalid["scored_count"], 0)

    def test_fixed_denominator_missing_and_repeat_decode_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fixture"
            path.write_bytes(b"array decoder fixture, not a video")
            manifest = {"samples": [
                {"sample_id": "missing", "path": str(Path(tmp) / "absent"), "split": "validation", "content_category": "single"},
                {"sample_id": "repeat_failure", "path": str(path), "split": "development", "content_category": "single"}],
                "observer": {"pixel_delta": 1, "min_support_fraction": 0.01, "max_support_fraction": 0.5},
                "decode": {"sample_hz": 2, "max_frames": 3, "max_pixels": 16, "timeout_seconds": 1}}
            calls = []
            def decoder(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("fixture second independent decode failure")
                return 4, 4, [bytes(16), bytes([0, 0, 0, 0, 0, 100] + [0] * 10)]
            result = run_manifest(manifest, Path(tmp) / "out", decoder=decoder)
            self.assertEqual(result["fixed_denominator"], 2)
            self.assertTrue(result["all_ids_reconciled"])
            self.assertTrue(all(r["terminal"] == "OPERATIONAL_BLOCKED" for r in result["rows"]))
            self.assertEqual(len(result["rows"][1]["repeats"]), 1)
            with self.assertRaises(FileExistsError):
                run_manifest(manifest, Path(tmp) / "out", decoder=decoder)


if __name__ == "__main__":
    unittest.main()
