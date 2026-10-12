"""CPU fault injection: process death, disposable /content, no model/media calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from experiments.paper_results_v1 import attack_eval, confirmation_v1 as confirmation
from experiments.paper_results_v1 import resume_media, resume_state

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/paper_results_v1/confirmation_v1.adopted.json"


class TensorStub:
    def float(self): return self
    def div(self, _): return self
    def detach(self): return self
    def cpu(self): return self


def latent_save(path, _value):
    resume_state.atomic_bytes(path, b'{"cpu_stub_latent":true}')


def latent_load(path):
    json.loads(path.read_text())
    return TensorStub()


def result(bit):
    return {"status": "READ", "R": 1, "decoded_bits": [bit] * 32,
            "votes": [{"ones": bit, "zeros": 1-bit} for _ in range(32)],
            "signed_votes": [[[bit] * 8] for _ in range(4)],
            "zero_mask": [[[False] * 8] for _ in range(4)],
            "bit_rows": [{"bit_index": i, "ones": bit, "zeros": 1-bit, "count": 1, "decoded": bit} for i in range(32)],
            "time_bit_rows": [], "original_reader_match": True}


def build_fixture(root):
    config = json.loads(CONFIG.read_text())
    attack = attack_eval.AttackRunStore(root / "attack", config, create=True)
    store = confirmation.initialize(attack.output, root / "confirmation", config, temp_root=root / "content")
    chosen = [row for row in store.rows("confirm_01") if row["arm"] == "OFF_NATIVE" and row["attack_id"] == "full" and row["receiver"] == "CENTERED_U"]
    for row in store.data["rows"]:
        if row["case_id"].startswith("confirm_"):
            row["status"] = "UNSUPPORTED"
    for row in chosen:
        row.update(status="PENDING_READ", physical_id="map_0000", operation={
            "received_index_map": [0,1,2,3,4], "output_frames": 5, "source_coordinate_map": [0,1,2,3,4]})
    store.data["physical_inputs"] = [{"physical_id": "map_0000", "artifact_id": "confirm_01/OFF_NATIVE/full/RECEIVED",
        "received_index_map": [0,1,2,3,4], "output_frames": 5, "status": "PLANNED",
        "logical_slots": [row["slot_id"] for row in chosen], "key_reads": {}}]
    store.save()
    return config, store


def worker(root, fault):
    root = Path(root)
    config = json.loads(CONFIG.read_text())
    store = confirmation.ConfirmationStore(root / "confirmation", config, temp_root=root / "content")
    def event(name):
        with (root / "calls.txt").open("a") as stream:
            stream.write(name + "\n"); stream.flush(); os.fsync(stream.fileno())
    def encode(_model, value, encode_observer):
        event("encode")
        encode_observer("STARTED", None)
        if fault == "during_vae": os._exit(23)
        encode_observer("RETURNED", None)
        return value
    def save(path, value):
        latent_save(path, value)
        if fault == "after_vae": os._exit(23)
    def read(_value, key, *_args):
        label = "K0" if key == config["keys"]["K0"] else "K1"
        if fault == "after_k0" and label == "K1": os._exit(23)
        event(label)
        return result(int(label == "K0"))
    if fault == "before_vae":
        def load(): os._exit(23)
    else:
        load = lambda: object()
    fake_np = types.SimpleNamespace(savez_compressed=lambda stream, **arrays: stream.write(json.dumps(arrays).encode()))
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
    with mock.patch.dict(sys.modules, {"numpy": fake_np, "torch": fake_torch}):
        confirmation.read_case(store, "confirm_01", load_vae=load, encode=encode,
                               payload_read=read, load_rgb=lambda *_: object(), operate=lambda *_: TensorStub(),
                               save_latent=save, load_latent=latent_load)


class CheckpointTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "CPU tensor/NPZ test requires existing torch and numpy")
    def test_actual_cpu_latent_and_vote_files_sync_before_replace(self):
        import torch
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); config,store=build_fixture(root)
            events=[]
            original_sync,original_replace=os.fsync,os.replace
            def sync(fd): events.append("sync"); return original_sync(fd)
            def replace(source,target): events.append(("replace",str(target))); return original_replace(source,target)
            latent=root/"latent.pt"
            with mock.patch.object(os,"fsync",side_effect=sync),mock.patch.object(os,"replace",side_effect=replace):
                confirmation._save_latent(latent,torch.zeros((1,16,2,4,4)))
                self.assertEqual(events[-2:],["sync",("replace",str(latent))])
                events.clear()
                group=store.data["physical_inputs"][0]
                confirmation._save_key_read(store,group,"K0",result(1))
                sidecar=store.output/"physical/map_0000/K0/votes.npz"
                replaced=events.index(("replace",str(sidecar)))
                self.assertEqual(events[replaced-1],"sync")
            self.assertEqual(tuple(confirmation._load_latent(latent).shape),(1,16,2,4,4))
            with np.load(sidecar,allow_pickle=False) as arrays:
                self.assertEqual(arrays["signed_votes"].shape,(4,1,8))
            with self.assertRaisesRegex(ValueError,"finite"):
                confirmation._save_latent(latent,torch.full((1,16,2,4,4),float("nan")))

    def test_unit_replay_preserves_fixed_plan_and_skips_malformed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            value = {"phases": {}, "rows": [{"slot_id":"one", "case_id":"confirm_01", "status":"PLANNED"}], "physical_inputs":[]}
            journal = resume_state.StateJournal(path); journal.save(value)
            folder = path.parent / "committed_units/state/rows"; folder.mkdir(parents=True)
            for name, row in (("unknown", {"slot_id":"unknown", "status":"READ"}),
                              ("missing-status", {"slot_id":"one", "case_id":"confirm_01"}),
                              ("missing-core", {"slot_id":"one", "status":"READ"})):
                (folder / (name+".json")).write_text(json.dumps(row))
            with self.assertWarns(RuntimeWarning): loaded = resume_state.read_state(path)
            self.assertEqual(loaded["rows"], value["rows"])
            self.assertEqual(len(loaded["rows"]),1)

    def test_microcommit_survives_death_before_primary_replace_with_new_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            value={"phases":{}, "rows":[{"slot_id":"one","status":"PLANNED"}],"physical_inputs":[]}
            journal=resume_state.StateJournal(path); journal.save(value)
            value["physical_inputs"].append({"physical_id":"map_0000","artifact_id":"source","status":"PLANNED",
                "received_index_map":[0,1,2,3,4],"output_frames":5,"logical_slots":["one"],"key_reads":{}})
            value["rows"][0].update(status="PENDING_READ",physical_id="map_0000")
            original=resume_state.atomic_bytes
            def stop_before_primary(target,payload):
                if Path(target)==path: raise KeyboardInterrupt("disk boundary")
                original(target,payload)
            with mock.patch.object(resume_state,"atomic_bytes",side_effect=stop_before_primary), self.assertRaises(KeyboardInterrupt):
                journal.save(value)
            loaded=resume_state.read_state(path)
            self.assertEqual(loaded["rows"][0]["physical_id"],loaded["physical_inputs"][0]["physical_id"])
            self.assertEqual(loaded["physical_inputs"][0]["logical_slots"],["one"])

    def test_reload_does_not_rewrite_unchanged_unit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            value={"phases":{},"rows":[{"slot_id":"one","status":"READ"}]}
            journal=resume_state.StateJournal(path); journal.save(value)
            second=resume_state.StateJournal(path); loaded=resume_state.read_state(path); second.seed_cache(loaded)
            with mock.patch.object(resume_state,"atomic_bytes",wraps=resume_state.atomic_bytes) as writes:
                second.save(loaded)
            self.assertFalse(any("committed_units" in str(call.args[0]) for call in writes.call_args_list))

    def test_read_without_prepared_groups_remains_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); config, store=build_fixture(root)
            store.data["physical_inputs"]=[]
            store.save()
            confirmation.read_case(store,"confirm_01")
            self.assertEqual(store.data["phases"]["read"]["confirm_01"]["status"],"PARTIAL")

    def test_hard_exit_and_new_process_resume_at_each_payload_boundary(self):
        for fault, encodes in (("before_vae",1), ("during_vae",2), ("after_vae",1), ("after_k0",1)):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); config, store = build_fixture(root)
                before_negative = copy.deepcopy(next(row for row in store.data["rows"] if row["status"] == "UNSUPPORTED"))
                first = subprocess.run([sys.executable, "-B", "-m", __name__, "worker", str(root), fault], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(first.returncode, 23, first.stderr)
                # /content is disposable; the latent checkpoint is under the durable run.
                (root / "content").mkdir(exist_ok=True)
                shutil.rmtree(root / "content")
                interrupted = confirmation.ConfirmationStore(store.output, config)
                if fault == "after_k0":
                    confirmation.report(interrupted)
                    evaluated = copy.deepcopy(next(row for row in interrupted.rows("confirm_01", key_label="K0") if row["status"] == "EVALUATED"))
                second = subprocess.run([sys.executable, "-B", "-m", __name__, "worker", str(root), "none"], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(second.returncode, 0, second.stderr)
                final = confirmation.ConfirmationStore(store.output, config)
                calls = (root / "calls.txt").read_text().splitlines()
                self.assertEqual(calls.count("encode"), encodes)
                self.assertEqual(calls.count("K0"), 1)
                self.assertEqual(calls.count("K1"), 1)
                self.assertEqual(next(row for row in final.data["rows"] if row["slot_id"] == before_negative["slot_id"]), before_negative)
                self.assertFalse((store.output / "physical/map_0000/latent.pt").exists())
                if fault == "after_k0":
                    self.assertEqual(next(row for row in final.data["rows"] if row["slot_id"] == evaluated["slot_id"]), evaluated)
                self.assertEqual(len(final.data["physical_inputs"][0].get("encode_attempts", [])), encodes)
                self.assertEqual(final.data["fixed_denominator"]["confirmation_rows"], 2880)

    def test_periodic_snapshot_and_corrupt_state_fallback_replays_unit_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_state.json"
            journal = resume_state.StateJournal(path, interval=0.03)
            value = {"phases": {}, "rows": [{"slot_id":"one", "status":"READ", "decoded_bits":[0]*32},
                                              {"slot_id":"two", "status":"PLANNED"}]}
            journal.save(value)
            with journal.periodic():
                time.sleep(0.09)
                snapshot = path.parent / "checkpoints/run_state/latest.json"
                self.assertTrue(snapshot.is_file())
                self.assertEqual(json.loads(snapshot.read_text())["rows"][0]["decoded_bits"], [0]*32)
            value["rows"][1].update(status="EVALUATED", decoded_bits=[1]*32)
            journal.save(value)
            # Simulate a damaged newest whole state. Newer micro-commit remains usable.
            path.write_text("{")
            with self.assertWarns(RuntimeWarning):
                loaded = resume_state.read_state(path)
            self.assertEqual({row["slot_id"] for row in loaded["rows"]}, {"one", "two"})
            self.assertEqual(loaded["rows"][1]["status"], "EVALUATED")
            self.assertEqual(loaded["rows"][1]["decoded_bits"], [1]*32)
            self.assertTrue(loaded["state_recovery_events"])
            self.assertIn("recoverable_checkpoint", json.loads((path.parent / "progress.json").read_text()))

    def test_full_alias_restores_from_codec_record_after_content_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, store = build_fixture(root)
            source = {"artifact_id":"source", "path":str(root / "lost.rgb8"), "shape":[5,320,512,3],
                      "codec_record":{"path":str(root / "codec.json")}}
            full = {"artifact_id":"full", "full_alias_of":"source", "path":str(root / "lost.rgb8"), "shape":[5,320,512,3]}
            original_full = copy.deepcopy(full)
            (root / "codec.json").write_text(json.dumps({"mp4":{"path":str(root / "published.mp4")}}))
            (root / "published.mp4").write_bytes(b"fixture-not-a-video")
            decoded = []
            def loader(row): return Path(row["path"]).read_bytes()
            def decode(mp4, target, expected_frames):
                decoded.append((mp4,expected_frames))
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"restored")
                return None, {"path":str(target)}
            value = resume_media.load_rgb(store, full, artifacts=[source,full], loader=loader, decode=decode, read_json=lambda p:json.loads(Path(p).read_text()))
            self.assertEqual(value,b"restored")
            self.assertEqual(decoded,[(root / "published.mp4",5)])
            self.assertEqual(resume_media.load_rgb(store, full, artifacts=[source,full], loader=loader, decode=decode), b"restored")
            self.assertEqual(len(decoded),1)
            self.assertEqual(resume_media.load_rgb(store, original_full, artifacts=[source,original_full], loader=loader, decode=decode), b"restored")
            self.assertEqual(len(decoded),1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker": worker(sys.argv[2], sys.argv[3])
    else: unittest.main()
