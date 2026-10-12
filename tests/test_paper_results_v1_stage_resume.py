"""Exercise real phase dispatch with CPU stubs and uncatchable process exits.

Only heavyweight model/array/codec boundaries are stubs. Plans, loops, state
stores, unit journals, skip decisions and disk persistence are the real code.
"""
import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from experiments.paper_results_v1 import real_eval, real_backends, attack_eval
from experiments.paper_results_v1.resume_state import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/paper_results_v1"
CASE = "confirm_01"


class Array:
    shape = (5, 320, 512, 3)
    dtype = "float32"
    def copy(self): return Array()
    def numpy(self): return self
    def to(self, **_): return self


def fixture(root, phase):
    generation = phase in ("generate", "decode", "framewise", "codec")
    config = json.loads((PACKAGE / ("real_eval.adopted.json" if generation else "confirmation_v1.adopted.json")).read_text())
    store = (real_eval.RunStore(root / "run", config, create=True) if generation else
             attack_eval.AttackRunStore(root / "run", config, create=True, temp_root=root / "content"))
    for row in store.data["artifacts"]:
        if row["artifact_id"].startswith(CASE + "/"):
            path = root / "inputs" / (row["artifact_id"].replace("/", "_") + ".stub")
            atomic_bytes(path, b"durable-array-stub")
            row.update(status="AVAILABLE", path=str(path), shape=list(Array.shape), bytes=path.stat().st_size)
    if generation:
        endings = {"generate": ["OFF_NATIVE/TERMINAL", "PAYLOAD_NATIVE/TERMINAL"],
                   "decode": ["OFF_NATIVE/PRE", "PAYLOAD_NATIVE/PRE"],
                   "framewise": ["SHARED_FRAMEWISE/LATENT", "PAYLOAD_FRAMEWISE_RECON/PRE", "PAYLOAD_FRAMEWISE_M05/PRE"],
                   "codec": ["OFF_NATIVE/POST", "PAYLOAD_NATIVE/POST"]}[phase]
        for ending in endings: real_eval._artifact(store, CASE + "/" + ending)["status"] = "PLANNED"
    elif phase == "baseline-codec":
        for method in attack_eval.BASELINES: store.artifact(f"{CASE}/{method}/NATIVE_POST")["status"] = "PLANNED"
    elif phase == "attack-media":
        for spec in config["attacks"][1:3]: store.artifact(f"{CASE}/OFF_NATIVE/{spec['attack_id']}/RECEIVED")["status"] = "PLANNED"
    elif phase == "receiver-clock":
        for row in store.data["receiver_rows"]:
            if row["case_id"] == CASE:
                row["status"] = "UNSUPPORTED"
                if row["arm"] == "OFF_NATIVE" and row["attack_id"] == "full" and row["mode"] == "BLIND_PATH":
                    row["status"] = "PLANNED"
    elif phase.startswith("baseline-embed"):
        store.artifact(f"{CASE}/videoseal/NATIVE_PRE")["status"] = "PLANNED"
    elif phase.startswith("baseline-extract"):
        for row in store.data["baseline_rows"]:
            if row["case_id"] == CASE and row["method"] == "videoseal":
                row["status"] = "PLANNED" if row["attack_id"] in [x["attack_id"] for x in config["attacks"][:2]] else "EVALUATED"
    elif phase == "quality":
        for row in store.data["quality_rows"]:
            if row["case_id"] == CASE: row["status"] = "EVALUATED"
        # Include the preview codec boundary, after all three metrics are saved.
        target = next(row for row in store.data["quality_rows"] if row["case_id"] == CASE and
                      row["candidate"] == "PAYLOAD_FRAMEWISE_M05" and row["reference"] == "OFF_NATIVE")
        for row in store.data["quality_rows"]:
            if row["case_id"] == CASE: row["side_by_side"] = {"status":"SAVED"}
        target.update(status="PLANNED", side_by_side={})
    store.save()
    (root / "config.json").write_text(json.dumps(config))
    return store


def worker(root, phase, fail):
    root = Path(root); config = json.loads((root / "config.json").read_text())
    generation = phase in ("generate", "decode", "framewise", "codec")
    store = real_eval.RunStore(root / "run", config) if generation else attack_eval.AttackRunStore(root / "run", config, temp_root=root / "content")
    calls = []
    def event(name):
        calls.append(name)
        with (root / "calls.txt").open("a") as stream:
            stream.write(name + "\n"); stream.flush(); os.fsync(stream.fileno())
        if fail and ((phase == "quality" and name == "preview") or (phase != "quality" and len(calls) == 2)):
            os._exit(23)
    def save_array(files, artifact_id, value, name):
        path = files.root / (name + ".stub")
        atomic_bytes(path, b"committed-array")
        files.store.artifact(artifact_id, status="AVAILABLE", path=str(path), shape=list(value.shape), bytes=path.stat().st_size)
    def load_array(row):
        Path(row["path"]).read_bytes()
        return Array()
    def trajectory(*args, **kwargs):
        arm = "OFF" if not calls else "PAYLOAD"
        event(arm)
        return Array(), {}
    decode_count = 0
    def decode(*args, **kwargs):
        nonlocal decode_count
        decode_count += 1; event("decode")
        return Array()
    class Framewise:
        def encode(self, _):
            # Shared writer encode does not consume the interruption counter.
            with (root / "encodes.txt").open("a") as stream: stream.write("encode\n")
            return Array()
        def decode(self, value): return decode(value)
        def write(self, value, key): return value, {}
        def close(self): pass
    runtime = types.ModuleType("runtime.wan.video_trajectory_conditional_joint_v1")
    runtime.execution_device_dtype = lambda: ("cpu", "float32")
    runtime.generation = types.SimpleNamespace(
        prepare_generation=lambda *a, **k:(types.SimpleNamespace(scheduler={}), Array(), None, None, "float32"),
        load_frozen_vae=lambda *a, **k:types.SimpleNamespace(parameters=lambda:iter([types.SimpleNamespace(device="cpu")])) )
    runtime.vae = types.SimpleNamespace(decode_normalized_latent=decode, quantize_rgb8_no_codec=lambda v:v)
    runtime.run_trajectory = trajectory
    runtime.release = lambda:None
    fake_torch = types.SimpleNamespace(save=lambda v,p:atomic_bytes(p,b"terminal-stub"),
                                       load=lambda p,**k:(Path(p).read_bytes(), Array())[1], float32="float32")
    fake_np = types.SimpleNamespace(load=lambda p,**k:(Path(p).read_bytes(),Array())[1],
                                    savez_compressed=lambda p,**k:atomic_bytes(p,b"score-stub"))
    def roundtrip(value, output, *, media_output):
        event("codec")
        rgb, mp4 = Path(output)/"received.rgb8", Path(media_output)/"source.mp4"
        atomic_bytes(rgb,b"received"); atomic_bytes(mp4,b"publication")
        return Array(), {"received_rgb":{"path":str(rgb),"shape":list(Array.shape),"bytes":8},
                          "mp4":{"path":str(mp4)}, "edited_rgb":{"path":str(rgb)}}
    def fixed_roundtrip(source, digest, mp4, rgb, *, count, event):
        # Same real caller protocol as the fixed publication codec.
        roundtrip(Array(), Path(rgb).parent, media_output=Path(mp4).parent)
        event("mp4", {"path":str(mp4)})
        event("rgb24", {"path":str(rgb), "shape":list(Array.shape), "bytes":8,"sha256":None})
    variable = types.ModuleType("runtime.wan.variable_rgb_media"); variable.roundtrip = roundtrip
    fixed = types.ModuleType("runtime.wan.fixed_rgb_media"); fixed.mp4_roundtrip = fixed_roundtrip
    clock_method = types.ModuleType("main.tube_state.video_trajectory_temporal_edit_receiver_v1")
    clock_method.PUBLIC = {}
    def score(value,key,public):
        event("clock-K0" if key==config["keys"]["K0"] else "clock-K1")
        return {"signed_projection":[], "rho":[], "received_frames":5,"source_frames":5}
    clock_method.score_framewise = score
    selector = types.ModuleType("main.tube_state.receiver_controls_v1")
    selector.receive_scores = lambda *a:{"estimate":{"status":"SUPPORTED"}, "operation":{"status":"SUPPORTED"}}
    class Adapter:
        def embed(self, value, bits):
            event("embed")
            return value, {}
        def extract(self, value):
            event("extract")
            return {"stub_output":True}
    def metric(name):
        def call(*a): event(name); return 1.0
        return call
    def preview(reference,candidate,output):
        event("preview"); atomic_bytes(output,b"preview")
        return {"status":"SAVED","path":str(output)}
    def save_pre(path,value):
        atomic_bytes(path,b"embedded")
        return {"path":str(path),"shape":list(Array.shape),"bytes":8}
    modules = {runtime.__name__:runtime, "torch":fake_torch,"numpy":fake_np, variable.__name__:variable,
               fixed.__name__:fixed,clock_method.__name__:clock_method,selector.__name__:selector}
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules,modules))
        for obj,name,replacement in [
            (real_eval,"run_off_trajectory",trajectory), (real_eval,"load_local_framewise_backend",lambda c:Framewise()),
            (real_backends,"load_local_framewise_backend",lambda c:Framewise()),
            (real_eval.ArtifactFiles,"save_rgb8",save_array), (real_eval.ArtifactFiles,"save_numpy",save_array),
            (real_eval.ArtifactFiles,"load_rgb8",staticmethod(load_array)),
            (attack_eval,"load_rgb",lambda s,r:load_array(r)), (attack_eval,"_release",lambda:None),
            (attack_eval,"_adapter",lambda *a:Adapter()), (attack_eval,"_save_rgb8",save_pre),
            (attack_eval,"apply_edit_rgb8",lambda v,s:(v,attack_eval.recipe_receipt(s))),
            (attack_eval,"_psnr",metric("psnr")), (attack_eval,"_lpips",metric("lpips")),
            (attack_eval,"_reference_flow_fluctuation",metric("flow")), (attack_eval,"_side_by_side_mp4",preview),
        ]: stack.enter_context(mock.patch.object(obj,name,replacement))
        if generation: getattr(real_eval,"phase_"+phase)(store,config,CASE)
        elif phase=="baseline-embed-videoseal":
            attack_eval.phase_baseline_embed(store,config,CASE,"videoseal")
            if fail: os._exit(23) # commit survives abrupt exit before the next phase
        elif phase=="baseline-extract-videoseal": attack_eval.phase_baseline_extract(store,config,CASE,"videoseal")
        else: getattr(attack_eval,"phase_"+phase.replace("-","_"))(store,config,CASE)


class StageResumeTests(unittest.TestCase):
    def test_all_execution_phases_survive_hard_exit_and_content_loss(self):
        for phase in ("generate","decode","framewise","codec","baseline-embed-videoseal",
                      "baseline-codec","attack-media","receiver-clock","baseline-extract-videoseal","quality"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); store=fixture(root,phase)
                first=subprocess.run([sys.executable,"-B","-m",__name__,"worker",str(root),phase,"fail"],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(first.returncode,23,first.stderr)
                (root/"content").mkdir(exist_ok=True); shutil.rmtree(root/"content")
                second=subprocess.run([sys.executable,"-B","-m",__name__,"worker",str(root),phase,"resume"],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(second.returncode,0,second.stderr)
                data=json.loads(store.path.read_text())
                self.assertEqual(data["phases"][phase]["cases"][CASE]["status"],"COMPLETE")
                calls=(root/"calls.txt").read_text().splitlines()
                expected=1 if phase=="baseline-embed-videoseal" else 5 if phase=="quality" else 3
                self.assertEqual(len(calls),expected,calls)
                if phase=="framewise": self.assertEqual((root/"encodes.txt").read_text().splitlines(),["encode"])
                if phase=="receiver-clock": self.assertEqual(calls.count("clock-K0"),1)
                if phase=="quality":
                    self.assertEqual([calls.count(metric) for metric in ("psnr","lpips","flow")],[1,1,1])
                if phase=="baseline-embed-videoseal":
                    row=next(r for r in data["artifacts"] if r["artifact_id"]==CASE+"/videoseal/NATIVE_PRE")
                    self.assertTrue(Path(row["path"]).is_file()); self.assertIn("pre_media",row["path"])
                # Re-running a COMPLETE phase must perform zero heavyweight calls.
                third=subprocess.run([sys.executable,"-B","-m",__name__,"worker",str(root),phase,"resume"],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(third.returncode,0,third.stderr)
                self.assertEqual((root/"calls.txt").read_text().splitlines(),calls)


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="worker": worker(sys.argv[2],sys.argv[3],sys.argv[4]=="fail")
    else: unittest.main()
