"""Independent fixed G/T user-run orchestration; historical M05 entrypoint unchanged."""
from __future__ import annotations
import argparse,copy,gc,gzip,hashlib,importlib.metadata,json,math,os,subprocess,sys,time
from pathlib import Path
from typing import Any
import numpy as np
from main.tube_state import video_local_fourier_rm_control as payload_method
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as sync_method
from main.tube_state import video_trajectory_payload_gt_v1 as gt_method
from runtime.wan import framewise_autoencoder_kl as framewise
from runtime.wan import quality,rgb8_source
from runtime.wan import vae as wan_adapter
from runtime.wan import video_local_fourier_rm_same_raster as media
from runtime.wan import video_trajectory_payload_framewise_sync_v1 as backend
from runtime.wan import video_trajectory_payload_gt_v1 as gt_backend
ROOT=Path(__file__).resolve().parents[2]
KEY_IDS=("K0","K1")
IN_PROGRESS_EVIDENCE_CEILING="Fixed user-run record in progress; missing and failed slots retained; no science PASS."
def execution_evidence_ceiling(done):
    return ("Fixed G/T execution complete. " if done else "Fixed G/T execution incomplete. ")+(
        "One prespecified source per experiment; full MP4 then fixed views. FULL is geometry only. "
        "Repeated payload recovery is not synchronization evidence. Scores/gaps and raw/normalized "
        "bit margins are descriptive. Residual temporal metrics are distinct from each video self-difference ratio. "
        "No calibrated threshold, FPR, generalization, perception or scientific PASS.")
def config_path(profile):
    if profile not in ("G","T"):raise ValueError("fixed G/T entrypoint required")
    return ROOT/("experiments/wan_state_clock/configs/video_trajectory_payload_"+profile.lower()+"_v1.json")
def load_config(profile):
    cfg=json.loads(config_path(profile).read_text())
    if cfg["profile"]!=profile:raise ValueError("fixed entrypoint/config mismatch")
    public=gt_method.public_protocol(profile)
    for view in cfg["views"].values():
        if list(sync_method.candidate_offsets(view["length"],public))!=view["public_offsets"]:
            raise ValueError("fixed public candidate domain mismatch")
    missing=[path for path in cfg["source_files"] if not (ROOT/path).is_file()]
    if missing:raise ValueError(f"missing source files: {missing}")
    return cfg


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dump_gzip_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    os.replace(temporary, path)


def read_gzip_json(path: str | Path) -> Any:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def environment_receipt() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "diffusers", "transformers", "numpy", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def observation_id(condition: str, view: str) -> str:
    value = "TRAJECTORY_PAYLOAD_GT_V1\x00" + condition + "\x00" + view
    return "obs_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _key_roster(cfg: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return (("K0", cfg["key"]), ("K1", cfg["wrong_key"]))


def _save_crop(path: Path, crop: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(crop.detach().cpu().contiguous().numpy().tobytes())
    os.replace(temporary, path)
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("persisted crop identity mismatch")
    return {**receipt, "path": str(path)}


def _load_observation_rgb(row: dict[str, Any]) -> Any:
    import torch

    shape = tuple(row["shape"])
    raw = Path(row["received_path"]).read_bytes()
    expected = math.prod(shape)
    if len(raw) != expected or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("persisted observation identity mismatch")
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy())



class Store:
    def __init__(self, output: str | Path, *, profile: str, create: bool = False):
        self.profile=profile
        self.cfg=load_config(profile)
        self.public=gt_method.public_protocol(profile)
        self.conditions=tuple(self.cfg["conditions"])
        self.sync_conditions=tuple(self.cfg["writer"]["conditions"])
        self.views=tuple(self.cfg["views"])
        self.quality_spaces=tuple(self.cfg["quality_spaces"])
        self.quality_pairs=tuple(tuple(pair) for pair in self.cfg["quality_pairs"])
        self.fixed=self.cfg["fixed_denominator"]
        self.module="experiments.wan_state_clock.video_trajectory_payload_"+profile.lower()+"_v1_run"
        self.output = Path(output)
        self.path = self.output / "result.json"
        if not create:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return
        cfg = self.cfg
        self.output.mkdir(parents=True, exist_ok=False)
        self.data = {
            "status": "RUNNING",
            "stage": "INITIALIZE",
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_worktree_status": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).splitlines(),
            "source_files": {
                path: sha256_file(ROOT / path) for path in cfg["source_files"]
            },
            "config_sha256": sha256_file(config_path(profile)),
            "profile": profile,
            "fixed_denominator": self.fixed,
            "public_protocol": sync_method.public_receipt(self.public),
            "source_protocol": cfg["source"],
            "framewise_vae_protocol": cfg["framewise_vae"],
            "original_payload_model": cfg["model"],
            "conditions": {},
            "rasters": {},
            "transport": {},
            "observations": {},
            "sync_reads": {},
            "payload_reads": {},
            "sync_posthoc": {},
            "payload_posthoc": {},
            "quality": {},
            "calls": {},
            "source_calls": {},
            "source_preparation": {
                "status": "PENDING" if profile=="G" else "NOT_REQUIRED",
                "arm": "PAYLOAD_MULTI" if profile=="G" else None,
                "steps": [{"index":i,"status":"PENDING"} for i in range(50)] if profile=="G" else [],
                "rgb_path": str(self.output/"prepared_source"/"source.rgb8"),
                "protocol": cfg.get("generation"),
            },
            "public_candidate_rosters": {
                str(spec["length"]): [
                    {"source_offset": offset, "supports": sync_method.tubelet_support_rows(spec["length"],offset,self.public)}
                    for offset in sync_method.candidate_offsets(spec["length"],self.public)
                ] for spec in cfg["views"].values()
            },
            "workers": {},
            "failures": [],
            "writers": {
                condition: {"status": "PENDING"}
                for condition in self.sync_conditions
            },
            "environment": environment_receipt(),
            "actual_generation_calls": False,
            "science_status": cfg["science_status"],
            "evidence_provenance": {
                "candidate_pre_run_template": cfg["evidence_ceiling"],
            },
            "evidence_ceiling": IN_PROGRESS_EVIDENCE_CEILING,
        }
        candidate_rosters=self.data.pop("public_candidate_rosters")
        roster_path=self.output/"public_candidate_rosters.json"
        dump_json(roster_path,candidate_rosters)
        self.data["public_candidate_rosters"]={"path":str(roster_path),"sha256":sha256_file(roster_path)}
        for condition in self.conditions:
            folder = self.output / condition
            self.data["conditions"][condition] = {"status": "PENDING"}
            self.data["rasters"][condition] = {
                "status": "PENDING",
                "path": str(folder / "source.rgb8"),
            }
            self.data["transport"][condition] = {"status": "PENDING", "events": {}}
            for view in self.views:
                oid = observation_id(condition, view)
                self.data["observations"][oid] = {
                    "status": "PENDING",
                    "condition_truth_join_only": condition,
                    "view_truth_join_only": view,
                    "received_path": None,
                }
                for key_id in KEY_IDS:
                    sid = oid + "/" + key_id
                    blind = self.output / "blind" / oid
                    self.data["sync_reads"][sid] = {
                        "status": "PENDING",
                        "path": str(blind / (key_id + ".sync.json.gz")),
                        "observation_id": oid,
                        "public_candidate_roster_id": str(cfg["views"][view]["length"]),
                        "planned_candidate_scores": len(cfg["views"][view]["public_offsets"]),
                        "planned_local_rows": sum(len(sync_method.tubelet_support_rows(cfg["views"][view]["length"],o,self.public)) for o in cfg["views"][view]["public_offsets"]),
                    }
                    self.data["payload_reads"][sid] = {
                        "status": "PENDING",
                        "observation_id": oid,
                        "decoded_bits": None,
                        "truth_used": False,
                    }
                    self.data["sync_posthoc"][sid] = {"status": "PENDING"}
                    self.data["payload_posthoc"][sid] = {"status": "PENDING"}
        for space in self.quality_spaces:
            for candidate, reference in self.quality_pairs:
                qid = space + "/" + candidate + "_vs_" + reference
                self.data["quality"][qid] = {
                    "status": "PENDING",
                    "diagnostic_only": True,
                    "threshold": None,
                }
        self.save()

    def save(self) -> None:
        expected_sizes = {
            "conditions":4,"rasters":4,"transport":4,
            "observations":self.fixed["observations"],
            "sync_reads":self.fixed["sync_readouts"],
            "payload_reads":self.fixed["payload_reads"],
            "sync_posthoc":self.fixed["sync_posthoc"],
            "payload_posthoc":self.fixed["payload_posthoc"],
            "quality":self.fixed["quality"],
        }
        for group, size in expected_sizes.items():
            if len(self.data[group]) != size:
                raise ValueError(f"fixed roster changed: {group}")
        if tuple(self.data["writers"]) != self.sync_conditions:
            raise ValueError("fixed writer slot roster changed")
        self.data["counts"] = {
            "rasters": sum(row["status"] == "SAVED" for row in self.data["rasters"].values()),
            "mp4_save": sum(
                row.get("events", {}).get("mp4", {}).get("status") == "SAVED"
                for row in self.data["transport"].values()
            ),
            "mp4_read": sum(row["status"] == "COMPLETE" for row in self.data["transport"].values()),
            "observations": sum(
                row["status"] == "READY" for row in self.data["observations"].values()
            ),
            "sync_readouts": sum(
                row["status"] == "SAVED" for row in self.data["sync_reads"].values()
            ),
            "sync_candidate_scores": sum(
                int(row.get("candidate_scores", 0)) for row in self.data["sync_reads"].values()
            ),
            "sync_candidate_tubelet_rows": sum(
                int(row.get("candidate_tubelet_rows", 0))
                for row in self.data["sync_reads"].values()
            ),
            "payload_reads": sum(
                row["status"] == "READ" for row in self.data["payload_reads"].values()
            ),
            "sync_posthoc": sum(
                row["status"].startswith("EVALUATED")
                for row in self.data["sync_posthoc"].values()
            ),
            "payload_posthoc": sum(
                row["status"].startswith("EVALUATED")
                for row in self.data["payload_posthoc"].values()
            ),
            "quality": sum(row["status"] == "MEASURED" for row in self.data["quality"].values()),
        }
        dump_json(self.path, self.data)

    def count(self, name: str, completed: bool) -> None:
        row = self.data["calls"].setdefault(name, {"attempted": 0, "completed": 0})
        row["completed" if completed else "attempted"] += 1
        if name == "generation":
            self.data["actual_generation_calls"] = True
        self.save()

    def call(self, name: str, operation: Any) -> Any:
        self.count(name, False)
        value = operation()
        self.count(name, True)
        return value

    def failure(self, where: str, exc: BaseException) -> None:
        self.data["failures"].append(
            {"stage": where, "error": f"{type(exc).__name__}: {exc}"}
        )
        self.save()

    def transport_event(self, condition: str, stage: str, row: dict[str, Any]) -> None:
        self.data["transport"][condition]["events"][stage] = row
        if stage == "rgb24" and row.get("status") == "SAVED":
            self.data["transport"][condition]["status"] = "COMPLETE"
        elif row.get("status") == "FAILED":
            self.data["transport"][condition]["status"] = "FAILED"
        self.save()

    def blind_snapshot(self) -> Path:
        path = self.output / "blind_receiver_readouts.json"
        dump_json(
            path,
            {
                "sync_reads": self.data["sync_reads"],
                "payload_reads": self.data["payload_reads"],
                "truth_inputs": False,
                "receiver_inputs": "received video, key, fixed public protocol",
            },
        )
        return path



def source_count(store,name,completed):
    row=store.data["source_calls"].setdefault(name,{"attempted":0,"completed":0})
    row["completed" if completed else "attempted"]+=1
    if name=="trajectory":store.data["actual_generation_calls"]=True
    store.save()

def source_call(store,name,operation):
    source_count(store,name,False)
    value=operation()
    source_count(store,name,True)
    return value

def source_worker(store,cfg):
    """One PAYLOAD_MULTI trajectory, free transformer, one native decode, save identity."""
    import torch
    from runtime.wan.generation import prepare_generation,load_frozen_vae
    from runtime.wan import video_local_fourier_rm_old8_two_state_v1 as old8
    from runtime.wan import trajectory
    if store.profile!="G":raise ValueError("T has no source generation")
    row=store.data["source_preparation"]
    row.update(status="RUNNING")
    store.data["stage"]="SOURCE_PREPARATION"
    store.save()
    device,dtype=old8.execution_device_dtype()
    pipe,initial,prompt,negative,input_dtype=source_call(store,"generation_load",
        lambda:prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype))
    initial_hash=trajectory.fingerprint(initial)
    def record(step):
        row["steps"][step["index"]]=dict(step,status="COMPLETE")
        store.save()
    terminal,receipt=source_call(store,"trajectory",lambda:old8.run_trajectory(
        pipe,initial,pipe.scheduler,prompt,negative,input_dtype,"PAYLOAD_MULTI",
        cfg["key"],payload_method.message_bits(cfg["message"]),
        lambda name,done:source_count(store,name,done),record,diagnostic=None))
    if trajectory.fingerprint(initial)!=initial_hash:raise ValueError("initial source state changed")
    row.update(trajectory_receipt=receipt,initial_sha256=initial_hash)
    store.save()
    del pipe,initial,prompt,negative
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    native_vae=source_call(store,"wan_vae_load",lambda:load_frozen_vae(cfg,device=device))
    rgb=source_call(store,"wan_decode",lambda:wan_adapter.decode_normalized_latent(
        native_vae,terminal.to(next(native_vae.parameters()).device)))
    raster=source_call(store,"raster_save",lambda:media.save_raster(
        wan_adapter.quantize_rgb8_no_codec(rgb),row["rgb_path"]))
    row.update(status="SAVED",rgb_receipt=raster)
    store.data["source_protocol"]={**cfg["source"],"path":raster["path"],"sha256":raster["sha256"],"bytes":raster["bytes"]}
    store.save()  # identity is frozen before any comparison or receiver output
    del terminal,rgb,native_vae
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def _record_quality(store,space,videos):
    start=store.cfg["quality_spaces"][space]["source_start"]
    for candidate,reference in store.quality_pairs:
        qid=space+"/"+candidate+"_vs_"+reference
        row=store.data["quality"][qid]
        try:
            metrics=store.call("quality_compute",lambda c=candidate,r=reference:quality.rgb_quality_metrics(
                videos[r].float().div(255.0),videos[c].float().div(255.0)))
            if reference=="P1_FRAMEWISE_RECON":
                metrics["condition_minus_p1_residual"]=gt_backend.residual_temporal_metrics(
                    videos[reference],videos[candidate],source_start=start)
            row.update(status="MEASURED",source_space=space,**metrics)
        except Exception as exc:
            row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
            store.failure("QUALITY/"+qid,exc)
        store.save()



def media_worker(store: Store, cfg: dict[str, Any]) -> None:
    import torch
    from runtime.wan.generation import load_frozen_vae

    store.data["stage"] = "SOURCE_READ"
    store.save()
    source_protocol=store.data["source_protocol"]
    source = store.call(
        "source_read",
        lambda: rgb8_source.read_rgb8_source(
            source_protocol["path"],
            expected_sha256=source_protocol["sha256"],
            shape=tuple(source_protocol["shape"]),
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store.data["stage"] = "FRAMEWISE_WRITER"
    store.save()
    frame_vae = store.call(
        "framewise_vae_load",
        lambda: framewise.load_frozen_framewise_vae(device=device),
    )
    encoded = store.call(
        "framewise_writer_encode",
        lambda: framewise.encode_rgb_frames(
            frame_vae, source.float().div(255.0), batch_frames=cfg["framewise_vae"]["batch_frames"]
        ),
    )
    p1_latent = encoded.clone()
    writer_source = encoded.numpy().copy()
    writer_source_sha256 = hashlib.sha256(writer_source.tobytes()).hexdigest()
    written_latents: dict[str, np.ndarray] = {}
    for condition in store.sync_conditions:
        written,writer_receipt=store.call("writer_sync",lambda c=condition:gt_method.write_condition(
            writer_source.copy(),cfg["key"],c,store.public))
        if hashlib.sha256(writer_source.tobytes()).hexdigest()!=writer_source_sha256:
            raise ValueError("shared encoded writer source changed")
        writer_path=store.output/("writer_receipt."+condition+".json.gz")
        dump_gzip_json(writer_path,writer_receipt)
        store.data["writers"][condition]={k:v for k,v in writer_receipt.items()
            if k not in ("rows","crop177_partial_rows","construction_summary")}
        store.data["writers"][condition].update(status="SAVED",path=str(writer_path),
            sha256=sha256_file(writer_path),row_count=len(writer_receipt["rows"]),
            input_scaled_latent_sha256=writer_source_sha256,independent_copy=True)
        written_latents[condition]=written
        store.save()
    raster_tensors={"P0_ORIGINAL_RGB":source}
    for condition,latent in [("P1_FRAMEWISE_RECON",p1_latent)]+[
        (c,torch.from_numpy(written_latents[c].copy())) for c in store.sync_conditions]:
        rgb=store.call("framewise_writer_decode",lambda z=latent:framewise.decode_rgb_frames(
            frame_vae,z,batch_frames=cfg["framewise_vae"]["batch_frames"]))
        raster_tensors[condition]=wan_adapter.quantize_rgb8_no_codec(rgb)
    del latent,rgb
    if not torch.equal(source, raster_tensors["P0_ORIGINAL_RGB"]):
        raise ValueError("P0 source changed")
    del encoded,p1_latent,writer_source,written_latents
    received_by_view={view:{} for view in store.views}
    for condition in store.conditions:
        store.data["stage"]="MEDIA_"+condition
        store.save()
        raster_row=store.call("raster_save",lambda c=condition:media.save_raster(
            raster_tensors[c],store.data["rasters"][c]["path"]))
        store.data["rasters"][condition].update(raster_row)
        store.data["conditions"][condition].update(status="RASTER_READY")
        store.save()
        condition_dir=store.output/condition
        received=media.mp4_roundtrip(raster_row["path"],raster_row["sha256"],
            condition_dir/"source.mp4",condition_dir/"received.rgb8",count=store.count,
            event=lambda stage,row,c=condition:store.transport_event(c,stage,row))
        for view,spec in cfg["views"].items():
            if spec["length"]==181:
                clip=received
                received_row=store.data["transport"][condition]["events"]["rgb24"]
            else:
                clip,crop_receipt=rgb8_source.crop_received_rgb(received,
                    source_start=spec["truth_start_posthoc_only"],source_stop=spec["truth_stop_posthoc_only"])
                received_row=_save_crop(condition_dir/("received."+view+".rgb8"),clip,crop_receipt)
            received_by_view[view][condition]=clip
            store.data["observations"][observation_id(condition,view)].update(status="READY",
                received_path=received_row["path"],sha256=received_row["sha256"],
                shape=list(clip.shape),frames=int(clip.shape[0]),geometry_only=spec["length"]==181)
        store.data["conditions"][condition].update(status="MEDIA_READY")
        store.save()
    _record_quality(store,"PRECODEC",raster_tensors)
    for view,videos in received_by_view.items():
        _record_quality(store,view+"_MP4",videos)
    del raster_tensors,received_by_view,received,clip
    store.data["stage"] = "BLIND_SYNC_RECEIVER"
    store.save()
    for oid, observation in store.data["observations"].items():
        received = _load_observation_rgb(observation)
        try:
            latent, encode_receipt = store.call(
                "framewise_receiver_encode",
                lambda r=received: backend.encode_received_video(
                    r,
                    frame_vae,
                    batch_frames=cfg["framewise_vae"]["batch_frames"],
                    public=store.public,
                ),
            )
        except Exception as exc:
            for key_id in KEY_IDS:
                store.data["sync_reads"][oid + "/" + key_id].update(
                    status="FAILED", error=f"{type(exc).__name__}: {exc}"
                )
            store.failure(oid + "/FRAMEWISE_ENCODE", exc)
            continue
        for key_id, key in _key_roster(cfg):
            sid = oid + "/" + key_id
            row = store.data["sync_reads"][sid]
            try:
                readout = store.call(
                    "sync_score",
                    lambda k=key: sync_method.score_received_latent(
                        latent, k, store.public
                    ),
                )
                dump_gzip_json(
                    row["path"],
                    {
                        "observation_id": oid,
                        "encode_receipt": encode_receipt,
                        "readout": readout,
                        "truth_inputs": False,
                    },
                )
                row.update(
                    status="SAVED",
                    sha256=sha256_file(row["path"]),
                    readout_status=readout["status"],
                    summary=readout["summary"],
                    **readout["counts"],
                )
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                store.failure(sid + "/SYNC", exc)
            store.save()
        del received, latent
    del frame_vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    store.data["stage"] = "BLIND_PAYLOAD_RECEIVER"
    store.save()
    wan_vae = store.call(
        "wan_vae_load", lambda: load_frozen_vae(cfg, device=device)
    )
    for oid, observation in store.data["observations"].items():
        received = _load_observation_rgb(observation)
        try:
            normalized = store.call(
                "wan_receiver_encode",
                lambda r=received: wan_adapter.reencode_rgb24_readback(
                    wan_vae, r.float().div(255.0)
                ),
            )
        except Exception as exc:
            for key_id in KEY_IDS:
                store.data["payload_reads"][oid + "/" + key_id].update(
                    status="FAILED", error=f"{type(exc).__name__}: {exc}"
                )
            store.failure(oid + "/WAN_ENCODE", exc)
            continue
        for key_id, key in _key_roster(cfg):
            sid = oid + "/" + key_id
            row = store.data["payload_reads"][sid]
            try:
                payload = store.call(
                    "payload_read",
                    lambda k=key: gt_backend.read_payload(normalized, k, int(received.shape[0])),
                )
                row.update(payload)
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                store.failure(sid + "/PAYLOAD", exc)
            store.save()
        del received, normalized
    del wan_vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    store.data["stage"] = "BLIND_READ_COMPLETE"
    store.save()



def settle(store: Store, reason: str) -> None:
    for group in (
        "conditions",
        "rasters",
        "transport",
        "observations",
        "sync_reads",
        "payload_reads",
        "sync_posthoc",
        "payload_posthoc",
        "quality",
    ):
        for row in store.data[group].values():
            if row["status"] in ("PENDING", "RUNNING"):
                row.update(status="NOT_COMPLETED", error=reason)
    for row in store.data["writers"].values():
        if row["status"] in ("PENDING", "RUNNING"):
            row.update(status="NOT_COMPLETED", error=reason)
    source=store.data["source_preparation"]
    if source["status"] in ("PENDING","RUNNING"):
        source.update(status="NOT_COMPLETED",error=reason)
        for row in source["steps"]:
            if row["status"]=="PENDING":row.update(status="NOT_COMPLETED",error=reason)
    store.save()
    store.blind_snapshot()



def evaluate(store: Store, cfg: dict[str, Any]) -> None:
    blind_path = store.blind_snapshot()
    sealed = sha256_file(blind_path)
    snapshot = json.loads(blind_path.read_text(encoding="utf-8"))
    for sid, blind_row in snapshot["sync_reads"].items():
        output = store.data["sync_posthoc"][sid]
        oid, key_id = sid.split("/")
        truth = store.data["observations"][oid]
        condition = truth["condition_truth_join_only"]
        view = truth["view_truth_join_only"]
        if blind_row["status"] != "SAVED":
            output.update(status="MISSING_READ", error=blind_row.get("error"))
            continue
        readout = read_gzip_json(blind_row["path"])["readout"]
        true_offset = cfg["views"][view]["truth_start_posthoc_only"]
        scores = {
            int(row["source_offset"]): row["score"]
            for row in readout["candidate_rows"]
            if row["score"] is not None
        }
        true_score = scores.get(true_offset)
        rank = None
        if true_score is not None:
            rank = 1 + sum(
                value > true_score + store.public.tie_atol
                for offset, value in scores.items()
                if offset != true_offset
            )
        other_scores=[value for offset,value in scores.items() if offset!=true_offset]
        best_other=max(other_scores) if other_scores else None
        output.pop("error", None)
        output.update(
            best_other_score=best_other,
            true_score_gap=None if true_score is None or best_other is None else true_score-best_other,
            status="EVALUATED_TRUTH" if key_id == "K0" else "EVALUATED_WRONG_KEY_CONTROL",
            condition=condition,
            view=view,
            key_role="REGISTERED" if key_id == "K0" else "WRONG_KEY",
            true_offset=true_offset,
            true_score=true_score,
            true_rank=rank,
            truth_in_top=true_offset in readout["summary"]["top_offsets"],
            unique_truth=readout["summary"]["top_offsets"] == [true_offset],
            full_singleton_geometry_only=cfg["views"][view]["length"] == 181,
            expected_sync_condition=condition in store.sync_conditions,
            sync_accepted=False,
        )
    truth_bits = payload_method.message_bits(cfg["message"])
    for sid, blind_row in snapshot["payload_reads"].items():
        output = store.data["payload_posthoc"][sid]
        oid, key_id = sid.split("/")
        truth = store.data["observations"][oid]
        bits = blind_row.get("decoded_bits")
        if blind_row["status"] == "READ":
            output.pop("error", None)
        else:
            output["error"] = blind_row.get("error")
        output.update(
            status=(
                "EVALUATED_TRUTH"
                if blind_row["status"] == "READ" and key_id == "K0"
                else "EVALUATED_WRONG_KEY_CONTROL"
                if blind_row["status"] == "READ"
                else "MISSING_READ"
            ),
            condition=truth["condition_truth_join_only"],
            view=truth["view_truth_join_only"],
            key_role="REGISTERED" if key_id == "K0" else "WRONG_KEY",
            bit_errors=(
                sum(left != right for left, right in zip(bits, truth_bits))
                if bits is not None
                else None
            ),
            exact_payload=bits == truth_bits if bits is not None else None,
            bit_rows=[
                dict(row,expected=int(truth_bits[row["bit_index"]]),
                     bit_error=int(row["decoded"]!=truth_bits[row["bit_index"]]))
                for row in blind_row.get("bit_rows",[])
            ],
            payload_accepted=False,
        )
    if sha256_file(blind_path) != sealed:
        raise ValueError("truth join changed sealed blind readouts")
    store.data["blind_receiver_sha256"] = sealed
    store.save()



def finish(store: Store, cfg: dict[str, Any]) -> bool:
    expected_counts = {
        key: value
        for key, value in store.fixed.items()
        if key not in ("source_cases", "conditions", "keys")
    }
    call_rows = {}
    for name, expected in cfg["planned_calls"].items():
        actual = store.data["calls"].get(name, {"attempted": 0, "completed": 0})
        call_rows[name] = {
            "expected": expected,
            "actual": actual,
            "match": actual == {"attempted": expected, "completed": expected},
        }
    store.data["call_integrity"] = {
        "status": "MATCH" if all(row["match"] for row in call_rows.values()) else "INCOMPLETE",
        "rows": call_rows,
    }
    source_rows={}
    for name,expected in cfg["source_preparation_planned_calls"].items():
        actual=store.data["source_calls"].get(name,{"attempted":0,"completed":0})
        source_rows[name]={"expected":expected,"actual":actual,"match":actual=={"attempted":expected,"completed":expected}}
    source_ok=all(row["match"] for row in source_rows.values())
    store.data["source_preparation_call_integrity"]={"status":"MATCH" if source_ok else "INCOMPLETE","rows":source_rows}
    done = (
        not store.data["failures"]
        and store.data["counts"] == expected_counts
        and store.data["call_integrity"]["status"] == "MATCH"
        and store.data["workers"].get("media", {}).get("status") == "COMPLETE"
        and source_ok
        and (store.profile=="T" and store.data["actual_generation_calls"] is False
             or store.profile=="G" and store.data["source_preparation"]["status"]=="SAVED"
             and store.data["workers"].get("source",{}).get("status")=="COMPLETE")
    )
    store.data.update(
        status="EXECUTION_COMPLETE" if done else "INCOMPLETE",
        stage="FINISHED",
        science_status=cfg["science_status"],
        evidence_ceiling=execution_evidence_ceiling(done),
    )
    store.save()
    print(json.dumps({"status": store.data["status"], "counts": store.data["counts"]}))
    return done



def run_worker_phase(store: Store, phase: str = "media") -> tuple[Store, bool]:
    command = [
        sys.executable,
        "-u",
        "-m",
        store.module,
        "--output",
        str(store.output),
        "--worker",
        phase,
    ]
    start = time.perf_counter()
    child = None
    returncode = None
    error = None
    cleanup_error = None
    interrupted = False
    try:
        with (store.output / (phase+".log")).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = child.wait()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        interrupted = not isinstance(exc, Exception)
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    returncode = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    returncode = child.wait(timeout=10)
            except BaseException as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
    finally:
        if child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as cleanup:
                cleanup_error = cleanup_error or f"{type(cleanup).__name__}: {cleanup}"
    store = Store(store.output,profile=store.profile)
    ok = returncode == 0 and error is None and cleanup_error is None
    store.data["workers"][phase] = {
        "status": "COMPLETE" if ok else "FAILED",
        "command": command,
        "returncode": returncode,
        "error": error,
        "cleanup_error": cleanup_error,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if not ok:
        failure_reason = error or cleanup_error or f"{phase} worker child exit {returncode}"
        store.failure("WORKER_"+phase.upper(), RuntimeError(failure_reason))
        settle(store, failure_reason)
    else:
        store.save()
    return store, interrupted



def main(profile):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",choices=("source","media"))
    args=parser.parse_args()
    cfg=load_config(profile)
    if args.worker:
        store=Store(args.output,profile=profile)
        try:
            (source_worker if args.worker=="source" else media_worker)(store,cfg)
        except Exception as exc:
            store.failure(store.data["stage"],exc)
            settle(store,str(exc))
            raise
        return
    store=Store(args.output,profile=profile,create=True)
    interrupted=False
    if profile=="G":
        store,interrupted=run_worker_phase(store,"source")
    if not interrupted and (profile=="T" or store.data["source_preparation"]["status"]=="SAVED"):
        store,interrupted=run_worker_phase(store,"media")
    else:
        settle(store,"source preparation did not complete")
    evaluate(store,cfg)
    done=finish(store,cfg)
    if interrupted:raise SystemExit(130)
    if not done:raise SystemExit(1)
