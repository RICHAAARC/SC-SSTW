"""User-run fixed PAYLOAD_MULTI RGB reconstruction and framewise-sync candidate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from main.tube_state import video_local_fourier_rm_control as payload_method
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as sync_method
from runtime.wan import framewise_autoencoder_kl as framewise
from runtime.wan import quality
from runtime.wan import rgb8_source
from runtime.wan import vae as wan_adapter
from runtime.wan import video_local_fourier_rm_same_raster as media
from runtime.wan import video_trajectory_payload_framewise_sync_v1 as backend

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_v1.json"
MODULE = "experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run"
CONDITIONS = ("P0_ORIGINAL_RGB", "P1_FRAMEWISE_RECON", "P2_FRAMEWISE_SYNC")
VIEWS = ("FULL", "CROP")
KEY_IDS = ("K0", "K1")
QUALITY_SPACES = ("PRECODEC", "FULL_MP4", "CROP_MP4")
QUALITY_PAIRS = (
    ("P1_FRAMEWISE_RECON", "P0_ORIGINAL_RGB"),
    ("P2_FRAMEWISE_SYNC", "P0_ORIGINAL_RGB"),
    ("P2_FRAMEWISE_SYNC", "P1_FRAMEWISE_RECON"),
)
FIXED = {
    "source_cases": 1,
    "conditions": 3,
    "observations": 6,
    "keys": 2,
    "sync_readouts": 12,
    "sync_candidate_scores": 36,
    "sync_candidate_tubelet_rows": 1626,
    "payload_reads": 12,
    "sync_posthoc": 12,
    "payload_posthoc": 12,
    "quality": 9,
    "rasters": 3,
    "mp4_save": 3,
    "mp4_read": 3,
}


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
    value = "TRAJECTORY_PAYLOAD_FRAMEWISE_SYNC_V1\x00" + condition + "\x00" + view
    return "obs_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(cfg["conditions"]) != CONDITIONS:
        raise ValueError("fixed condition roster changed")
    if cfg["fixed_denominator"] != FIXED:
        raise ValueError("fixed denominator changed")
    if cfg["source"]["shape"] != [181, 320, 512, 3]:
        raise ValueError("fixed source geometry changed")
    if cfg["views"]["CROP"]["truth_start_posthoc_only"] != 1:
        raise ValueError("fixed nonzero crop changed")
    if cfg["views"]["CROP"]["truth_stop_posthoc_only"] != 178:
        raise ValueError("fixed crop stop changed")
    if cfg["framewise_vae"]["id"] != framewise.MODEL_ID:
        raise ValueError("framewise model changed")
    if cfg["framewise_vae"]["revision"] != framewise.MODEL_REVISION:
        raise ValueError("framewise revision changed")
    if cfg["framewise_vae"]["scaled_latent_factor"] != framewise.SCALED_LATENT_FACTOR:
        raise ValueError("scaled latent factor changed")
    if cfg["receiver"]["R"] != 44:
        raise ValueError("original payload R changed")
    public = sync_method.public_receipt()
    if tuple(public["received_frame_counts"]) != (181, 177):
        raise ValueError("public received lengths changed")
    missing = [path for path in cfg["source_files"] if not (ROOT / path).is_file()]
    if missing:
        raise ValueError(f"source file roster contains missing paths: {missing}")
    return cfg


class Store:
    def __init__(self, output: str | Path, *, create: bool = False):
        self.output = Path(output)
        self.path = self.output / "result.json"
        if not create:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return
        cfg = load_config()
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
            "config_sha256": sha256_file(CONFIG),
            "fixed_denominator": FIXED,
            "public_protocol": sync_method.public_receipt(),
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
            "workers": {},
            "failures": [],
            "writer": {"status": "PENDING"},
            "environment": environment_receipt(),
            "actual_generation_calls": False,
            "science_status": cfg["science_status"],
            "evidence_ceiling": cfg["evidence_ceiling"],
        }
        for condition in CONDITIONS:
            folder = self.output / condition
            self.data["conditions"][condition] = {"status": "PENDING"}
            self.data["rasters"][condition] = {
                "status": "PENDING",
                "path": str(folder / "source.rgb8"),
            }
            self.data["transport"][condition] = {"status": "PENDING", "events": {}}
            for view in VIEWS:
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
                    }
                    self.data["payload_reads"][sid] = {
                        "status": "PENDING",
                        "observation_id": oid,
                        "decoded_bits": None,
                        "truth_used": False,
                    }
                    self.data["sync_posthoc"][sid] = {"status": "PENDING"}
                    self.data["payload_posthoc"][sid] = {"status": "PENDING"}
        for space in QUALITY_SPACES:
            for candidate, reference in QUALITY_PAIRS:
                qid = space + "/" + candidate + "_vs_" + reference
                self.data["quality"][qid] = {
                    "status": "PENDING",
                    "diagnostic_only": True,
                    "threshold": None,
                }
        self.save()

    def save(self) -> None:
        expected_sizes = {
            "conditions": 3,
            "rasters": 3,
            "transport": 3,
            "observations": 6,
            "sync_reads": 12,
            "payload_reads": 12,
            "sync_posthoc": 12,
            "payload_posthoc": 12,
            "quality": 9,
        }
        for group, size in expected_sizes.items():
            if len(self.data[group]) != size:
                raise ValueError(f"fixed roster changed: {group}")
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


def _record_quality(
    store: Store,
    space: str,
    videos: dict[str, Any],
) -> None:
    for candidate, reference in QUALITY_PAIRS:
        qid = space + "/" + candidate + "_vs_" + reference
        row = store.data["quality"][qid]
        try:
            metrics = store.call(
                "quality_compute",
                lambda c=candidate, r=reference: quality.rgb_quality_metrics(
                    videos[r].float().div(255.0),
                    videos[c].float().div(255.0),
                ),
            )
            row.update(status="MEASURED", source_space=space, **metrics)
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            store.failure("QUALITY/" + qid, exc)
        store.save()


def media_worker(store: Store, cfg: dict[str, Any]) -> None:
    import torch
    from runtime.wan.generation import load_frozen_vae

    store.data["stage"] = "SOURCE_READ"
    store.save()
    source = store.call(
        "source_read",
        lambda: rgb8_source.read_rgb8_source(
            cfg["source"]["path"],
            expected_sha256=cfg["source"]["sha256"],
            shape=tuple(cfg["source"]["shape"]),
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
    p2_numpy, writer_receipt = store.call(
        "writer_sync",
        lambda: sync_method.apply_projection_margin(
            encoded.numpy().copy(), cfg["key"], sync_method.PUBLIC
        ),
    )
    writer_path = store.output / "writer_receipt.json.gz"
    dump_gzip_json(writer_path, writer_receipt)
    store.data["writer"] = {
        key: value for key, value in writer_receipt.items() if key != "rows"
    }
    store.data["writer"].update(
        status="SAVED",
        path=str(writer_path),
        sha256=sha256_file(writer_path),
        row_count=len(writer_receipt["rows"]),
    )
    p1_rgb = store.call(
        "framewise_writer_decode",
        lambda: framewise.decode_rgb_frames(
            frame_vae, p1_latent, batch_frames=cfg["framewise_vae"]["batch_frames"]
        ),
    )
    p2_rgb = store.call(
        "framewise_writer_decode",
        lambda: framewise.decode_rgb_frames(
            frame_vae,
            torch.from_numpy(p2_numpy.copy()),
            batch_frames=cfg["framewise_vae"]["batch_frames"],
        ),
    )
    raster_tensors = {
        "P0_ORIGINAL_RGB": source,
        "P1_FRAMEWISE_RECON": wan_adapter.quantize_rgb8_no_codec(p1_rgb),
        "P2_FRAMEWISE_SYNC": wan_adapter.quantize_rgb8_no_codec(p2_rgb),
    }
    if not torch.equal(source, raster_tensors["P0_ORIGINAL_RGB"]):
        raise ValueError("P0 source changed")
    del encoded, p1_latent, p2_numpy, p1_rgb, p2_rgb
    full_received: dict[str, Any] = {}
    crop_received: dict[str, Any] = {}
    for condition in CONDITIONS:
        store.data["stage"] = "MEDIA_" + condition
        store.save()
        raster_row = store.call(
            "raster_save",
            lambda c=condition: media.save_raster(
                raster_tensors[c], store.data["rasters"][c]["path"]
            ),
        )
        store.data["rasters"][condition].update(raster_row)
        store.data["conditions"][condition].update(status="RASTER_READY")
        store.save()
        condition_dir = store.output / condition
        received = media.mp4_roundtrip(
            raster_row["path"],
            raster_row["sha256"],
            condition_dir / "source.mp4",
            condition_dir / "received.rgb8",
            count=store.count,
            event=lambda stage, row, c=condition: store.transport_event(c, stage, row),
        )
        full_received[condition] = received
        crop, crop_receipt = rgb8_source.crop_received_rgb(
            received, source_start=1, source_stop=178
        )
        crop_received[condition] = crop
        crop_row = _save_crop(condition_dir / "received.crop1_178.rgb8", crop, crop_receipt)
        for view, received_row in (
            (
                "FULL",
                {
                    "received_path": store.data["transport"][condition]["events"]["rgb24"]["path"],
                    "sha256": store.data["transport"][condition]["events"]["rgb24"]["sha256"],
                    "shape": list(received.shape),
                    "frames": 181,
                    "geometry_only": True,
                },
            ),
            (
                "CROP",
                {
                    "received_path": crop_row["path"],
                    "sha256": crop_row["sha256"],
                    "shape": list(crop.shape),
                    "frames": 177,
                    "geometry_only": False,
                },
            ),
        ):
            oid = observation_id(condition, view)
            store.data["observations"][oid].update(status="READY", **received_row)
        store.data["conditions"][condition].update(status="MEDIA_READY")
        store.save()
    _record_quality(store, "PRECODEC", raster_tensors)
    _record_quality(store, "FULL_MP4", full_received)
    _record_quality(store, "CROP_MP4", crop_received)
    del raster_tensors, full_received, crop_received
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
                    public=sync_method.PUBLIC,
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
                        latent, k, sync_method.PUBLIC
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
                    lambda k=key: payload_method.payload_read(
                        normalized, k, cfg["receiver"]["R"]
                    ),
                )
                row.update(payload)
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                store.failure(sid + "/PAYLOAD", exc)
            store.save()
        del received, normalized
    del wan_vae
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
    if store.data["writer"]["status"] in ("PENDING", "RUNNING"):
        store.data["writer"].update(status="NOT_COMPLETED", error=reason)
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
        true_offset = 0 if view == "FULL" else 1
        scores = {
            int(row["source_offset"]): row["score"]
            for row in readout["candidate_rows"]
            if row["score"] is not None
        }
        true_score = scores.get(true_offset)
        rank = None
        if true_score is not None:
            rank = 1 + sum(
                value > true_score + sync_method.PUBLIC.tie_atol
                for offset, value in scores.items()
                if offset != true_offset
            )
        output.update(
            status="EVALUATED_TRUTH" if key_id == "K0" else "EVALUATED_WRONG_KEY_CONTROL",
            condition=condition,
            view=view,
            key_role="REGISTERED" if key_id == "K0" else "WRONG_KEY",
            true_offset=true_offset,
            true_score=true_score,
            true_rank=rank,
            truth_in_top=true_offset in readout["summary"]["top_offsets"],
            unique_truth=readout["summary"]["top_offsets"] == [true_offset],
            full_singleton_geometry_only=view == "FULL",
            expected_sync_condition=condition == "P2_FRAMEWISE_SYNC",
            sync_accepted=False,
        )
    truth_bits = payload_method.message_bits(cfg["message"])
    for sid, blind_row in snapshot["payload_reads"].items():
        output = store.data["payload_posthoc"][sid]
        oid, key_id = sid.split("/")
        truth = store.data["observations"][oid]
        bits = blind_row.get("decoded_bits")
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
            payload_accepted=False,
        )
    if sha256_file(blind_path) != sealed:
        raise ValueError("truth join changed sealed blind readouts")
    store.data["blind_receiver_sha256"] = sealed
    store.save()


def finish(store: Store, cfg: dict[str, Any]) -> bool:
    expected_counts = {
        key: value
        for key, value in FIXED.items()
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
    done = (
        not store.data["failures"]
        and store.data["counts"] == expected_counts
        and store.data["call_integrity"]["status"] == "MATCH"
        and store.data["workers"].get("media", {}).get("status") == "COMPLETE"
        and store.data["actual_generation_calls"] is False
    )
    store.data.update(
        status="EXECUTION_COMPLETE" if done else "INCOMPLETE",
        stage="FINISHED",
        science_status=cfg["science_status"],
    )
    store.save()
    print(json.dumps({"status": store.data["status"], "counts": store.data["counts"]}))
    return done


def run_worker_phase(store: Store) -> tuple[Store, bool]:
    command = [
        sys.executable,
        "-u",
        "-m",
        MODULE,
        "--output",
        str(store.output),
        "--worker",
        "media",
    ]
    start = time.perf_counter()
    child = None
    returncode = None
    error = None
    cleanup_error = None
    interrupted = False
    try:
        with (store.output / "media.log").open("w", encoding="utf-8") as log:
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
    store = Store(store.output)
    ok = returncode == 0 and error is None and cleanup_error is None
    store.data["workers"]["media"] = {
        "status": "COMPLETE" if ok else "FAILED",
        "command": command,
        "returncode": returncode,
        "error": error,
        "cleanup_error": cleanup_error,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if not ok:
        store.failure(
            "WORKER_MEDIA",
            RuntimeError(error or cleanup_error or f"child exit {returncode}"),
        )
    settle(store, error or "media worker did not complete")
    return store, interrupted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("media",))
    args = parser.parse_args()
    cfg = load_config()
    if args.worker:
        store = Store(args.output)
        try:
            media_worker(store, cfg)
        except Exception as exc:
            store.failure(store.data["stage"], exc)
            settle(store, str(exc))
            raise
        return
    store = Store(args.output, create=True)
    store, interrupted = run_worker_phase(store)
    evaluate(store, cfg)
    done = finish(store, cfg)
    if interrupted:
        raise SystemExit(130)
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
