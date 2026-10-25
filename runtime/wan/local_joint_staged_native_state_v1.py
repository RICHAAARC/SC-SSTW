"""Fixed paired saved-native-terminal experiment. No generation or gradients."""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

from main.tube_state import local_joint_staged_native_state_v1 as method
from main.tube_state.grow_video_reference import read_latent_bits, message_bits

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_VAE_CACHE = Path("/content/video-wm-cache/models/Wan2.1-T2V-1.3B-Diffusers")


def execution_artifacts(output):
    """Presence of prior work spends the attempt; setup files alone do not."""
    output = Path(output)
    records = ("attempt.json", "result.json", "checkpoint.json", "ledger.json", "evaluation.json")
    found = [name+suffix for name in records for suffix in ("", ".tmp")
             if (output/(name+suffix)).exists() or (output/(name+suffix)).is_symlink()]
    for folder in ("blind", "features", "media", "quality"):
        root = output/folder
        if root.is_dir():
            found.extend(str(path.relative_to(output)) for path in root.rglob("*")
                         if path.is_file() or path.is_symlink())
    return sorted(found)


def existing_attempt(output, artifacts):
    """Read saved small JSON only; corrupt/missing records never grant a retry."""
    errors = {}
    for name in ("result.json", "checkpoint.json", "result.json.tmp", "checkpoint.json.tmp"):
        path = output/name
        if not path.exists():
            continue
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(prior, dict):
                raise ValueError("saved result is not a JSON object")
        except (OSError, ValueError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        return dict(status="EXISTING_ATTEMPT_NOT_REEXECUTED", saved=prior, saved_record=name,
                    directory=str(output), existing_artifacts=artifacts, read_errors=errors)
    return dict(status="EXISTING_ATTEMPT_WITHOUT_CHECKPOINT", directory=str(output),
                existing_artifacts=artifacts, read_errors=errors, fixed_denominator=fixed_denominator(),
                action="read existing records only; no physical calls repeated")


def select_vae_source(model_id, cache_base=DEFAULT_VAE_CACHE):
    """Locate one available VAE before the sole load; no load fallback/retry."""
    explicit = Path(model_id)
    if explicit.is_dir():
        return dict(location=str(explicit), subfolder="vae" if (explicit/"vae").is_dir() else None,
                    origin="explicit_local", weights_may_download=False)
    if model_id == DEFAULT_MODEL_ID:
        base = Path(cache_base)
        candidates = [base/"available", base]
        if base.is_dir():
            candidates.extend(sorted(path for path in base.iterdir() if path.is_dir()))
        for path in candidates:
            vae = path/"vae"
            if (vae/"config.json").is_file() and (any(vae.glob("*.safetensors")) or any(vae.glob("*.bin"))):
                return dict(location=str(path), subfolder="vae", origin="session_cache", weights_may_download=False)
    return dict(location=model_id, subfolder="vae", origin="model_id", weights_may_download=True)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def units():
    result = {name: kind for name, kind in (("source", "terminal_load"),
              ("construct", "construct"), ("vae", "vae_load"))}
    for arm in method.ARMS:
        for name, kind in (("decode", "ordinary_decode"), ("encode_float", "posterior_encode"),
                           ("save_mp4", "mp4_save"), ("probe", "ffprobe"),
                           ("read_mp4", "mp4_decode"), ("encode_mp4", "posterior_encode")):
            result[f"{arm}/{name}"] = kind
        for layer in method.LAYERS:
            for key_id in method.KEYS:
                result[f"{arm}/{layer}/{key_id}/state"] = "state_read"
                result[f"{arm}/{layer}/{key_id}/payload"] = "payload_fft_read"
    result.update({"quality/float_rgb": "paired_quality", "quality/mp4": "paired_quality"})
    return result


def fixed_denominator():
    return dict(sources=1, arms=2, layers=3, keys=2, state_tables=12,
                state_positions=264, roi_scores=23232, pooled_scores=5808,
                payload_reads=12, payload_bits=384, quality_pairs=2,
                frames_per_quality_pair=181, temporal_pairs_per_quality_pair=180)


class Journal:
    def __init__(self, output):
        self.output = Path(output)
        self.data = {name: dict(kind=kind, status="NOT_ENTERED") for name, kind in units().items()}
        self.save()

    def save(self):
        dump(self.output/"ledger.json", self.data)

    def call(self, name, operation):
        row = self.data[name]
        if row["status"] != "NOT_ENTERED":
            raise RuntimeError("a fixed unit may only be attempted once")
        row.update(status="STARTED", started_unix=time.time())
        self.save()
        started = time.perf_counter()
        try:
            value = operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                       error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter()-started)
            if hasattr(exc, "receipt"):
                row["detail"] = exc.receipt
            self.save()
            if not isinstance(exc, Exception):
                raise
            return None
        row.update(status="RETURNED", seconds=time.perf_counter()-started)
        self.save()
        return value

    def counts(self):
        result = {}
        for row in self.data.values():
            count = result.setdefault(row["kind"], dict(budget=0, started=0, returned=0, failed=0, interrupted=0))
            count["budget"] += 1
            count["started"] += row["status"] != "NOT_ENTERED"
            count["returned"] += row["status"] == "RETURNED"
            count["failed"] += row["status"] == "FAILED"
            count["interrupted"] += row["status"] == "INTERRUPTED"
        return result


def validate_rgb(rgb):
    if not isinstance(rgb, torch.Tensor) or tuple(rgb.shape) != method.VIDEO_SHAPE:
        raise ValueError("expected clean RGB [181,320,512,3]; no crop/pad/repair")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise ValueError("RGB must be finite floating point")
    if float(rgb.min()) < 0 or float(rgb.max()) > 1:
        raise ValueError("RGB must be in [0,1]")
    return rgb.detach().cpu().float()


class WanBackend:
    def __init__(self, model_id):
        self.model_id = model_id
        self.vae = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_terminal(self, path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        method.validate_latent(value)
        value = value.detach().float()
        method.validate_latent(value)
        return value

    def load(self):
        from diffusers import AutoencoderKLWan
        # Use available model/source. There is no expected revision/version test.
        selected = select_vae_source(self.model_id)
        kwargs = dict(torch_dtype=torch.float32, cache_dir="/content/video-wm-cache/huggingface/hub")
        if selected["subfolder"]:
            kwargs["subfolder"] = selected["subfolder"]
        try:
            self.vae = AutoencoderKLWan.from_pretrained(selected["location"], **kwargs)
            self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        except Exception as exc:
            exc.receipt = dict(model_selection=selected)
            raise
        return dict(device=self.device, model_selection=selected, dtype="float32", frozen=True)

    def decode(self, terminal):
        from runtime.wan.vae import decode_normalized_latent
        return validate_rgb(decode_normalized_latent(self.vae, terminal.to(self.device)))

    def encode(self, rgb):
        from runtime.wan.vae import reencode_rgb24_readback
        result = reencode_rgb24_readback(self.vae, rgb).detach().cpu().float()
        method.validate_latent(result)
        return result

    def save_mp4(self, rgb, path):
        from runtime.wan.io import encode_rgb
        encode_rgb(rgb, path, fps=8, crf=18)
        return dict(path=str(path), bytes=path.stat().st_size, fps=8, crf=18,
                    codec="libx264", pixel_format="yuv420p")

    def probe(self, path):
        proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt", "-of", "json", str(path)],
            check=True, capture_output=True, text=True)
        stream = json.loads(proc.stdout)["streams"][0]
        if (int(stream["height"]), int(stream["width"])) != (320, 512):
            raise ValueError(f"MP4 geometry mismatch: {stream}")
        return stream

    def read_mp4(self, path, probe):
        import numpy as np
        proc = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True)
        height, width = int(probe["height"]), int(probe["width"])
        if len(proc.stdout) != 181*height*width*3:
            raise ValueError("MP4 did not decode into the fixed 181 RGB24 frames")
        rgb = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.uint8).copy().reshape(181, height, width, 3))
        return validate_rgb(rgb.float()/255)


def paired_quality(base, marked):
    validate_rgb(base)
    validate_rgb(marked)
    frames, temporal = [], []
    previous = None
    for i in range(181):
        delta = marked[i].double()-base[i].double()
        frames.append(dict(frame=i, mse=float(delta.square().mean()), max_abs=float(delta.abs().max())))
        if previous is not None:
            diff = delta-previous
            temporal.append(dict(frame_before=i-1, frame_after=i,
                                 mse=float(diff.square().mean()), max_abs=float(diff.abs().max())))
        previous = delta
    mse = sum(row["mse"] for row in frames)/181
    return dict(status="SCORED", frames=frames, temporal=temporal,
                rmse=math.sqrt(mse), psnr_db=None if mse==0 else -10*math.log10(mse),
                identical=mse==0, max_abs=max(row["max_abs"] for row in frames),
                temporal_residual_rmse=math.sqrt(sum(row["mse"] for row in temporal)/180),
                temporal_max_abs=max(row["max_abs"] for row in temporal),
                meaning="paired distortion only; no perceptual threshold")


def _blank_observations():
    return {f"{arm}/{layer}/{key_id}": {kind: dict(status="NOT_ENTERED") for kind in ("state", "payload")}
            for arm in method.ARMS for layer in method.LAYERS for key_id in method.KEYS}


def paired_readout_comparison(evaluation):
    comparisons = {}
    for layer in method.LAYERS:
        for key_id in method.KEYS:
            base = evaluation[f"PAYLOAD_ONLY/{layer}/{key_id}"]
            marked = evaluation[f"PAYLOAD_STATE/{layer}/{key_id}"]
            entry = dict(status="NOT_SCORED", state_status="NOT_SCORED", payload_status="NOT_SCORED",
                state_denominator=22, payload_denominator=32,
                state_arm_status=[x["state"]["status"] for x in (base, marked)],
                payload_arm_status=[x["payload"]["status"] for x in (base, marked)],
                state_rows=[dict(position=i, status="NOT_SCORED") for i in method.STATES],
                payload_rows=[dict(bit=i, status="NOT_SCORED") for i in range(32)])
            if all(x["payload"]["status"]=="EVALUATED" for x in (base, marked)):
                bbits, mbits = base["payload"]["rows"], marked["payload"]["rows"]
                entry.update(payload_status="EVALUATED",
                    payload_errors_base=base["payload"]["bit_errors"],
                    payload_errors_state=marked["payload"]["bit_errors"],
                    lost_correct_bits=[i for i in range(32) if bbits[i]["correct"] and not mbits[i]["correct"]],
                    gained_correct_bits=[i for i in range(32) if not bbits[i]["correct"] and mbits[i]["correct"]],
                    payload_rows=[dict(bit=i, status="EVALUATED", baseline_correct=bbits[i]["correct"],
                                       marked_correct=mbits[i]["correct"]) for i in range(32)])
            if all(x["state"]["status"]=="EVALUATED" for x in (base, marked)):
                entry.update(state_status="EVALUATED",
                    baseline_exact_states=base["state"]["exact_count"], marked_exact_states=marked["state"]["exact_count"],
                    state_rows=[dict(position=i, status="EVALUATED",
                        baseline_margin=base["state"]["rows"][i]["true_minus_max_wrong"],
                        marked_margin=marked["state"]["rows"][i]["true_minus_max_wrong"],
                        baseline_label=base["state"]["rows"][i]["label"],
                        marked_label=marked["state"]["rows"][i]["label"]) for i in method.STATES])
            valid = sum(entry[name]=="EVALUATED" for name in ("state_status", "payload_status"))
            entry["status"] = ("NOT_SCORED", "PARTIAL", "EVALUATED")[valid]
            comparisons[f"{layer}/{key_id}"] = entry
    return comparisons


def run(source, output, model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", *, backend=None):
    """One attempt per output directory; dependency injection is for CPU tests."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = execution_artifacts(output)
    if artifacts:
        return existing_attempt(output, artifacts)
    try:
        with (output/"attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(source=str(source), started_unix=time.time()), stream)
    except FileExistsError:
        # Read-only: another process may still be running. Never alter its ledger.
        return existing_attempt(output, execution_artifacts(output))
    journal = Journal(output)
    observations = _blank_observations()
    result = dict(status="RUNNING", source_path=str(source), fixed_denominator=fixed_denominator(),
                  observations=observations, mapping=method.support_mapping(),
                  construction=dict(status="NOT_ENTERED"), quality={k: dict(status="NOT_ENTERED") for k in ("float_rgb", "mp4")},
                  environment=dict(python=platform.python_version(), torch=torch.__version__),
                  zero_budget=["DiT", "native_sampling", "scheduler", "gradient_decode", "VJP", "retry"],
                  evidence_ceiling="single cached native point, clean public grid; not online generation or attack recovery")
    backend = backend or WanBackend(model_id)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        result["counts"] = journal.counts()
        dump(output/"checkpoint.json", result)

    def read_feature(arm, layer, feature):
        method.validate_latent(feature)
        for key_id, key in method.KEYS.items():
            slot = f"{arm}/{layer}/{key_id}"
            for kind, reader in (("state", method.read_state), ("payload", read_latent_bits)):
                def read_and_save():
                    raw = reader(feature, key)
                    dump(output/"blind"/arm/layer/f"{key_id}_{kind}.json", raw)
                    return raw
                raw = journal.call(f"{slot}/{kind}", read_and_save)
                observations[slot][kind] = raw if raw is not None else dict(journal.data[f"{slot}/{kind}"])
                checkpoint()

    def save_feature(arm, layer, feature):
        path = output/"features"/arm/f"{layer}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature, path)
        return feature

    checkpoint()
    started = time.perf_counter()
    media = {}
    try:
        base = journal.call("source", lambda: backend.load_terminal(source))
        candidates = {}
        if base is not None:
            method.validate_latent(base)
            candidates["PAYLOAD_ONLY"] = base
            made = journal.call("construct", lambda: method.write_state(base, method.KEYS["CORRECT"]))
            if made is not None:
                candidates["PAYLOAD_STATE"], result["construction"] = made
            else:
                result["construction"] = dict(journal.data["construct"])
            # terminal readers do not depend on model availability.
            for arm, feature in candidates.items():
                save_feature(arm, "terminal", feature)
                read_feature(arm, "terminal", feature)
            loaded = journal.call("vae", backend.load)
            result["model"] = loaded if loaded is not None else dict(journal.data["vae"])
            checkpoint()
            if loaded is not None:
                for arm, feature in candidates.items():
                    rgb = journal.call(f"{arm}/decode", lambda: backend.decode(feature))
                    if rgb is None:
                        continue
                    media[(arm, "float_rgb")] = rgb
                    encoded = journal.call(f"{arm}/encode_float", lambda: save_feature(arm, "float_rgb", backend.encode(rgb)))
                    if encoded is not None:
                        read_feature(arm, "float_rgb", encoded)
                    path = output/"media"/f"{arm}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    saved = journal.call(f"{arm}/save_mp4", lambda: backend.save_mp4(rgb, path))
                    if saved is None:
                        continue
                    probe = journal.call(f"{arm}/probe", lambda: backend.probe(path))
                    if probe is None:
                        continue
                    dump(output/"media"/f"{arm}_probe.json", probe)
                    reopened = journal.call(f"{arm}/read_mp4", lambda: backend.read_mp4(path, probe))
                    if reopened is None:
                        continue
                    media[(arm, "mp4")] = reopened
                    encoded = journal.call(f"{arm}/encode_mp4", lambda: save_feature(arm, "mp4", backend.encode(reopened)))
                    if encoded is not None:
                        read_feature(arm, "mp4", encoded)
        for layer in ("float_rgb", "mp4"):
            if all((arm, layer) in media for arm in method.ARMS):
                quality = journal.call(f"quality/{layer}", lambda: paired_quality(
                    media[("PAYLOAD_ONLY", layer)], media[("PAYLOAD_STATE", layer)]))
                result["quality"][layer] = quality or dict(journal.data[f"quality/{layer}"])
                dump(output/"quality"/f"{layer}.json", result["quality"][layer])
    except BaseException as exc:
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
    # Only now read evaluation truth; every available raw/decision is already on disk.
    evaluation = {slot: dict(state=method.evaluate_state(raw["state"], list(method.STATES)),
                            payload=method.evaluate_payload(raw["payload"], message_bits("OKOK")))
                  for slot, raw in observations.items()}
    dump(output/"evaluation.json", evaluation)
    result["evaluation"] = evaluation
    result["paired_readout"] = paired_readout_comparison(evaluation)
    result["seconds"] = time.perf_counter()-started
    result["counts"] = journal.counts()
    result["calls"] = journal.data
    result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    result["status"] = "EXECUTION_COMPLETE" if not result.get("fatal_error") and all(
        row["status"]=="RETURNED" for row in journal.data.values()) else "INCOMPLETE"
    dump(output/"result.json", result)
    checkpoint()
    return result
