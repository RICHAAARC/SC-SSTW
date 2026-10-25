"""Single user-run paired terminal projection / VAE / MP4 observation package."""
from __future__ import annotations

import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch

from main.tube_state import local_joint_positive_projection_margin_v1 as method
from main.tube_state.grow_video_reference import message_bits, read_latent_bits
from runtime.wan.local_joint_positive_projection_margin_quality_v1 import distortion

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_VAE_CACHE = Path("/content/video-wm-cache/models/Wan2.1-T2V-1.3B-Diffusers")
QUALITY_LAYERS = ("preclip", "float_rgb", "rgb8", "mp4")


def execution_artifacts(output):
    """Copied narrow filesystem semantics: any scientific residue spends attempt."""
    output = Path(output)
    records = ("attempt.json", "result.json", "checkpoint.json", "ledger.json", "evaluation.json", "protocol.json", "construction.json")
    found = [name+suffix for name in records for suffix in ("", ".tmp")
             if (output/(name+suffix)).exists() or (output/(name+suffix)).is_symlink()]
    for folder in ("blind", "features", "media", "quality"):
        root = output/folder
        if root.is_dir():
            found.extend(str(path.relative_to(output)) for path in root.rglob("*")
                         if path.is_file() or path.is_symlink())
    return sorted(found)


def existing_attempt(output, artifacts):
    errors = {}
    for name in ("result.json", "checkpoint.json", "result.json.tmp", "checkpoint.json.tmp"):
        path = output/name
        if not path.is_file():
            continue
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("saved record is not an object")
        except (OSError, ValueError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        return dict(status="EXISTING_ATTEMPT_NOT_REEXECUTED", saved=saved, saved_record=name,
                    existing_artifacts=artifacts, read_errors=errors)
    return dict(status="EXISTING_ATTEMPT_WITHOUT_CHECKPOINT", existing_artifacts=artifacts,
                read_errors=errors, fixed_denominator=fixed_denominator())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def select_vae_source(model_id, cache_base=DEFAULT_VAE_CACHE):
    """Same successful cache selection and one-load semantics as card44."""
    explicit = Path(model_id)
    if explicit.is_dir():
        return dict(location=str(explicit), subfolder="vae" if (explicit/"vae").is_dir() else None,
                    origin="explicit_local", weights_may_download=False)
    if model_id == DEFAULT_MODEL_ID:
        base = Path(cache_base)
        candidates = [base/"available", base]
        if base.is_dir():
            candidates.extend(sorted(p for p in base.iterdir() if p.is_dir()))
        for path in candidates:
            vae = path/"vae"
            if (vae/"config.json").is_file() and (any(vae.glob("*.safetensors")) or any(vae.glob("*.bin"))):
                return dict(location=str(path), subfolder="vae", origin="session_cache", weights_may_download=False)
    return dict(location=model_id, subfolder="vae", origin="model_id", weights_may_download=True)


def validate_rgb(value, *, preclip=False, pixels=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != method.VIDEO_SHAPE:
        raise ValueError("expected complete clean [181,320,512,3] RGB; no shape repair")
    if pixels:
        if value.dtype != torch.uint8:
            raise ValueError("expected already quantized uint8 RGB24")
    else:
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("finite floating RGB required")
        if not preclip and (float(value.min()) < 0 or float(value.max()) > 1):
            raise ValueError("clipped RGB must be in [0,1]")
    return value.detach().cpu()


class WanBackend:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae = None
        self.blocks = {"encoder": None, "decoder": None}
        self.hooks = []

    def load_terminal(self, path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        method.validate_latent(value)
        value = value.detach().float()
        method.validate_latent(value)
        return value

    def load(self):
        from diffusers import AutoencoderKLWan
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
        for name in self.blocks:
            module = getattr(self.vae, name, None)
            if module is not None and callable(getattr(module, "register_forward_pre_hook", None)):
                self.blocks[name] = 0
                def count(_module, _args, label=name):
                    self.blocks[label] += 1
                self.hooks.append(module.register_forward_pre_hook(count))
        return dict(device=self.device, model_selection=selected, dtype="float32", frozen=True,
                    internal_block_observation="encoder/decoder module forward hooks when available; otherwise null")

    def decode(self, terminal):
        from runtime.wan.vae import decode_normalized_latent
        observed = {}
        def retain(preclip):
            observed["preclip"] = validate_rgb(preclip, preclip=True)
        rgb = decode_normalized_latent(self.vae, terminal.to(self.device), diagnostic_observer=retain)
        return dict(preclip=observed["preclip"], float_rgb=validate_rgb(rgb))

    def encode(self, rgb):
        from runtime.wan.vae import reencode_rgb24_readback
        value = rgb.float()/255 if rgb.dtype == torch.uint8 else rgb
        result = reencode_rgb24_readback(self.vae, value).detach().cpu().float()
        method.validate_latent(result)
        return result

    def quantize(self, rgb):
        from runtime.wan.vae import quantize_rgb8_no_codec
        return validate_rgb(quantize_rgb8_no_codec(rgb), pixels=True)

    def save_mp4(self, pixels, path):
        from runtime.wan.io import encode_rgb8
        encode_rgb8(pixels, path, fps=8, crf=18)
        return dict(path=str(path), bytes=path.stat().st_size, fps=8, crf=18,
                    codec="libx264", pixel_format="yuv420p", input="same saved uint8 RGB24; no requantization")

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
            raise ValueError("MP4 did not decode into 181 complete RGB24 frames")
        pixels = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.uint8).copy().reshape(181, height, width, 3))
        return validate_rgb(pixels, pixels=True)


def units():
    result = {"source": "terminal_load", "construct": "construct", "vae": "vae_load"}
    result.update({f"codebook/{key}": "public_codebook" for key in method.KEYS})
    for arm in method.ARMS:
        for name, kind in (("decode", "ordinary_decode"), ("encode_float_rgb", "posterior_encode"),
                           ("quantize", "rgb8_quantize"), ("encode_rgb8", "posterior_encode"),
                           ("save_mp4", "mp4_save"), ("probe", "ffprobe"), ("read_mp4", "mp4_decode"),
                           ("encode_mp4", "posterior_encode")):
            result[f"{arm}/{name}"] = kind
        for layer in method.LAYERS:
            for key in method.KEYS:
                result[f"{arm}/{layer}/{key}/local"] = "joint_read"
                result[f"{arm}/{layer}/{key}/legacy"] = "payload_fft_read"
        for transition in ("clip", "quantize", "codec"):
            result[f"quality/{arm}/{transition}"] = "processing_distortion"
    result.update({f"quality/paired/{layer}": "paired_quality" for layer in QUALITY_LAYERS})
    return result


def fixed_denominator():
    return dict(sources=1, arms=2, layers=4, keys=2, joint_tables=16, roi_scores=76032, pooled_scores=19008,
                state_positions=352, state_competitors=7392, local_payload_chips=11264,
                local_message_bits=512, legacy_message_bits=512, paired_quality_layers=4,
                processing_distortion_pairs=6, frames_per_quality=181, adjacent_pairs_per_quality=180)


def unavailable_quality(status, detail=None):
    return dict(status=status, frame_denominator=181, adjacent_denominator=180, detail=detail,
                frames=[dict(frame=i, status=status) for i in range(181)],
                adjacent=[dict(frame_before=i, frame_after=i+1, status=status) for i in range(180)])


class Journal:
    def __init__(self, output, backend):
        self.output, self.backend = output, backend
        self.data = {name: dict(kind=kind, status="NOT_ENTERED") for name, kind in units().items()}
        self.save()

    def save(self):
        dump(self.output/"ledger.json", self.data)

    def call(self, name, operation):
        row = self.data[name]
        if row["status"] != "NOT_ENTERED":
            raise RuntimeError("fixed operation cannot be attempted twice")
        row.update(status="STARTED", started_unix=time.time())
        self.save()
        started = time.perf_counter()
        blocks = dict(getattr(self.backend, "blocks", {}))
        value = None
        try:
            value = operation()
            row["status"] = "RETURNED"
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED", error=f"{type(exc).__name__}: {exc}")
            if hasattr(exc, "receipt"):
                row["detail"] = exc.receipt
            if not isinstance(exc, Exception):
                raise
        finally:
            row["seconds"] = time.perf_counter()-started
            row["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
            if row["kind"] in ("ordinary_decode", "posterior_encode"):
                after = getattr(self.backend, "blocks", {})
                row["internal_block_forward_delta"] = {k: None if v is None or after.get(k) is None else after[k]-v for k, v in blocks.items()}
            self.save()
        return value

    def counts(self):
        result = {}
        for row in self.data.values():
            group = result.setdefault(row["kind"], dict(budget=0, started=0, returned=0, failed=0, interrupted=0))
            group["budget"] += 1
            group["started"] += row["status"] != "NOT_ENTERED"
            group["returned"] += row["status"] == "RETURNED"
            group["failed"] += row["status"] == "FAILED"
            group["interrupted"] += row["status"] == "INTERRUPTED"
        return result


def compare_bits(base, marked):
    rows = [dict(bit=i, baseline=base["rows"][i], candidate=marked["rows"][i]) for i in range(32)]
    common = [i for i in range(32) if all(x["rows"][i]["decoded"] is not None for x in (base, marked))]
    return dict(denominator=32, rows=rows, common_bit_indices=common, common_bit_count=len(common),
                baseline_common_ber=None if not common else sum(not base["rows"][i]["correct"] for i in common)/len(common),
                candidate_common_ber=None if not common else sum(not marked["rows"][i]["correct"] for i in common)/len(common),
                added_coverage=[i for i in range(32) if base["rows"][i]["decoded"] is None and marked["rows"][i]["decoded"] is not None],
                lost_correct_bits=[i for i in range(32) if base["rows"][i]["correct"] is True and marked["rows"][i]["correct"] is not True],
                gained_correct_bits=[i for i in range(32) if base["rows"][i]["correct"] is not True and marked["rows"][i]["correct"] is True])


def evaluate_all(observations):
    bits = message_bits("OKOK")
    evaluated = {}
    for slot, raw in observations.items():
        legacy = raw["legacy"]
        rows = [dict(decoded=b) for b in legacy.get("decoded_bits", [])]
        evaluated[slot] = dict(local=method.evaluate_local(raw["local"], list(method.STATES), bits),
                               legacy=method.evaluate_bits(rows, bits, legacy.get("status", "NOT_ENTERED")))
    pairs = {}
    for layer in method.LAYERS:
        for key in method.KEYS:
            p, j = [evaluated[f"{arm}/{layer}/{key}"] for arm in method.ARMS]
            pairs[f"{layer}/{key}"] = dict(state_denominator=22, state_rows=[dict(position=i,
                baseline=p["local"]["states"][i], candidate=j["local"]["states"][i]) for i in method.STATES],
                local_payload=compare_bits(p["local"]["payload"], j["local"]["payload"]),
                legacy_payload=compare_bits(p["legacy"], j["legacy"]))
    return evaluated, pairs


def run(source, output, model_id=DEFAULT_MODEL_ID, *, backend=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = execution_artifacts(output)
    if artifacts:
        return existing_attempt(output, artifacts)
    try:
        with (output/"attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(source=str(source), started_unix=time.time()), stream)
    except FileExistsError:
        return existing_attempt(output, execution_artifacts(output))
    backend = backend or WanBackend(model_id)
    journal = Journal(output, backend)
    observations = {f"{arm}/{layer}/{key}": {kind: dict(status="NOT_ENTERED") for kind in ("local", "legacy")}
                    for arm in method.ARMS for layer in method.LAYERS for key in method.KEYS}
    result = dict(status="RUNNING", source_path=str(source), fixed_denominator=fixed_denominator(),
                  protocol=method.protocol_record(), construction=dict(status="NOT_ENTERED"), quality={},
                  environment=dict(python=platform.python_version(), torch=torch.__version__),
                  zero_budget=["DiT", "native_sampling", "gradient", "VJP", "retry", "additional_candidate"],
                  evidence_ceiling="paired terminal projection and clean media channel only; no unknown-start sync or multistep generation claim")
    for layer in QUALITY_LAYERS:
        result["quality"][f"paired/{layer}"] = unavailable_quality("NOT_ENTERED")
    for arm in method.ARMS:
        for stage in ("clip", "quantize", "codec"):
            result["quality"][f"{arm}/{stage}"] = unavailable_quality("NOT_ENTERED")
    if getattr(backend, "device", "cpu") == "cuda":
        torch.cuda.reset_peak_memory_stats()
    books, media = {}, {}

    def checkpoint():
        result["counts"] = journal.counts()
        dump(output/"checkpoint.json", result)

    def save_tensor(folder, arm, name, value):
        path = output/folder/arm/f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return value

    def read_feature(arm, layer, feature):
        method.validate_latent(feature)
        for key_id, key in method.KEYS.items():
            slot = f"{arm}/{layer}/{key_id}"
            for kind in ("local", "legacy"):
                if kind == "local" and books.get(key_id) is None:
                    continue
                def observe():
                    raw = method.read_local(feature, key, books[key_id]) if kind == "local" else read_latent_bits(feature, key)
                    dump(output/"blind"/arm/layer/f"{key_id}_{kind}.json", raw)
                    return raw
                raw = journal.call(f"{slot}/{kind}", observe)
                observations[slot][kind] = raw if raw is not None else dict(journal.data[f"{slot}/{kind}"])
                checkpoint()

    def quality(name, left, right, meaning):
        row = journal.call("quality/"+name, lambda: distortion(left, right, meaning=meaning))
        result["quality"][name] = row if row is not None else unavailable_quality(journal.data["quality/"+name]["status"], journal.data["quality/"+name])
        dump(output/"quality"/(name+".json"), result["quality"][name])
        checkpoint()

    def encode_and_read(arm, layer, rgb):
        feature = journal.call(f"{arm}/encode_{layer}", lambda: save_tensor("features", arm, layer, backend.encode(rgb)))
        if feature is not None:
            read_feature(arm, layer, feature)

    started = time.perf_counter()
    dump(output/"protocol.json", method.protocol_record())
    checkpoint()
    try:
        base = journal.call("source", lambda: backend.load_terminal(source))
        candidates = {}
        if base is not None:
            method.validate_latent(base)
            candidates[method.ARMS[0]] = save_tensor("features", method.ARMS[0], "terminal", base)
            for key_id, key in method.KEYS.items():
                books[key_id] = journal.call(f"codebook/{key_id}", lambda key=key: method.public_codebook(key))
            read_feature(method.ARMS[0], "terminal", base)
            host = observations[f"{method.ARMS[0]}/terminal/CORRECT"]["local"]
            if books.get("CORRECT") is not None and host.get("status") == "READ":
                made = journal.call("construct", lambda: method.write_projection(base, host, message_bits("OKOK"), books["CORRECT"]))
                if made is not None:
                    candidate, result["construction"] = made
                    candidates[method.ARMS[1]] = save_tensor("features", method.ARMS[1], "terminal", candidate)
                    dump(output/"construction.json", result["construction"])
                    read_feature(method.ARMS[1], "terminal", candidate)
                else:
                    result["construction"] = dict(journal.data["construct"])
            loaded = journal.call("vae", backend.load)
            result["model"] = loaded if loaded is not None else dict(journal.data["vae"])
            checkpoint()
            if loaded is not None:
                for arm, feature in candidates.items():
                    decoded = journal.call(f"{arm}/decode", lambda: backend.decode(feature))
                    if decoded is None:
                        continue
                    for layer in ("preclip", "float_rgb"):
                        value = validate_rgb(decoded[layer], preclip=layer == "preclip")
                        media[arm, layer] = save_tensor("media", arm, layer, value)
                    quality(f"{arm}/clip", decoded["preclip"], decoded["float_rgb"], "clipping effect within this arm")
                    encode_and_read(arm, "float_rgb", decoded["float_rgb"])
                    pixels = journal.call(f"{arm}/quantize", lambda: save_tensor("media", arm, "rgb8", backend.quantize(decoded["float_rgb"])))
                    if pixels is None:
                        continue
                    media[arm, "rgb8"] = pixels
                    quality(f"{arm}/quantize", decoded["float_rgb"], pixels, "RGB8 rounding effect within this arm")
                    encode_and_read(arm, "rgb8", pixels)
                    path = output/"media"/f"{arm}.mp4"
                    saved = journal.call(f"{arm}/save_mp4", lambda: backend.save_mp4(pixels, path))
                    if saved is None:
                        continue
                    probe = journal.call(f"{arm}/probe", lambda: backend.probe(path))
                    if probe is None:
                        continue
                    dump(output/"media"/f"{arm}_probe.json", probe)
                    reopened = journal.call(f"{arm}/read_mp4", lambda: backend.read_mp4(path, probe))
                    if reopened is None:
                        continue
                    media[arm, "mp4"] = save_tensor("media", arm, "mp4_readback_rgb8", reopened)
                    quality(f"{arm}/codec", pixels, reopened, "codec roundtrip effect within this arm")
                    encode_and_read(arm, "mp4", reopened)
        for layer in QUALITY_LAYERS:
            if all((arm, layer) in media for arm in method.ARMS):
                quality(f"paired/{layer}", media[method.ARMS[0], layer], media[method.ARMS[1], layer],
                        "incremental distortion relative to PAYLOAD_ONLY; not total distortion relative to OFF or perceptual acceptance")
    except BaseException as exc:
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
    # Evaluator is the first consumer of semantic truth for blind outputs.
    # Writer truth never enters read_feature or either reader.
    evaluation, pairs = evaluate_all(observations)
    dump(output/"evaluation.json", evaluation)
    result["evaluation"] = evaluation
    result["paired_readout"] = pairs
    result["seconds"] = time.perf_counter()-started
    result["counts"], result["calls"] = journal.counts(), journal.data
    result["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    if getattr(backend, "device", "cpu") == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["status"] = "EXECUTION_COMPLETE" if not result.get("fatal_error") and all(
        row["status"] == "RETURNED" for row in journal.data.values()) else "INCOMPLETE"
    result["scientific_pass"] = None
    dump(output/"result.json", result)
    checkpoint()
    return result
