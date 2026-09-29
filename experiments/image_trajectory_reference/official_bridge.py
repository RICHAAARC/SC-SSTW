"""Load byte-verified upstream core and expose a truth-free PNG reader."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "diagnostics/image_trajectory_reference_v1/official_source_manifest.json"
UPSTREAM_SHA = "6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(upstream_root):
    root = Path(upstream_root).resolve()
    entry = next(r for r in json.loads(MANIFEST.read_text())["repositories"]
                 if r["repo"] == "luopengchen/GROW")
    if entry["commit"] != UPSTREAM_SHA:
        raise ValueError("fixed upstream commit mismatch")
    for rel, item in entry["files"].items():
        if sha256(root / rel) != item["sha256"]:
            raise ValueError(f"upstream byte mismatch: {rel}")
    return dict(repo=entry["repo"], commit=UPSTREAM_SHA, files_verified=len(entry["files"]),
                files={p: item["sha256"] for p,item in entry["files"].items()})


def load_official(upstream_root):
    """Execute unchanged core files, avoiding the unused AttackSuite facade.

    No upstream function body is copied, monkeypatched or reimplemented. The
    namespace wrapper omits grow/__init__.py's unrelated torchvision attacks.
    """
    verify_source(upstream_root)
    package_path = (Path(upstream_root) / "GROW/grow").resolve()
    name = "_fixed_image_reference_grow"
    if name in sys.modules:
        if list(sys.modules[name].__path__) != [str(package_path)]:
            raise ValueError("upstream module already bound to another source root")
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        sys.modules[name] = package
    config = importlib.import_module(name + ".config")
    watermark = importlib.import_module(name + ".watermark")
    codec = importlib.import_module(name + ".codec")
    return config.GrowConfig, watermark.GrowWatermarker, codec


@dataclass(frozen=True)
class PublicReaderConfig:
    """No message, writer output, confidence target or truth field exists."""
    device: str
    dtype: str
    secret_key: str
    message_bit_length: int = 32
    dct_min: float = 0.2
    dct_max: float = 0.5
    channels: tuple = (0, 1, 2, 3)
    center_ratio: float = 1.0


def read_saved_png(path, public_config, pipe, upstream_root):
    from PIL import Image
    _, Watermarker, codec = load_official(upstream_root)
    if not isinstance(public_config, PublicReaderConfig):
        raise TypeError("independent public reader configuration required")
    if public_config.message_bit_length != 32:
        raise ValueError("fixed public 32-bit payload length required")
    path = Path(path)
    digest_before = sha256(path)
    reader = Watermarker(public_config, pipe=pipe)
    with Image.open(path) as opened:
        if opened.format != "PNG" or opened.size != (512,512):
            raise ValueError("fixed 512x512 PNG readback required")
        opened.load()
        image = opened.convert("RGB")
    # The only upstream reading call. It has no truth or confidence selection.
    bits = reader._read_bits(image)
    if len(bits) != 32 or any(type(bit) is not int or bit not in (0,1) for bit in bits):
        raise ValueError("invalid 32-bit official readout")
    if sha256(path) != digest_before:
        raise RuntimeError("PNG changed during readback")
    return dict(status="READ", png_sha256=digest_before, decoded_bits=bits,
                decoded_string_diagnostic=codec.bits_to_message(bits),
                input_kind="reopened_saved_PNG", truth_used=False,
                public_message_bit_length=32,
                function="GrowWatermarker._read_bits")


def layout_receipt(correct_key, wrong_key):
    """Audit the upstream coordinate seed/order; no alternate receiver."""
    import numpy as np
    def one(key):
        seed = sum(map(ord,key))
        coords = [(h,w) for h in range(12,32) for w in range(12,32)]
        np.random.default_rng(seed).shuffle(coords)
        # Each channel carries 8 bits. Order alone is insufficient: also check
        # the coordinate-to-bit assignment (j % 8) used by the official reader.
        assignment = sorted((h,w,j%8) for j,(h,w) in enumerate(coords))
        digest = lambda x: hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
        return dict(seed=seed, coordinate_order_sha256=digest(coords),
                    bit_assignment_sha256=digest(assignment), coordinates=coords,
                    assignment=assignment)
    correct, wrong = one(correct_key), one(wrong_key)
    if (correct["seed"] == wrong["seed"] or correct["coordinates"] == wrong["coordinates"]
            or correct["assignment"] == wrong["assignment"]):
        raise ValueError("wrong-key layout or bit assignment collides")
    for item in (correct,wrong):
        item.pop("coordinates");item.pop("assignment")
    return dict(correct=correct, wrong=wrong, layout_different=True,
                public_latent_shape=[1,4,64,64], mask_elements=1600,
                audit_only=True)
