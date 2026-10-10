"""Variable-length RGB8 -> fixed-parameter H264 -> RGB24 roundtrip.

This is the narrow temporal-attack counterpart of ``fixed_rgb_media``.  It
keeps 320x512 RGB, 8 fps, libx264 CRF18 and yuv420p fixed while accepting the
adopted attack lengths.  It never pads or truncates decoded media.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


HEIGHT = 320
WIDTH = 512
CHANNELS = 3
FPS = 8


def _identity(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"sha256": value, "sha256_observation_error": None, "sha256_blocking": False}
    except OSError as exc:
        return {
            "sha256": None,
            "sha256_observation_error": f"{type(exc).__name__}: {exc}",
            "sha256_blocking": False,
        }


def _array(value):
    import numpy as np

    candidate = value
    for method in ("detach", "cpu"):
        function = getattr(candidate, method, None)
        if callable(function):
            candidate = function()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    result = np.asarray(candidate)
    if result.dtype != np.uint8 or result.ndim != 4 or tuple(result.shape[1:]) != (HEIGHT, WIDTH, CHANNELS):
        raise ValueError("codec input must be uint8 [T,320,512,3]")
    if int(result.shape[0]) < 1:
        raise ValueError("codec input must contain at least one frame")
    return result


def commands(mp4_path: Path, frames: int) -> dict[str, list[str]]:
    return {
        "save": [
            "ffmpeg", "-v", "error", "-threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-frames:v", str(frames), "-n", str(mp4_path),
        ],
        "probe": [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
            "-of", "json", str(mp4_path),
        ],
        "read": [
            "ffmpeg", "-v", "error", "-threads", "1", "-noautorotate",
            "-i", str(mp4_path), "-map", "0:v:0", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
    }


def roundtrip(rgb8, output: Path, *, media_output: Path | None = None, event=lambda *_: None):
    """Persist one edited RGB raster, publish once, and return exact readback."""

    import numpy as np

    array = _array(rgb8)
    frames = int(array.shape[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    media_output = Path(media_output) if media_output is not None else output
    media_output.mkdir(parents=True, exist_ok=True)
    raster = output / "edited.rgb8"
    mp4 = media_output / "published.mp4"
    received = output / "received.rgb8"
    raw = array.tobytes(order="C")
    raster.write_bytes(raw)
    raster_receipt = {
        "status": "SAVED", "path": str(raster), "bytes": len(raw),
        "shape": list(array.shape), "dtype": "uint8", **_identity(raster),
    }
    event("edited_rgb", dict(raster_receipt))
    cmds = commands(mp4, frames)
    row = {"status": "RUNNING", "command": cmds["save"], "path": str(mp4)}
    event("mp4", dict(row))
    child = subprocess.run(cmds["save"], input=raw, capture_output=True, check=False)
    (media_output / "save.stdout.txt").write_bytes(child.stdout)
    (media_output / "save.stderr.txt").write_bytes(child.stderr)
    if child.returncode:
        raise RuntimeError(f"variable-length MP4 save exited {child.returncode}")
    row.update(status="SAVED", returncode=child.returncode, bytes=mp4.stat().st_size, **_identity(mp4))
    event("mp4", dict(row))
    probe = subprocess.run(cmds["probe"], capture_output=True, check=False)
    (media_output / "probe.json").write_bytes(probe.stdout)
    (media_output / "probe.stderr.txt").write_bytes(probe.stderr)
    if probe.returncode:
        raise RuntimeError(f"variable-length MP4 probe exited {probe.returncode}")
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    if (int(stream["height"]), int(stream["width"]), int(stream["nb_frames"])) != (HEIGHT, WIDTH, frames):
        raise ValueError("variable-length MP4 frame geometry/count mismatch")
    decoded = subprocess.run(cmds["read"], capture_output=True, check=False)
    (media_output / "read.stderr.txt").write_bytes(decoded.stderr)
    if decoded.returncode:
        raise RuntimeError(f"variable-length MP4 readback exited {decoded.returncode}")
    expected_bytes = frames * HEIGHT * WIDTH * CHANNELS
    if len(decoded.stdout) != expected_bytes:
        raise ValueError("variable-length MP4 readback byte count mismatch")
    temporary = received.with_suffix(".tmp")
    temporary.write_bytes(decoded.stdout)
    os.replace(temporary, received)
    received_receipt = {
        "status": "SAVED", "path": str(received), "bytes": expected_bytes,
        "shape": [frames, HEIGHT, WIDTH, CHANNELS], "dtype": "uint8",
        **_identity(received),
    }
    event("received_rgb", dict(received_receipt))
    return np.frombuffer(decoded.stdout, np.uint8).reshape(frames, HEIGHT, WIDTH, CHANNELS).copy(), {
        "edited_rgb": raster_receipt, "mp4": row,
        "probe": {"status": "COMPLETE", "metadata": metadata, "command": cmds["probe"]},
        "received_rgb": received_receipt,
        "codec": {"codec": "libx264", "fps": FPS, "crf": 18, "pix_fmt": "yuv420p"},
    }


def decode_published(mp4_path: Path, output_path: Path, *, expected_frames: int | None = None):
    """Decode and validate an already-published variable-length MP4.

    Recovery uses this entry point instead of publishing the edited raster a
    second time.  The returned receipt describes the new local RGB8 cache and
    keeps the persisted MP4 as its source.
    """
    import numpy as np

    mp4_path = Path(mp4_path)
    if not mp4_path.is_file():
        raise FileNotFoundError(f"published MP4 is unavailable: {mp4_path}")
    probe_command = commands(mp4_path, expected_frames or 1)["probe"]
    probe = subprocess.run(probe_command, capture_output=True, check=False)
    if probe.returncode:
        raise RuntimeError(f"variable-length MP4 probe exited {probe.returncode}")
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    frames = int(stream["nb_frames"])
    if (int(stream["height"]), int(stream["width"])) != (HEIGHT, WIDTH):
        raise ValueError("published MP4 geometry mismatch")
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"published MP4 frame count {frames} != expected {expected_frames}")
    read_command = commands(mp4_path, frames)["read"]
    decoded = subprocess.run(read_command, capture_output=True, check=False)
    if decoded.returncode:
        raise RuntimeError(f"variable-length MP4 readback exited {decoded.returncode}")
    expected_bytes = frames * HEIGHT * WIDTH * CHANNELS
    if len(decoded.stdout) != expected_bytes:
        raise ValueError("published MP4 readback byte count mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(decoded.stdout)
    os.replace(temporary, output_path)
    receipt = {
        "status": "SAVED", "path": str(output_path), "bytes": expected_bytes,
        "shape": [frames, HEIGHT, WIDTH, CHANNELS], "dtype": "uint8",
        "source_mp4": str(mp4_path), "source_mp4_observation": _identity(mp4_path),
        **_identity(output_path),
    }
    return np.frombuffer(decoded.stdout, np.uint8).reshape(frames, HEIGHT, WIDTH, CHANNELS).copy(), receipt
