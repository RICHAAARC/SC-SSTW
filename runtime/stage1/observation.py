"""Selected frame-difference weighted centroid, fixed camera/single subject.

No tracking or foreground identity inference: multiple moving objects remain
outside the construction's content assumption. All thresholds are explicit.
"""
from dataclasses import dataclass, asdict
import math
import json
import subprocess


@dataclass(frozen=True)
class ObserverConfig:
    pixel_delta: int
    min_support_fraction: float
    max_support_fraction: float

    def __post_init__(self):
        if type(self.pixel_delta) is not int or not 0 <= self.pixel_delta < 255:
            raise ValueError("pixel_delta must be integer in [0,254]")
        if not 0 < self.min_support_fraction <= self.max_support_fraction <= 1:
            raise ValueError("invalid support bounds")


@dataclass(frozen=True)
class Observation:
    sample_index: int
    time_seconds: float
    q: tuple[float, float] | None
    valid: bool
    reason: str
    support_fraction: float
    motion_mass: float

    def __post_init__(self):
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("invalid sample index")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("invalid public time")
        if not math.isfinite(self.support_fraction) or not 0 <= self.support_fraction <= 1:
            raise ValueError("invalid support")
        if not math.isfinite(self.motion_mass) or self.motion_mass < 0:
            raise ValueError("invalid motion mass")
        if self.valid:
            if self.q is None or len(self.q) != 2 or not all(math.isfinite(v) for v in self.q):
                raise ValueError("valid observation requires finite two-dimensional q")
            if self.reason != "VALID":
                raise ValueError("valid reason mismatch")
        elif self.q is not None or not self.reason or self.reason == "VALID":
            raise ValueError("invalid observation must retain reason and no q")


def observe_frames(frames, *, width, height, sample_hz, config):
    if width < 2 or height < 2 or not math.isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("invalid geometry or sampling")
    rows, previous = [], None
    for index, frame in enumerate(frames):
        if len(frame) != width * height or any(type(v) is not int or not 0 <= v <= 255 for v in frame):
            raise ValueError("expected complete uint8 grayscale frame")
        q, reason, support, mass = None, "INITIAL_FRAME", 0.0, 0.0
        if previous is not None:
            weights = [abs(a - b) if abs(a - b) > config.pixel_delta else 0
                       for a, b in zip(frame, previous)]
            mass = float(sum(weights))
            support = sum(w > 0 for w in weights) / (width * height)
            if mass == 0 or support < config.min_support_fraction:
                reason = "NO_MOTION_SUPPORT"
            elif support > config.max_support_fraction:
                reason = "GLOBAL_CHANGE_OR_CUT"
            else:
                q = (sum((i % width) * w for i, w in enumerate(weights)) / mass / (width - 1),
                     sum((i // width) * w for i, w in enumerate(weights)) / mass / (height - 1))
                reason = "VALID"
        rows.append(Observation(index, index / sample_hz, q, q is not None, reason, support, mass))
        previous = tuple(frame)
    return tuple(rows)


def summarize_observations(rows):
    points = [r.q for r in rows if r.valid]
    n = len(points)
    if n:
        x, y = (sum(p[i] for p in points) / n for i in range(2))
        xx = sum((p[0] - x) ** 2 for p in points) / n
        yy = sum((p[1] - y) ** 2 for p in points) / n
        xy = sum((p[0] - x) * (p[1] - y) for p in points) / n
        determinant = max(0.0, xx * yy - xy * xy)
    else:
        xx = yy = xy = determinant = None
    return {"count": len(rows), "valid_count": n, "valid_fraction": n / len(rows) if rows else 0.0,
            "covariance_xx": xx, "covariance_yy": yy, "covariance_xy": xy,
            "covariance_determinant": determinant,
            "degeneracy_decision": "UNDETERMINED_NO_FROZEN_THRESHOLD",
            "subject_identity": "NOT_MEASURED", "localization_accuracy": "UNDETERMINED_NO_LABELS"}


def decode_video(path, *, sample_hz, max_frames, max_pixels, timeout_seconds):
    """CPU ffmpeg decode; stored orientation, grayscale, no resize, regular fps.

    ffmpeg fps start_time=0 defines sampled timeline; sample indices are output
    indices, never source/truth correspondences. Exceeding caps is explicit.
    """
    if (not math.isfinite(sample_hz) or sample_hz <= 0 or type(max_frames) is not int
            or max_frames < 2 or type(max_pixels) is not int or max_pixels < 4
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("invalid explicit decode budget")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation", "-of", "json", str(path)],
                           capture_output=True, check=True, timeout=timeout_seconds)
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    rotations = [stream.get("tags", {}).get("rotate", 0)] + [s.get("rotation", 0) for s in stream.get("side_data_list", [])]
    if any(float(rotation) % 360 != 0 for rotation in rotations):
        raise ValueError("ROTATED_INPUT_REQUIRES_EXPLICIT_PROTOCOL_DECISION")
    if width < 2 or height < 2 or width * height > max_pixels:
        raise ValueError("DECODE_PIXEL_BUDGET_OR_GEOMETRY")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path),
                              "-map", "0:v:0", "-vf", f"fps=fps={sample_hz}:start_time=0",
                              "-frames:v", str(max_frames + 1), "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                             capture_output=True, check=True, timeout=timeout_seconds)
    size = width * height
    if len(decoded.stdout) % size:
        raise ValueError("PARTIAL_DECODE_FRAME")
    count = len(decoded.stdout) // size
    if count > max_frames:
        raise ValueError("DECODE_FRAME_BUDGET")
    if count < 2:
        raise ValueError("INSUFFICIENT_DECODE_FRAMES")
    return width, height, tuple(decoded.stdout[i * size:(i + 1) * size] for i in range(count))


def repeat_error(first, second):
    if len(first) != len(second) or any((a.sample_index, a.time_seconds, a.valid, a.reason) !=
                                      (b.sample_index, b.time_seconds, b.valid, b.reason)
                                      for a, b in zip(first, second)):
        return {"structure_equal": False, "max_q_error": None}
    errors = [abs(x - y) for a, b in zip(first, second) if a.valid for x, y in zip(a.q, b.q)]
    return {"structure_equal": True, "max_q_error": max(errors) if errors else None,
            "all_observation_fields_equal": first == second}
