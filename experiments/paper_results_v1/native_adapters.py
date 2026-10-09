"""Native-output adapters over explicitly supplied, already-loaded backends.

These adapters do not load weights, select devices, download source, choose a
capacity mapping, or turn frame/spatial outputs into a sequence decision.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping


class NativeAdapterError(ValueError):
    """The declared native interface cannot be used without changing semantics."""


def _bits(value, expected_length):
    if not isinstance(value, list) or len(value) != expected_length:
        raise NativeAdapterError(f"native_message_bits must contain exactly {expected_length} bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise NativeAdapterError("native_message_bits must contain integer 0/1 values")
    return list(value)


def _backend_metadata(value, required, method):
    if not isinstance(value, Mapping):
        raise NativeAdapterError(f"{method} backend_metadata must be an object")
    missing = [field for field in required if not isinstance(value.get(field), str) or not value[field].strip()]
    if missing:
        raise NativeAdapterError(f"{method} backend_metadata missing explicit fields: {', '.join(missing)}")
    return _json_value(dict(value))


def _json_value(value):
    """Losslessly retain small native outputs without importing tensor packages."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NativeAdapterError("native output contains a non-finite value")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    candidate = value
    for method in ("detach", "cpu"):
        function = getattr(candidate, method, None)
        if callable(function):
            candidate = function()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    item = getattr(candidate, "item", None)
    if callable(item):
        return _json_value(item())
    raise NativeAdapterError(f"native output type is not losslessly serializable: {type(value).__name__}")


def _shape(value):
    declared = getattr(value, "shape", None)
    if declared is not None:
        try:
            return [int(part) for part in declared]
        except (TypeError, ValueError):
            pass
    value = _json_value(value)
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    child = _shape(value[0])
    if any(_shape(item) != child for item in value[1:]):
        raise NativeAdapterError("native output is ragged and has no single reviewable shape")
    return [len(value), *child]


def _output_shapes(value):
    if isinstance(value, Mapping):
        return {str(key): _output_shapes(item) for key, item in value.items()}
    return _shape(value)


def _output_counts(value):
    if isinstance(value, Mapping):
        return {str(key): _output_counts(item) for key, item in value.items()}
    value = _json_value(value)
    if isinstance(value, list):
        return sum(_output_counts(item) for item in value)
    return 1


class VideoSealLoadedBackend:
    """Thin call-through for a supplied VideoSeal model and message builder.

    The model is expected to expose the documented ``embed`` and ``detect``
    calls. The caller owns tensor construction and declares the backend version.
    """

    def __init__(self, model, message_builder, *, lowres_attenuation=False):
        if not callable(message_builder):
            raise TypeError("message_builder must be callable")
        self.model = model
        self.message_builder = message_builder
        self.lowres_attenuation = bool(lowres_attenuation)

    def embed(self, media, message_bits, output_uri=None):
        del output_uri
        return self.model.embed(
            media,
            msgs=self.message_builder(list(message_bits)),
            is_video=True,
            lowres_attenuation=self.lowres_attenuation,
        )

    def embedded_media(self, output):
        if not isinstance(output, Mapping) or "imgs_w" not in output:
            raise NativeAdapterError("VideoSeal embed output lacks imgs_w")
        return output["imgs_w"]

    def extract(self, media):
        return self.model.detect(media, is_video=True)


class RivaGANPathBackend:
    """Explicit wrapper around upstream path encode/decode.

    Upstream ``encode`` writes OpenCV mp4v at 20 fps. The transport metadata is
    deliberately exposed so this path cannot masquerade as the shared codec.
    """

    transport_metadata = {
        "transport_mode": "PATH_API_NATIVE_MP4V_20FPS",
        "native_path_codec": "opencv_mp4v",
        "native_path_fps": 20,
        "color_layout": "BGR",
        "decoder_normalization": "value/127.5-1",
        "codec_comparability": "NATIVE_PATH_CODEC_NOT_SHARED_PLANNED_CODEC",
    }

    def __init__(self, model):
        self.model = model

    def embed(self, media, message_bits, output_uri=None):
        if not output_uri:
            raise NativeAdapterError("RivaGAN path backend requires explicit output_uri")
        self.model.encode(str(media), tuple(message_bits), str(output_uri))
        return str(output_uri)

    def embedded_media(self, output):
        return output

    def extract(self, media):
        return self.model.decode(str(media))


class CallableTensorBackend:
    """Backend for an explicitly supplied tensor-level embed/extract pair."""

    def __init__(self, embed_call, extract_call, *, transport_metadata):
        if not callable(embed_call) or not callable(extract_call):
            raise TypeError("embed_call and extract_call must be callable")
        self.embed_call = embed_call
        self.extract_call = extract_call
        self.transport_metadata = dict(transport_metadata)

    def embed(self, media, message_bits, output_uri=None):
        return self.embed_call(media, tuple(message_bits), output_uri=output_uri)

    def embedded_media(self, output):
        return output

    def extract(self, media):
        return self.extract_call(media)


class VideoSealNativeAdapter:
    method = "videoseal"

    def __init__(self, backend, *, native_message_length, backend_metadata):
        if type(native_message_length) is not int or native_message_length <= 0:
            raise NativeAdapterError("VideoSeal native_message_length must be a positive integer")
        self.backend = backend
        self.native_message_length = native_message_length
        self.backend_metadata = _backend_metadata(
            backend_metadata,
            ("source_version", "model_version", "weight_identity", "detect_output_layout"),
            "VideoSeal",
        )

    def embed(self, media, message_bits, *, output_uri=None):
        message = _bits(message_bits, self.native_message_length)
        started = time.perf_counter()
        embed_output = self.backend.embed(media, message, output_uri=output_uri)
        embedded = self.backend.embedded_media(embed_output)
        embed_seconds = time.perf_counter() - started
        return embedded, {
            "api": "embed(media,msgs=message_builder(bits),is_video=True,lowres_attenuation=declared)",
            "native_message": {"length_bits": len(message), "bits": message},
            "raw_native_output": _json_value(embed_output),
            "output_shapes": _output_shapes(embed_output),
            "output_element_counts": _output_counts(embed_output),
            "seconds": embed_seconds,
        }

    def extract(self, media):
        started = time.perf_counter()
        extraction = self.backend.extract(media)
        extract_seconds = time.perf_counter() - started
        return {
            "api": "detect(media,is_video=True)",
            "truth_input_supplied": False,
            "raw_native_output": _json_value(extraction),
            "output_shapes": _output_shapes(extraction),
            "output_element_counts": _output_counts(extraction),
            "spatial_or_temporal_reducer": "PENDING_NOT_APPLIED",
            "seconds": extract_seconds,
            "main_32bit_mapping": "PENDING_NOT_APPLIED",
        }

    def run(self, media, message_bits, *, output_uri=None):
        embedded, embed = self.embed(media, message_bits, output_uri=output_uri)
        extract = self.extract(embedded)
        return embedded, {
            "method": self.method, "status": "SUCCEEDED", "backend_metadata": self.backend_metadata,
            "native_message": embed["native_message"], "embed": embed, "extract": extract,
            "main_32bit_mapping": "PENDING_NOT_APPLIED",
        }


class RivaGANNativeAdapter:
    method = "rivagan"

    def __init__(self, backend, *, backend_metadata):
        self.backend = backend
        self.backend_metadata = dict(backend_metadata)
        transport = getattr(backend, "transport_metadata", None)
        if transport is not None:
            self.backend_metadata["transport"] = dict(transport)
        self.backend_metadata = _backend_metadata(
            self.backend_metadata,
            ("source_version", "model_version", "weight_identity"),
            "RivaGAN",
        )
        transport = self.backend_metadata.get("transport")
        required_transport = ("transport_mode", "color_layout", "decoder_normalization", "codec_comparability")
        if not isinstance(transport, Mapping) or any(
            not isinstance(transport.get(field), str) or not transport[field].strip()
            for field in required_transport
        ):
            raise NativeAdapterError("RivaGAN transport metadata must explicitly declare mode, color, normalization, and codec comparability")

    def embed(self, media, message_bits, *, output_uri=None):
        message = _bits(message_bits, 32)
        started = time.perf_counter()
        embed_output = self.backend.embed(media, message, output_uri=output_uri)
        embedded = self.backend.embedded_media(embed_output)
        embed_seconds = time.perf_counter() - started
        return embedded, {
            "api": "explicit_backend.embed(media,tuple(bits),output_uri=declared)",
            "native_message": {"length_bits": 32, "bits": message},
            "raw_native_output": _json_value(embed_output),
            "output_shapes": _output_shapes(embed_output),
            "output_element_counts": _output_counts(embed_output),
            "seconds": embed_seconds,
        }

    def extract(self, media):
        started = time.perf_counter()
        decoded = self.backend.extract(media)
        if not isinstance(decoded, Iterable):
            raise NativeAdapterError("RivaGAN decode output must be a frame iterable")
        frame_soft = []
        frame_bits = []
        frame_shapes = []
        for frame in decoded:
            values = _json_value(frame)
            shape = _shape(frame)
            if len(shape) != 1:
                raise NativeAdapterError("RivaGAN frame logits must have one native bit dimension")
            frame_soft.append(values)
            frame_shapes.append(shape)
            frame_bits.append([int(value >= 0.0) for value in values])
        if not frame_soft:
            raise NativeAdapterError("RivaGAN decode returned no frames")
        extract_seconds = time.perf_counter() - started
        return {
            "api": "explicit_backend.extract(media)",
            "truth_input_supplied": False,
            "frame_soft_outputs": frame_soft,
            "frame_native_zero_threshold_bits": frame_bits,
            "frame_shapes": frame_shapes,
            "decoded_frame_count": len(frame_soft),
            "soft_value_count": sum(len(frame) for frame in frame_soft),
            "seconds": extract_seconds,
            "sequence_32bit_recovery": "PENDING_REDUCER_NOT_APPLIED",
            "per_frame_accuracy": "NOT_COMPUTED_WITHOUT_TRUTH_EVALUATION",
        }

    def run(self, media, message_bits, *, output_uri=None):
        embedded, embed = self.embed(media, message_bits, output_uri=output_uri)
        extract = self.extract(embedded)
        return embedded, {
            "method": self.method, "status": "SUCCEEDED", "backend_metadata": self.backend_metadata,
            "native_message": embed["native_message"], "embed": embed, "extract": extract,
            "sequence_32bit_recovery": "PENDING_REDUCER_NOT_APPLIED",
            "per_frame_accuracy": "NOT_COMPUTED_WITHOUT_TRUTH_EVALUATION",
        }
