"""Native-output adapters over explicitly supplied, already-loaded backends.

These adapters do not load weights, select devices, download source, choose a
capacity mapping, or turn frame/spatial outputs into a sequence decision.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path


class NativeAdapterError(ValueError):
    """The declared native interface cannot be used without changing semantics."""


def _bits(value, expected_length):
    if not isinstance(value, list) or len(value) != expected_length:
        raise NativeAdapterError(f"native_message_bits must contain exactly {expected_length} bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise NativeAdapterError("native_message_bits must contain integer 0/1 values")
    return list(value)


def _backend_metadata(value, required, method, *, optional=()):
    if not isinstance(value, Mapping):
        raise NativeAdapterError(f"{method} backend_metadata must be an object")
    missing = [field for field in required if not isinstance(value.get(field), str) or not value[field].strip()]
    if missing:
        raise NativeAdapterError(f"{method} backend_metadata missing explicit fields: {', '.join(missing)}")
    result = dict(value)
    for field in optional:
        result.setdefault(field, None)
    return _json_value(result)


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
    declared = getattr(value, "shape", None)
    if declared is not None:
        try:
            return math.prod(int(part) for part in declared)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        return sum(_output_counts(item) for item in value)
    _json_value(value)
    return 1


def _count_total(value):
    if isinstance(value, Mapping):
        return sum(_count_total(item) for item in value.values())
    return int(value)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_descriptor(value, backend=None):
    shape = _shape(value)
    count = math.prod(shape) if shape else None
    dtype = getattr(value, "dtype", None)
    descriptor = {
        "storage": "ARTIFACT_VALUE_NOT_INLINED",
        "shape": shape,
        "element_count": count,
        "dtype": str(dtype) if dtype is not None else type(value).__name__,
        "identity_status": "NOT_PROVIDED_BY_BACKEND",
    }
    if isinstance(value, (str, Path)):
        path = Path(value)
        descriptor["uri"] = str(value)
        if path.is_file():
            descriptor.update(identity_status="FILE_SHA256", sha256=_file_sha256(path), bytes=path.stat().st_size)
    identity = getattr(backend, "media_identity", None)
    if callable(identity):
        supplied = identity(value)
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise NativeAdapterError("backend media_identity must return an object or null")
            descriptor.update(identity_status="BACKEND_PROVIDED", identity=_json_value(supplied))
    return descriptor


def _preserve_native_output(value, *, label, store, inline_element_limit):
    shapes = _output_shapes(value)
    counts = _output_counts(value)
    total = _count_total(counts)
    base = {"output_shapes": shapes, "output_element_counts": counts, "total_element_count": total}
    if total <= inline_element_limit:
        return {**base, "storage": "INLINE_FULL", "raw_native_output": _json_value(value)}
    if store is None:
        raise NativeAdapterError(
            f"{label} has {total} elements; an explicit lossless native_output_store is required"
        )
    reference = store(label, value, {"shapes": shapes, "counts": counts, "total_element_count": total})
    if not isinstance(reference, Mapping) or reference.get("lossless") is not True:
        raise NativeAdapterError("native_output_store must return an object with lossless=true")
    for field in ("uri", "format"):
        if not isinstance(reference.get(field), str) or not reference[field]:
            raise NativeAdapterError(f"native_output_store reference must include {field}")
    return {**base, "storage": "LOSSLESS_SIDECAR", "lossless_native_output": _json_value(reference)}


def _flat_message(value):
    value = _json_value(value)
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flat_message(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
        raise NativeAdapterError("VideoSeal message builder produced a non-bit value")
    return [int(value)]


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

    def build_message(self, message_bits):
        return self.message_builder(list(message_bits))

    def embed(self, media, prepared_message, output_uri=None):
        del output_uri
        return self.model.embed(
            media,
            msgs=prepared_message,
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


class RivaGANLoadedTensorBackend:
    """Pinned-source tensor wiring without RivaGAN's hidden mp4v path codec.

    ``tensor_module`` is an explicitly supplied torch-compatible module and
    ``array_module`` is NumPy-compatible. The loaded model must expose
    ``encoder`` and ``decoder`` callables.
    """

    def __init__(self, model, *, tensor_module, array_module, device, input_color_layout="BGR_UINT8"):
        if input_color_layout not in ("BGR_UINT8", "RGB_UINT8_TO_BGR"):
            raise NativeAdapterError("RivaGAN tensor input_color_layout must be BGR_UINT8 or RGB_UINT8_TO_BGR")
        self.model = model
        self.tensor_module = tensor_module
        self.array_module = array_module
        self.device = device
        self.input_color_layout = input_color_layout
        self.transport_metadata = {
            "transport_mode": "LOADED_ENCODER_DECODER_TENSOR_NO_CODEC",
            "color_layout": input_color_layout,
            "decoder_normalization": "uint8/127.5-1",
            "codec_comparability": "NO_HIDDEN_CODEC_SHARED_CODEC_REQUIRED_AFTER_EMBED",
        }

    def _frames(self, media):
        shape = getattr(media, "shape", None)
        if shape is None or len(shape) != 4 or int(shape[-1]) != 3:
            raise NativeAdapterError("RivaGAN tensor media must have shape [T,H,W,3]")
        dtype = str(getattr(media, "dtype", "")).lower()
        if "uint8" not in dtype:
            raise NativeAdapterError("RivaGAN tensor media must be uint8")
        return media

    def _bgr_frame(self, frame):
        if self.input_color_layout == "RGB_UINT8_TO_BGR":
            return frame[..., ::-1].copy()
        return frame

    def _model_frame(self, frame):
        tensor = self.tensor_module.as_tensor(
            self._bgr_frame(frame), dtype=self.tensor_module.float32, device=self.device,
        )
        return (tensor / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)

    def embed(self, media, message_bits, output_uri=None):
        del output_uri
        frames = self._frames(media)
        message = _bits(list(message_bits), 32)
        data = self.tensor_module.as_tensor(
            [message], dtype=self.tensor_module.float32, device=self.device,
        )
        if [int(part) for part in data.shape] != [1, 32]:
            raise NativeAdapterError("RivaGAN message tensor must have shape [1,32]")
        encoded_frames = []
        for frame in frames:
            encoded = self.model.encoder(self._model_frame(frame), data).clamp(-1.0, 1.0)
            output = encoded[0, :, 0, :, :].permute(1, 2, 0)
            output = ((output + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")
            if self.input_color_layout == "RGB_UINT8_TO_BGR":
                output = output[..., ::-1].copy()
            encoded_frames.append(output)
        return self.array_module.stack(encoded_frames, axis=0)

    def embedded_media(self, output):
        return output

    def extract(self, media):
        frames = self._frames(media)
        for frame in frames:
            logits = self.model.decoder(self._model_frame(frame))[0]
            yield logits.detach().cpu().numpy()


class VideoSealNativeAdapter:
    method = "videoseal"

    def __init__(
        self,
        backend,
        *,
        native_message_length,
        backend_metadata,
        native_output_store=None,
        inline_element_limit=1_000_000,
    ):
        if type(native_message_length) is not int or native_message_length <= 0:
            raise NativeAdapterError("VideoSeal native_message_length must be a positive integer")
        self.backend = backend
        self.native_message_length = native_message_length
        self.backend_metadata = _backend_metadata(
            backend_metadata,
            ("model_version", "detect_output_layout"),
            "VideoSeal",
            optional=("source_version", "weight_identity"),
        )
        if native_output_store is not None and not callable(native_output_store):
            raise NativeAdapterError("native_output_store must be callable")
        if type(inline_element_limit) is not int or inline_element_limit <= 0:
            raise NativeAdapterError("inline_element_limit must be a positive integer")
        self.native_output_store = native_output_store
        self.inline_element_limit = inline_element_limit

    def embed(self, media, message_bits, *, output_uri=None):
        message = _bits(message_bits, self.native_message_length)
        prepared_message = self.backend.build_message(message)
        submitted_shape = _shape(prepared_message)
        submitted_bits = _flat_message(prepared_message)
        if submitted_shape != [1, self.native_message_length] or submitted_bits != message:
            raise NativeAdapterError(
                "VideoSeal message builder must preserve declared bits exactly with shape [1,native_message_length]"
            )
        started = time.perf_counter()
        embed_output = self.backend.embed(media, prepared_message, output_uri=output_uri)
        embedded = self.backend.embedded_media(embed_output)
        embed_seconds = time.perf_counter() - started
        non_media = {
            str(key): value for key, value in embed_output.items() if key != "imgs_w"
        } if isinstance(embed_output, Mapping) else {}
        return embedded, {
            "api": "embed(media,msgs=message_builder(bits),is_video=True,lowres_attenuation=declared)",
            "native_message": {"length_bits": len(message), "bits": message},
            "submitted_msgs": {
                "bits": submitted_bits,
                "shape": submitted_shape,
                "element_count": len(submitted_bits),
            },
            "embedded_media": _media_descriptor(embedded, self.backend),
            "non_media_native_output": _json_value(non_media),
            "seconds": embed_seconds,
        }

    def extract(self, media):
        started = time.perf_counter()
        extraction = self.backend.extract(media)
        extract_seconds = time.perf_counter() - started
        preserved = _preserve_native_output(
            extraction,
            label="videoseal_detect_output",
            store=self.native_output_store,
            inline_element_limit=self.inline_element_limit,
        )
        return {
            "api": "detect(media,is_video=True)",
            "truth_input_supplied": False,
            **preserved,
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
            ("model_version",),
            "RivaGAN",
            optional=("source_version", "weight_identity"),
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
            "embedded_media": _media_descriptor(embedded, self.backend),
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
            if shape != [32]:
                raise NativeAdapterError("RivaGAN frame logits must have strict shape [32]")
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
