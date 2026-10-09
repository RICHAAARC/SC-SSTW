"""Deterministic CPU fixtures that exercise the real workflow call path."""
from __future__ import annotations

from experiments.paper_results_v1.native_adapters import (
    CallableTensorBackend,
    RivaGANNativeAdapter,
    VideoSealLoadedBackend,
    VideoSealNativeAdapter,
)


def _map_numeric(value, function):
    if isinstance(value, (list, tuple)):
        return [_map_numeric(item, function) for item in value]
    return function(float(value))


def _offset(amount):
    def operation(value, context):
        del context
        return _map_numeric(value, lambda item: item + amount)
    return operation


def _fixture_codec(value, context):
    precision = int(context["codec"]["parameters"]["decimal_places"])
    return _map_numeric(value, lambda item: round(item, precision))


def build_fixture_operations():
    return {
        "fixture_off_native": _offset(0.0),
        "fixture_payload_native": _offset(0.01),
        "fixture_framewise_recon": _offset(0.005),
        "fixture_framewise_m05": _offset(0.002),
        "fixture_codec_roundtrip": _fixture_codec,
    }


class _FixtureVideoSealModel:
    def embed(self, media, *, msgs, is_video, lowres_attenuation):
        assert is_video is True
        assert lowres_attenuation is False
        bits = list(msgs[0])
        return {
            "imgs_w": _map_numeric(media, lambda item: item + 0.003),
            "msgs": [bits],
        }

    def detect(self, media, *, is_video):
        assert is_video is True
        base = sum(_flatten(media)) / len(_flatten(media))
        # T, 1+K, H, W. The adapter retains every value and applies no reducer.
        return {
            "preds": [
                [[[base]], [[-0.8]], [[0.7]], [[0.6]], [[-0.5]]],
                [[[base + 0.1]], [[-0.7]], [[0.8]], [[0.5]], [[-0.4]]],
            ]
        }


def _flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _rivagan_embed(media, message_bits, *, output_uri=None):
    del message_bits, output_uri
    return _map_numeric(media, lambda item: item + 0.004)


def _rivagan_extract(media):
    del media
    message = [0, 1] * 16
    return [
        [-0.9 if bit == 0 else 0.9 for bit in message],
        [-0.7 if bit == 0 else 0.7 for bit in message],
    ]


def build_fixture_native_adapters():
    videoseal_backend = VideoSealLoadedBackend(
        _FixtureVideoSealModel(), message_builder=lambda bits: [list(bits)], lowres_attenuation=False,
    )
    rivagan_backend = CallableTensorBackend(
        _rivagan_embed,
        _rivagan_extract,
        transport_metadata={
            "transport_mode": "TENSOR_FIXTURE_NO_HIDDEN_CODEC",
            "color_layout": "SYNTHETIC_SCALAR_ARRAY",
            "decoder_normalization": "NONE_SYNTHETIC_FIXTURE",
            "codec_comparability": "SHARED_FIXTURE_CODEC_APPLIED_BY_WORKFLOW",
        },
    )
    return {
        "videoseal": VideoSealNativeAdapter(
            videoseal_backend,
            native_message_length=4,
            backend_metadata={
                "evidence_role": "SYNTHETIC_FIXTURE_ONLY",
                "source_version": "current-main-interface-reference; historical-e00b-pin-not-reverified",
                "model_version": "synthetic_fixture_backend",
                "weight_identity": "synthetic_fixture_no_weight",
                "detect_output_layout": "T,1+K,H,W",
            },
        ),
        "rivagan": RivaGANNativeAdapter(
            rivagan_backend,
            backend_metadata={
                "evidence_role": "SYNTHETIC_FIXTURE_ONLY",
                "source_version": "efffa72a4ca46d4d5051f6970c96424c2cdab441",
                "model_version": "synthetic_fixture_backend",
                "weight_identity": "synthetic_fixture_no_checkpoint",
            },
        ),
    }
