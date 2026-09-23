"""Directed CPU checks for the isolated RGB/DCT presence candidate."""
import inspect
import math

import numpy as np
import pytest

from main.tube_state import rgb_dct_presence as receiver
from runtime.wan import rgb_dct_presence_adapter as adapter


pytestmark = pytest.mark.unit
SHAPE = (181, 320, 512, 3)


def _constant(value, dtype=np.float32):
    return np.broadcast_to(np.array(value, dtype=dtype).reshape(1, 1, 1, 1), SHAPE)


def test_fixed_shake_vectors_and_balanced_codes():
    # The fixed bytes independently pin the single final NUL in each domain.
    assert receiver._shake_bytes(receiver._SPACE_DOMAIN, b"alpha", 160)[:16].hex() == (
        "2c34831e0a40ebfd92bcda881ccd8724"
    )
    assert receiver._shake_bytes(receiver._TIME_DOMAIN, b"alpha", 30)[:16].hex() == (
        "bdc0514fec965a736c7a891696788f6d"
    )
    assert receiver._SPACE_DOMAIN[-1:] == receiver._TIME_DOMAIN[-1:] == bytes([0])
    space, group, temporal = receiver.key_codes(b"alpha")
    assert space.shape == (160,) and np.count_nonzero(space == 1) == 80
    assert np.count_nonzero(space == -1) == 80
    assert group.shape == (30,) and np.count_nonzero(group == 1) == 15
    assert np.count_nonzero(group == -1) == 15
    assert receiver._GROUPS.tolist() == [g for g in range(29) for _ in range(6)] + [29] * 7
    assert np.mean(temporal) == pytest.approx(0, abs=1e-15)
    assert np.mean(temporal * temporal) == pytest.approx(1, abs=1e-15)
    assert temporal[0] == pytest.approx(-1.0055402085998904, abs=1e-14)
    assert temporal[180] == pytest.approx(0.9944903161976938, abs=1e-14)


def test_orthonormal_dct_orientation_and_difference():
    k = receiver.dct_difference_kernel()
    assert k.shape == (32, 32)
    assert np.sum(k) == pytest.approx(0, abs=1e-14)
    assert np.sum(k * k) == pytest.approx(2, abs=1e-14)
    assert np.max(np.abs(k + k.T)) < 1e-15
    x = np.arange(32, dtype=np.float64)
    b1 = math.sqrt(2 / 32) * np.cos(math.pi * (2 * x + 1) / 64)
    b2 = math.sqrt(2 / 32) * np.cos(math.pi * (2 * x + 1) * 2 / 64)
    vertical_2_horizontal_1 = np.outer(b2, b1)
    vertical_1_horizontal_2 = np.outer(b1, b2)
    assert np.sum(vertical_2_horizontal_1 * k) == pytest.approx(1, abs=1e-14)
    assert np.sum(vertical_1_horizontal_2 * k) == pytest.approx(-1, abs=1e-14)


def test_zero_static_and_spatially_uniform_temporal_changes_cancel():
    zero = receiver.score_rgb(_constant(0.0), b"alpha")
    assert zero["status"] == "SCORED" and zero["score"] == 0.0
    assert zero["frames_used"] == 181 and zero["feature_count"] == 181 * 160
    assert zero["decision"] is None and zero["calibration_status"] == "UNCALIBRATED"
    assert zero["spec_sha256"] and zero["spatial_code_sha256"] and zero["temporal_code_sha256"]
    static = np.broadcast_to(
        np.linspace(0.1, 0.9, 320 * 512 * 3, dtype=np.float32).reshape(1, 320, 512, 3), SHAPE
    )
    assert abs(receiver.score_rgb(static, b"alpha")["score"]) < 1e-10
    levels = np.linspace(0.1, 0.9, 181, dtype=np.float32).reshape(181, 1, 1, 1)
    uniform = np.broadcast_to(levels, SHAPE)
    assert abs(receiver.score_rgb(uniform, b"alpha")["score"]) < 1e-10


def test_spatial_balance_cancels_nonzero_equal_block_features():
    # Repeated non-DC blocks have nonzero d, so cancellation needs all 80/80 signs.
    pattern = np.tile(receiver.dct_difference_kernel(), (10, 16))
    rgb = np.empty(SHAPE, dtype=np.float32)
    for t, amplitude in enumerate(np.linspace(0.05, 0.2, 181)):
        rgb[t] = (0.5 + amplitude * pattern)[:, :, None]
    row = receiver.score_rgb(rgb, b"alpha")
    assert row["status"] == "SCORED"
    assert abs(row["score"]) < 1e-10


def test_aligned_rgb_pattern_has_predicted_sign_and_amplitude():
    spatial, _, temporal = receiver.key_codes(b"alpha")
    pattern = (spatial.reshape(10, 16, 1, 1) * receiver.dct_difference_kernel())
    pattern = pattern.transpose(0, 2, 1, 3).reshape(320, 512)
    amplitude = 0.1
    rgb = np.empty(SHAPE, dtype=np.float32)
    for t in range(181):
        rgb[t] = (0.5 + amplitude * temporal[t] * pattern)[:, :, None]
    positive = receiver.score_rgb(rgb, b"alpha")
    expected = 2 * amplitude / math.sqrt(4 * amplitude**2 + (1 / 255) ** 2)
    assert positive["status"] == "SCORED"
    assert positive["score"] == pytest.approx(expected, abs=2e-6)
    rgb[:] = 1.0 - rgb
    negative = receiver.score_rgb(rgb, b"alpha")
    assert negative["score"] == pytest.approx(-expected, abs=2e-6)


def test_first_and_last_frames_contribute_with_seven_frame_final_group():
    spatial, _, temporal = receiver.key_codes(b"alpha")
    block = receiver.dct_difference_kernel()
    rgb = np.full(SHAPE, 0.5, dtype=np.float32)
    rgb[0, :32, :32, :] = (0.5 + 0.1 * block)[:, :, None]
    rgb[180, :32, :32, :] = (0.5 + 0.2 * block)[:, :, None]
    row = receiver.score_rgb(rgb, b"alpha")
    numerator = spatial[0] * 2 * (0.1 * temporal[0] + 0.2 * temporal[180]) / (181 * 160)
    energy = (2 * 0.1) ** 2 + (2 * 0.2) ** 2
    expected = numerator / math.sqrt(energy / (181 * 160) + (1 / 255) ** 2)
    assert row["score"] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_invalid_value_is_retained_as_invalid(bad):
    row = receiver.score_rgb(_constant(bad), b"alpha")
    assert row["status"] == "INVALID" and row["score"] is None
    assert row["reason"] == ("NONFINITE_RGB" if not np.isfinite(bad) else "RGB_OUT_OF_RANGE")


def test_invalid_shape_dtype_key_and_blind_signature():
    assert tuple(inspect.signature(receiver.score_rgb).parameters) == ("rgb", "key")
    assert receiver.score_rgb(None, b"alpha")["reason"] == "CPU_NUMPY_ARRAY_REQUIRED"
    assert receiver.score_rgb(np.empty((180, 320, 512, 3), dtype=np.float32), b"alpha")["reason"] == "FULL_RGB_SHAPE_REQUIRED"
    assert receiver.score_rgb(_constant(0, np.uint8), b"alpha")["reason"] == "FLOAT32_OR_FLOAT64_REQUIRED"
    assert receiver.score_rgb(_constant(0, np.float16), b"alpha")["reason"] == "FLOAT32_OR_FLOAT64_REQUIRED"
    assert receiver.score_rgb(_constant(0), b"")["reason"] == "INVALID_KEY"
    assert receiver.score_rgb(_constant(0), "alpha")["reason"] == "INVALID_KEY"
    assert receiver.score_rgb(_constant(0, np.float64), b"alpha")["status"] == "SCORED"


def test_mp4_adapter_reuses_shared_reader_and_reports_decode_failure(monkeypatch):
    assert tuple(inspect.signature(adapter.score_mp4).parameters) == ("path", "key")
    paths = []

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return _constant(0.0)

    def fake_read(path):
        paths.append(path)
        return FakeTensor()

    monkeypatch.setattr(adapter, "read_mp4", fake_read)
    row = adapter.score_mp4("sample.mp4", b"alpha")
    assert row["status"] == "SCORED" and row["score"] == 0.0
    assert str(paths[0]) == "sample.mp4"
    monkeypatch.setattr(adapter, "read_mp4", lambda path: (_ for _ in ()).throw(OSError("decode")))
    failed = adapter.score_mp4("sample.mp4", b"alpha")
    assert failed["status"] == "INVALID" and failed["reason"] == "MP4_READ_FAILED"
    assert failed["score"] is None and failed["decision"] is None
