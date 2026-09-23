"""Pure CPU tests of an uncalibrated raw-logit reduction; no model/media run."""
import inspect

import numpy as np
import pytest

from runtime.wan import videoseal_score_diagnostic as diagnostic

pytestmark = pytest.mark.unit


def test_fixed_domain_shake256_bit_order_and_key_separation():
    alpha = diagnostic.key_bits(b"alpha")
    assert alpha.shape == (256,) and set(alpha.tolist()) == {0, 1}
    assert np.packbits(alpha, bitorder="big").tobytes().hex() == (
        "d0040ff2cee05a10ccc88f7286b2dcec7fa1d471badfb13436aa070b7d323cf6"
    )
    assert not np.array_equal(alpha, diagnostic.key_bits(b"beta"))
    assert diagnostic.key_identifier(b"alpha") != diagnostic.key_identifier(b"beta")


def test_manual_score_uses_every_frame_and_spatial_point_and_omits_channel_zero():
    logits = np.zeros((181, 257, 2, 3), dtype=np.float32)
    logits[:, 0] = 100.0
    logits[:, 1] = .5
    logits[180, 1, 1, 2] = 6.5
    logits[:, 2] = -1.0
    signs = diagnostic.key_bits(b"alpha")[:2].astype(np.float64) * 2 - 1
    expected = (signs[0] * (.5 + 6.0 / (181 * 2 * 3)) + signs[1] * -1.0) / 256
    row = diagnostic.reduce_raw_detect_logits(logits, b"alpha")
    assert row["status"] == "SCORED"
    assert row["score"] == pytest.approx(expected, abs=1e-14)
    assert row["detection_logit_mean"] == 100.0
    assert row["detection_channel_in_score"] is False
    assert row["decision"] is None and row["calibration_status"] == "UNCALIBRATED"
    assert row["spec_sha256"] is None and row["checkpoint_sha256"] is None
    logits[:, 0] = -1234.0
    changed = diagnostic.reduce_raw_detect_logits(logits, b"alpha")
    assert changed["score"] == row["score"]
    assert changed["detection_logit_mean"] == -1234.0
    assert diagnostic.reduce_raw_detect_logits(logits, b"beta")["score"] != row["score"]


def test_full_181_frame_mean_and_zero_logit_score():
    logits = np.zeros((181, 257, 1, 1), dtype=np.float32)
    assert diagnostic.reduce_raw_detect_logits(logits, b"alpha")["score"] == 0.0
    logits[:, 1, 0, 0] = np.arange(181, dtype=np.float32)
    sign = int(diagnostic.key_bits(b"alpha")[0]) * 2 - 1
    assert diagnostic.reduce_raw_detect_logits(logits, b"alpha")["score"] == pytest.approx(sign * 90 / 256)


@pytest.mark.parametrize("shape", [(180, 257, 2, 2), (182, 257, 2, 2), (181, 256, 2, 2),
    (181, 258, 2, 2), (181, 257), (1, 256), (181, 257, 0, 2), (181, 257, 2, 0)])
def test_wrong_or_aggregated_shapes_are_invalid(shape):
    row = diagnostic.reduce_raw_detect_logits(np.zeros(shape, dtype=np.float32), b"alpha")
    assert row["status"] == "INVALID" and row["reason"] == "RAW_PER_FRAME_SHAPE_REQUIRED"
    assert row["score"] is None and row["decision"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_are_invalid(bad):
    logits = np.zeros((181, 257, 1, 1), dtype=np.float32)
    logits[179, 17, 0, 0] = bad
    row = diagnostic.reduce_raw_detect_logits(logits, b"alpha")
    assert row["status"] == "INVALID" and row["reason"] == "NONFINITE_LOGITS"


def test_blind_interface_rejects_wrong_types_and_has_no_truth_off_or_arm():
    assert tuple(inspect.signature(diagnostic.reduce_raw_detect_logits).parameters) == ("logits", "key")
    assert diagnostic.reduce_raw_detect_logits([[0]], b"alpha")["reason"] == "CPU_NUMPY_ARRAY_REQUIRED"
    assert diagnostic.reduce_raw_detect_logits(np.zeros((181, 257, 1, 1), dtype=np.int16), b"alpha")["reason"] == "FLOAT_LOGITS_REQUIRED"
    assert diagnostic.reduce_raw_detect_logits(np.zeros((181, 257, 1, 1), dtype=np.float32), b"")["reason"] == "INVALID_KEY"
    assert diagnostic.reduce_raw_detect_logits(np.zeros((181, 257, 1, 1), dtype=np.float32), "alpha")["reason"] == "INVALID_KEY"
