import pytest

from main.sc_sstw.aisb import make_double_redundant_templates
from experiments.feasibility.synthetic.channel import SyntheticChannel, generate_observations
from main.sc_sstw.calibration import (
    calibrate_from_pilot_pairs,
    equalize_observations,
)


@pytest.mark.unit
def test_double_redundant_template_supports_two_missing_points():
    templates = make_double_redundant_templates()
    assert len(templates) == 3
    assert all(len(template.points) == 12 for template in templates)
    assert all(
        template.points[0] == template.points[6] == template.points[9]
        and template.points[1] == template.points[7] == template.points[10]
        and template.points[2] == template.points[8] == template.points[11]
        for template in templates
    )


@pytest.mark.quick
def test_public_only_affine_calibration_recovers_constructed_channel():
    expected = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.4, 0.7)]
    channel = SyntheticChannel(
        matrix=((1.2, -0.3), (0.25, 0.8)),
        bias=(0.1, -0.2),
        noise_std=0.0,
    )
    observed = generate_observations(expected, channel, seed=7)
    fit = calibrate_from_pilot_pairs(list(zip(expected, observed, strict=True)))
    assert fit.condition_number < 10
    assert fit.pilot_reconstruction_mse == pytest.approx(0.0, abs=1e-12)
    equalized = equalize_observations(observed, fit, ridge=0.0)
    assert equalized == pytest.approx(expected, abs=1e-6)
