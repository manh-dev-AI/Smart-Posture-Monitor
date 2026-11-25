import numpy as np
import pytest

from src.personal_calibration import PersonalCalibration
from src.representation_builder import RepresentationBuilder


def _sample(value: float) -> np.ndarray:
    return np.full(PersonalCalibration.FEATURE_DIM, value, dtype=float)


def test_baseline_median_dung() -> None:
    calibration = PersonalCalibration(target_samples=3)
    assert calibration.add_sample(_sample(1.0))
    assert calibration.add_sample(_sample(3.0))
    assert calibration.add_sample(_sample(5.0))

    baseline = calibration.baseline

    assert calibration.is_ready
    assert calibration.sample_count == 3
    np.testing.assert_allclose(
        baseline,
        _sample(3.0),
    )


def test_representation_builder_transform() -> None:
    builder = RepresentationBuilder()
    baseline = _sample(2.0)
    # Ensure baseline shoulder width (index 11) is positive for normalization
    baseline[PersonalCalibration.SHOULDER_WIDTH_INDEX] = 100.0

    current = _sample(2.0)
    current[PersonalCalibration.SHOULDER_WIDTH_INDEX] = 100.0

    rep = builder.transform(current, baseline)

    assert rep.shape == (builder.output_feature_count,)
    assert rep.dtype == np.float32
    # At identical values, deltas are 0, log-ratios are 0
    np.testing.assert_allclose(rep, np.zeros(builder.output_feature_count), atol=1e-5)


def test_reset_hoat_dong() -> None:
    calibration = PersonalCalibration(target_samples=1)
    calibration.add_sample(_sample(1.0))

    calibration.reset()

    assert calibration.sample_count == 0
    assert not calibration.is_ready
    assert calibration.baseline is None


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([np.nan] * PersonalCalibration.FEATURE_DIM),
        np.array([np.inf] * PersonalCalibration.FEATURE_DIM),
        np.array([-np.inf] * PersonalCalibration.FEATURE_DIM),
    ],
)
def test_nan_inf_bi_reject(bad_vector: np.ndarray) -> None:
    calibration = PersonalCalibration(target_samples=1)

    assert not calibration.add_sample(bad_vector)
    assert calibration.sample_count == 0


def test_vector_sai_dimension_bi_reject() -> None:
    calibration = PersonalCalibration(target_samples=1)

    assert not calibration.add_sample(np.array([1.0, 2.0]))
    assert calibration.sample_count == 0


def test_fit_offline_dung() -> None:
    calibration = PersonalCalibration(target_samples=3)
    correct_samples = np.vstack(
        [
            _sample(1.0),
            _sample(3.0),
            _sample(5.0),
        ]
    )

    baseline = calibration.fit(correct_samples)

    assert calibration.is_ready
    np.testing.assert_allclose(baseline, _sample(3.0))

