import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from src.acquisition import expected_improvement, probability_of_improvement, upper_confidence_bound
from src.validation import (
    acquisition_values,
    cartesian_resolution_sensitivity,
    exhaustive_kappa_sensitivity_4d,
    fixed_length_scale_validation,
    kernel_choice_validation,
    length_scale_cv_sensitivity,
    loocv_predictions,
    random_resolution_sensitivity,
    sequential_performance,
)


class DeterministicGP:
    def predict(self, candidates, return_std=True):
        candidates = np.asarray(candidates)
        mean = candidates.sum(axis=1)
        std = np.full(len(candidates), 0.2)
        return (mean, std) if return_std else mean


def small_regression_data():
    inputs = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    outputs = np.sin(inputs[:, 0] * np.pi)
    return inputs, outputs


def fixed_rbf(length_scale=0.25, bounds="fixed"):
    return RBF(length_scale=length_scale, length_scale_bounds=bounds)


def test_sequential_performance_tracks_incumbent_and_improvements():
    performance, summary = sequential_performance(
        initial_outputs=np.array([1.0, 2.0, 1.5]),
        query_outputs=np.array([1.8, 2.5, 2.4, 3.0]),
    )

    assert_allclose(performance["best_so_far"], [2.0, 2.5, 2.5, 3.0])
    assert_allclose(performance["improvement"], [0.0, 0.5, 0.0, 0.5])
    metrics = summary.set_index("metric")["value"]
    assert metrics["Initial incumbent"] == pytest.approx(2.0)
    assert metrics["Final best observed"] == pytest.approx(3.0)
    assert metrics["Queries improving incumbent"] == 2


def test_loocv_predictions_returns_one_prediction_per_observation():
    inputs, outputs = small_regression_data()

    predictions = loocv_predictions(
        inputs,
        outputs,
        fixed_rbf(),
        n_restarts_optimizer=0,
        random_state=0,
    )

    assert predictions.shape == outputs.shape
    assert np.all(np.isfinite(predictions))


def test_kernel_choice_validation_returns_expected_columns():
    inputs, outputs = small_regression_data()

    result = kernel_choice_validation(
        inputs,
        outputs,
        {"short": fixed_rbf(0.15), "long": fixed_rbf(0.5)},
        n_restarts_optimizer=0,
        random_state=0,
    )

    assert list(result.columns) == [
        "kernel",
        "LOOCV_RMSE",
        "Spearman_rank_correlation",
    ]
    assert set(result["kernel"]) == {"short", "long"}
    assert np.all(result["LOOCV_RMSE"] >= 0.0)


def test_fixed_length_scale_validation_tests_all_scales():
    inputs, outputs = small_regression_data()

    result = fixed_length_scale_validation(
        inputs,
        outputs,
        lambda length_scale: fixed_rbf(length_scale),
        [0.1, 0.3],
    )

    assert_allclose(result["length_scale"], [0.1, 0.3])
    assert len(result) == 2


def test_length_scale_cv_sensitivity_builds_initialisation_and_bounds_rows():
    inputs, outputs = small_regression_data()

    result = length_scale_cv_sensitivity(
        inputs,
        outputs,
        lambda initial_scale, bounds: RBF(
            length_scale=initial_scale,
            length_scale_bounds=bounds,
        ),
        historical_initial_scale=0.2,
        historical_bounds="fixed",
        initial_scales=[0.1, 0.4],
        bounds_to_test=["fixed"],
        n_restarts_optimizer=0,
        random_state=0,
    )

    assert list(result["test"]) == ["Initialisation", "Initialisation", "Bounds"]
    assert len(result) == 3


@pytest.mark.parametrize("name", ["UCB", "EI", "PI"])
def test_acquisition_values_delegates_to_acquisition_module(name):
    mean = np.array([0.2, 0.8])
    std = np.array([0.1, 0.3])
    kwargs = {"best": 0.5, "xi": 0.05, "kappa": 1.5}

    result = acquisition_values(mean, std, name, **kwargs)

    if name == "UCB":
        expected = upper_confidence_bound(mean, std, 1.5)
    elif name == "EI":
        expected = expected_improvement(mean, std, 0.5, 0.05)
    else:
        expected = probability_of_improvement(mean, std, 0.5, 0.05)
    assert_allclose(result, expected)


def test_acquisition_values_rejects_unknown_name():
    with pytest.raises(ValueError, match="UCB.*EI.*PI"):
        acquisition_values(np.array([0.0]), np.array([1.0]), "BAD", best=0.0)


def test_cartesian_resolution_sensitivity_reports_candidate_counts():
    result = cartesian_resolution_sensitivity(
        DeterministicGP(),
        dimension=2,
        resolutions=[2, 3],
        acquisition="UCB",
        best=0.0,
        kappa=1.0,
    )

    assert list(result["n_candidates"]) == [4, 9]
    assert_allclose(result.iloc[0]["candidate"], [1.0, 1.0])
    assert_allclose(result.iloc[1]["candidate"], [1.0, 1.0])


def test_random_resolution_sensitivity_is_reproducible():
    kwargs = dict(
        gp=DeterministicGP(),
        lower=np.array([0.0, 0.0]),
        upper=np.array([1.0, 1.0]),
        candidate_counts=[8, 16],
        acquisition="UCB",
        best=0.0,
        kappa=1.0,
        seed=9,
    )

    first = random_resolution_sensitivity(**kwargs)
    second = random_resolution_sensitivity(**kwargs)

    assert list(first["n_candidates"]) == [8, 16]
    for a, b in zip(first["candidate"], second["candidate"]):
        assert_allclose(a, b)
    assert_allclose(first["acquisition_max"], second["acquisition_max"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known regression: validation passes slabs_per_batch to the compatibility "
        "_original_grid_blocks_4d wrapper, but the wrapper forwards to "
        "original_grid_blocks, which expects block_size instead."
    ),
)
def test_exhaustive_kappa_sensitivity_4d_runs_on_compatibility_wrapper():
    result = exhaustive_kappa_sensitivity_4d(
        DeterministicGP(),
        resolution=2,
        kappas=[1.0, 2.0],
        slabs_per_batch=1,
    )

    assert set(result) == {1.0, 2.0}
    assert_allclose(result[1.0]["candidate"], [1.0, 1.0, 1.0, 1.0])
