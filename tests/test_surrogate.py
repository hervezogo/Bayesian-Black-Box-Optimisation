import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

from src.surrogate import build_gp, build_kernel, fit_gp, predict_gp


def test_build_rbf_kernel_without_noise():
    kernel = build_kernel("rbf", length_scale=0.3, noise_level=None)

    assert isinstance(kernel, RBF)
    assert kernel.length_scale == pytest.approx(0.3)


def test_build_matern_kernel_requires_dimension():
    with pytest.raises(ValueError, match="dimension"):
        build_kernel("matern", dimension=None)


def test_build_matern_kernel_contains_requested_dimension_and_nu():
    kernel = build_kernel(
        "matern",
        dimension=3,
        length_scale=0.2,
        nu=2.5,
        noise_level=None,
    )

    # ConstantKernel * Matern -> Product; k2 is the Matern component.
    assert isinstance(kernel.k2, Matern)
    assert_allclose(kernel.k2.length_scale, [0.2, 0.2, 0.2])
    assert kernel.k2.nu == pytest.approx(2.5)


def test_build_kernel_rejects_unknown_type():
    with pytest.raises(ValueError, match="rbf.*matern"):
        build_kernel("linear")


def test_build_gp_returns_unfitted_normalised_regressor():
    model = build_gp(RBF(length_scale=0.2), random_state=5)

    assert isinstance(model, GaussianProcessRegressor)
    assert model.normalize_y is True
    assert model.random_state == 5


def test_fit_and_predict_gp_return_expected_shapes():
    inputs = np.array([[0.0], [0.5], [1.0]])
    outputs = np.array([0.0, 1.0, 0.0])
    model = build_gp(
        RBF(length_scale=0.2, length_scale_bounds="fixed"),
        n_restarts_optimizer=0,
        random_state=0,
    )

    fitted = fit_gp(model, inputs, outputs)
    mean, std = predict_gp(fitted, np.array([[0.25], [0.75]]))

    assert fitted is model
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std >= 0.0)
