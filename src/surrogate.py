"""Gaussian-process surrogate construction, fitting, and prediction."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel


### Building Kernel (RBF or Matérn)

def build_kernel(
    kernel_type: str,
    dimension: int | None = None,
    length_scale: float = 0.2,
    length_scale_bounds: str | tuple[float, float] = (1e-2, 3.0),
    nu: float = 2.5,
    noise_level: float | None = 1e-6,
):

    """Build an RBF or Matérn kernel."""

    if kernel_type == "rbf":
        kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)

    elif kernel_type == "matern":
        if dimension is None:
            raise ValueError("dimension is required for a Matérn kernel")

        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.full(dimension, length_scale),
            length_scale_bounds=length_scale_bounds,
            nu=nu)

    else:
        raise ValueError("kernel_type must be 'rbf' or 'matern'")

    if noise_level is not None:
        kernel += WhiteKernel(noise_level=noise_level)

    return kernel


### Building Gaussian process Surrogate Function

def build_gp(
    kernel,
    alpha: float = 1e-10,
    n_restarts_optimizer: int = 0,
    random_state: int | None = None,
) -> GaussianProcessRegressor:
    
    """Build an unfitted Gaussian-process surrogate."""

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state)


### Fitting the  Gaussian process Surrogate Function

def fit_gp(
    model: GaussianProcessRegressor,
    inputs: np.ndarray,
    outputs: np.ndarray,
) -> GaussianProcessRegressor:
 
    """Fit the Gaussian-process surrogate."""

    model.fit(inputs, outputs)
    return model


### Model Prediction given candidates

def predict_gp(
    model: GaussianProcessRegressor,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    
    """Return posterior mean and standard deviation."""

    return model.predict(candidates, return_std=True)