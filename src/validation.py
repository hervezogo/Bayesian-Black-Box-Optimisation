"""Reusable quantitative validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from collections.abc import Iterable, Mapping

from .search import cartesian_grid, random_candidates, _original_grid_blocks_4d
from .acquisition import ( 
    expected_improvement, 
    probability_of_improvement, 
    upper_confidence_bound)


# Sequential optimisation performance

def sequential_performance(
    initial_outputs: np.ndarray,
    query_outputs: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
   
    """Summarise incumbent improvement over sequential queries."""

    initial_outputs = np.asarray(initial_outputs, dtype=float).reshape(-1)
    query_outputs = np.asarray(query_outputs, dtype=float).reshape(-1)

    initial_best = float(np.max(initial_outputs))
    best_so_far = np.maximum.accumulate(
        np.concatenate(([initial_best], query_outputs))
    )

    performance = pd.DataFrame({
        "query": np.arange(1, len(query_outputs) + 1),
        "query_output": query_outputs,
        "best_so_far": best_so_far[1:],
        "improvement": np.maximum(query_outputs - best_so_far[:-1], 0.0),
    })

    final_best = float(best_so_far[-1])
    absolute_improvement = final_best - initial_best

    summary = pd.DataFrame({
        "metric": [
            "Initial incumbent",
            "Final best observed",
            "Absolute improvement",
            "Relative improvement",
            "Queries improving incumbent",
        ],
        "value": [
            initial_best,
            final_best,
            absolute_improvement,
            absolute_improvement / abs(initial_best),
            int((performance["improvement"] > 0).sum()),
        ],
    })

    return performance, summary



# LOOCV and kernel-choice validation

def loocv_predictions(
    inputs: np.ndarray,
    outputs: np.ndarray,
    kernel,
    *,
    alpha: float = 1e-10,
    normalize_y: bool = True,
    n_restarts_optimizer: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Return leave-one-out GP predictions."""

    predictions = np.empty(len(outputs), dtype=float)

    for train_idx, test_idx in LeaveOneOut().split(inputs):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        gp.fit(inputs[train_idx], outputs[train_idx])
        predictions[test_idx] = gp.predict(inputs[test_idx])

    return predictions


def kernel_choice_validation(
    inputs: np.ndarray,
    outputs: np.ndarray,
    kernels: Mapping[str, object],
    *,
    alpha: float = 1e-10,
    normalize_y: bool = True,
    n_restarts_optimizer: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare kernels using LOOCV RMSE and Spearman correlation."""

    rows = []

    for name, kernel in kernels.items():
        predictions = loocv_predictions(
            inputs,
            outputs,
            kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        rows.append({
            "kernel": name,
            "LOOCV_RMSE": np.sqrt(mean_squared_error(outputs, predictions)),
            "Spearman_rank_correlation": spearmanr(outputs, predictions).statistic,
        })

    return pd.DataFrame(rows)

# Kernels Length-scale validation

def fixed_length_scale_validation(
    inputs,
    outputs,
    kernel_factory,
    length_scales,
    alpha=1e-10,
    normalize_y=True,
):
    """Compare fixed length scales using LOOCV prediction diagnostics."""

    rows = []

    for length_scale in length_scales:
        predictions = loocv_predictions(
            inputs,
            outputs,
            kernel_factory(length_scale),
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=0,
        )

        rows.append({
            "length_scale": length_scale,
            "LOOCV_RMSE": np.sqrt(
                mean_squared_error(outputs, predictions)
            ),
            "Spearman_rank_correlation": spearmanr(
                outputs, predictions
            ).statistic,
        })

    return pd.DataFrame(rows)



def length_scale_cv_sensitivity(
    inputs,
    outputs,
    kernel_factory,
    historical_initial_scale,
    historical_bounds,
    initial_scales,
    bounds_to_test,
    alpha=1e-10,
    normalize_y=True,
    n_restarts_optimizer=0,
    random_state=42,
):
    """Assess initialisation and bounds sensitivity using LOOCV diagnostics."""

    rows = []

    tests = [
        ("Initialisation", initial_scale, historical_bounds)
        for initial_scale in initial_scales
    ] + [
        ("Bounds", historical_initial_scale, bounds)
        for bounds in bounds_to_test
    ]

    for test, initial_scale, bounds in tests:
        predictions = loocv_predictions(
            inputs,
            outputs,
            kernel_factory(initial_scale, bounds),
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        rows.append({
            "test": test,
            "initial_scale": initial_scale,
            "bounds": bounds,
            "LOOCV_RMSE": np.sqrt(
                mean_squared_error(outputs, predictions)
            ),
            "Spearman_rank_correlation": spearmanr(
                outputs, predictions
            ).statistic,
        })

    return pd.DataFrame(rows)



def acquisition_values(
    mean: np.ndarray,
    std: np.ndarray,
    acquisition: str,
    best: float,
    *,
    xi: float = 0.0,
    kappa: float = 2.0,
) -> np.ndarray:
    """Return acquisition scores using src.acquisition."""

    if acquisition == "UCB":
        return upper_confidence_bound(mean, std, kappa)

    if acquisition == "EI":
        return expected_improvement(mean, std, best, xi)

    if acquisition == "PI":
        return probability_of_improvement(mean, std, best, xi)

    raise ValueError("acquisition must be 'UCB', 'EI', or 'PI'")


def cartesian_resolution_sensitivity(
    gp: GaussianProcessRegressor,
    dimension: int,
    resolutions: Iterable[int],
    acquisition: str,
    best: float,
    *,
    xi: float = 0.0,
    kappa: float = 2.0,
) -> pd.DataFrame:
    """Check candidate stability as Cartesian-grid resolution increases."""

    rows = []

    for resolution in resolutions:
        candidates = cartesian_grid(dimension, resolution)
        mean, std = gp.predict(candidates, return_std=True)
        scores = acquisition_values(
            mean,
            std,
            acquisition,
            best,
            xi=xi,
            kappa=kappa,
        )

        idx = np.argmax(scores)

        rows.append({
            "resolution": resolution,
            "n_candidates": len(candidates),
            "candidate": candidates[idx],
            "acquisition_max": scores[idx],
        })

    return pd.DataFrame(rows)


def random_resolution_sensitivity(
    gp: GaussianProcessRegressor,
    lower: np.ndarray,
    upper: np.ndarray,
    candidate_counts: Iterable[int],
    acquisition: str,
    best: float,
    *,
    xi: float = 0.0,
    kappa: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Check candidate stability as random candidate count increases."""

    rows = []
    dimension = len(lower)

    for n_candidates in candidate_counts:
        candidates = random_candidates(
            dimension,
            n_candidates,
            lower,
            upper,
            seed,
        )

        mean, std = gp.predict(candidates, return_std=True)
        scores = acquisition_values(
            mean,
            std,
            acquisition,
            best,
            xi=xi,
            kappa=kappa,
        )

        idx = np.argmax(scores)

        rows.append({
            "n_candidates": n_candidates,
            "candidate": candidates[idx],
            "acquisition_max": scores[idx],
        })

    return pd.DataFrame(rows)


def exhaustive_kappa_sensitivity_4d(
    gp,
    resolution,
    kappas,
    *,
    slabs_per_batch=2,
):
    """Evaluate several UCB kappas on the same exhaustive original-order grid."""
    best = {kappa: {"score": -np.inf,"candidate": None} for kappa in kappas}

    for candidates in _original_grid_blocks_4d(resolution, slabs_per_batch=slabs_per_batch):
        mean, std = gp.predict(candidates, return_std=True)

        for kappa in kappas:
            scores = mean + kappa * std
            idx = int(np.argmax(scores))
            score = float(scores[idx])

            if score > best[kappa]["score"]:
                best[kappa] = {"score": score, "candidate": candidates[idx].copy()}

    return best
