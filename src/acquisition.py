"""Acquisition functions for maximisation problems."""

import numpy as np
from scipy.stats import norm


### UCB Acquisition Function

def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:

    """Return Upper Confidence Bound scores."""

    return mean + kappa * std


### EI Acquisition Function

def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
    xi: float = 0.0,
) -> np.ndarray:
    
    """Return Expected Improvement scores."""

    improvement = mean - best - xi
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    scores = improvement * norm.cdf(z) + safe_std * norm.pdf(z)

    return np.where(std > 1e-12, scores, 0.0)


### PI Acquisition Function

def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
    xi: float = 0.0,
) -> np.ndarray:
    
    """Return Probability of Improvement scores."""

    safe_std = np.maximum(std, 1e-12)
    z = (mean - best - xi) / safe_std

    return np.where(std > 1e-12, norm.cdf(z), 0.0)


### Decaying Kappa for less exploration / more exploitation

def decaying_kappa(
    iteration: int,
    scale: float = 2.5,
) -> float:
    
    """Return the decaying UCB exploration coefficient."""

    return scale / np.sqrt(iteration)