"""Search-region definition, candidate generation, and selection."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor

from .config import RANDOM_SEED


def distance_to_n_closest_point(
    x: float, 
    y: float, 
    n: int,  
    domain: np.ndarray,
) -> float:
    
    """Calculate distance to of point(x,y) from n closest point"""
    
    p = np.array([x, y])
    d = np.array([np.linalg.norm(p - np.array([u,v])) for (u,v) in domain ])
    r = np.sort(d)[:n]
    
    return float(r.sum())


def cartesian_grid(
    dimension: int,
    points_per_axis: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:

    """Generate a Cartesian grid over the unit hypercube."""
    
    if lower is None:
        lower = np.zeros(dimension)

    if upper is None:
        upper = np.ones(dimension)

    axes = [np.linspace(lower[i], upper[i], points_per_axis) for i in range(dimension)]
    mesh = np.meshgrid(*axes)
    
    return np.column_stack([axis.ravel() for axis in mesh])



def search_region(
    centre: np.ndarray,
    radius: float | np.ndarray,
    upper_radius: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    """Define a bounded search region around a centre point, 
    or optionality asymmetric searhc region"""
    
    if upper_radius is None:
        upper_radius = radius
    
    lower = np.clip(centre - radius, 0.0, 1.0)
    upper = np.clip(centre + radius, 0.0, 1.0)
    
    return lower, upper


def random_candidates(
    dimension: int,
    n_candidates: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    
    """Generate uniformly distributed random candidates."""

    rng = np.random.default_rng(seed)

    if lower is None:
        lower = np.zeros(dimension)

    if upper is None:
        upper = np.ones(dimension)

    return rng.uniform(lower, upper, size=(n_candidates, dimension))


def sobol_candidates(
    dimension: int,
    power: int = 14,
    seed: int = 42,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    
    """Generate scrambled Sobol candidates."""

    sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=seed,
    )
    candidates = sampler.random_base2(power)

    if lower is not None and upper is not None:
        candidates = qmc.scale(candidates, lower, upper)

    return candidates



def filter_candidates(
    candidates: np.ndarray,
    inputs: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    
    """Remove candidates too close to observed points."""

    tree = cKDTree(inputs)
    distances, _ = tree.query(candidates)

    return candidates[distances > min_distance]


def incumbent(
    inputs: np.ndarray,
    outputs: np.ndarray,
) -> tuple[np.ndarray, float]:
    
    """Return the best observed point and value."""

    index = np.argmax(outputs)

    return inputs[index], outputs[index]


def nearest_observation_distance(
    candidates: np.ndarray,
    observed_inputs: np.ndarray,
) -> np.ndarray:
    
    """Return each candidate's distance to its nearest observed point."""

    tree = cKDTree(observed_inputs)
    distances, _ = tree.query(candidates)

    return distances


def furthest_candidate(
    candidates: np.ndarray,
    observed_inputs: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Return the candidate furthest from its nearest observed point."""

    distances = nearest_observation_distance(candidates, observed_inputs)

    if len(distances) == 0:
        raise ValueError("at least one candidate is required")

    index = int(np.argmax(distances))

    return candidates[index].copy(), float(distances[index]), index


def run_search_strategy(
    inputs,
    outputs,
    candidates,
    kernel,
    acquisition_function,
    gp_kwargs=None,
    **acquisition_kwargs,
):
    
    if gp_kwargs is None:
        gp_kwargs = {}

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gp_kwargs.get("alpha", 1e-10),
        normalize_y=gp_kwargs.get("normalize_y", True),
        n_restarts_optimizer=gp_kwargs.get("n_restarts_optimizer", 0),
        random_state=gp_kwargs.get("random_state", None))

    gp.fit(inputs, outputs)

    mu, sigma = gp.predict(candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-12)

    scores = acquisition_function(
        mu,
        sigma,
        **acquisition_kwargs
    )

    next_point = np.round(candidates[np.argmax(scores)], 6)

    return next_point, scores