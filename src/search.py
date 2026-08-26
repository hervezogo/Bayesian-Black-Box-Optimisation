"""Search-region definition, candidate generation, and selection."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor

from .config import RANDOM_SEED
from .acquisition import expected_improvement, upper_confidence_bound

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


def _original_grid_blocks_4d(
    resolution,
    *,
    slabs_per_batch=2,
    lower=0.0,
    upper=1.0,
):
    """Yield the historical 4D Cartesian grid in original meshgrid order."""

    axis = np.linspace(lower, upper, resolution)

    # Original flattened order:
    # x2 (slowest) -> x1 -> x3 -> x4 (fastest)
    x1_block, x3_block, x4_block = np.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij",
    )

    base = np.column_stack([
        x1_block.ravel(),
        x3_block.ravel(),
        x4_block.ravel(),
    ])

    for start in range(0, resolution, slabs_per_batch):
        x2_values = axis[
            start:min(start + slabs_per_batch, resolution)
        ]

        yield np.vstack([
            np.column_stack([
                base[:, 0],
                np.full(len(base), x2),
                base[:, 1],
                base[:, 2],
            ])
            for x2 in x2_values
        ])



def exhaustive_cartesian_search_4d(
    gp,
    resolution,
    *,
    best=None,
    kappa=None,
    xi=0.0,
    compute_ei=False,
    compute_ucb=False,
    slabs_per_batch=2,
):
    """
    Exhaustively scan every point of the original 4D Cartesian grid.

    This gives the same argmax as materialising the full grid, while
    evaluating the GP in memory-safe batches.
    """
    if not compute_ei and not compute_ucb:
        raise ValueError("At least one acquisition must be requested.")

    if compute_ei and best is None:
        raise ValueError("'best' is required when computing EI.")

    if compute_ucb and kappa is None:
        raise ValueError("'kappa' is required when computing UCB.")

    best_ei_score = -np.inf
    best_ucb_score = -np.inf

    best_ei_point = None
    best_ucb_point = None

    best_ei_index = None
    best_ucb_index = None

    offset = 0

    for candidates in _original_grid_blocks_4d(
        resolution,
        slabs_per_batch=slabs_per_batch,
    ):
        mean, std = gp.predict(
            candidates,
            return_std=True,
        )

        if compute_ei:
            ei = expected_improvement(mean,std,best=best,xi=xi)
            idx = int(np.argmax(ei))
            score = float(ei[idx])

            # Strict '>' preserves the first occurrence globally,
            # matching np.argmax on the fully materialised grid.
            if score > best_ei_score:
                best_ei_score = score
                best_ei_point = candidates[idx].copy()
                best_ei_index = offset + idx

        if compute_ucb:
            ucb = mean + kappa * std
            idx = int(np.argmax(ucb))
            score = float(ucb[idx])

            if score > best_ucb_score:
                best_ucb_score = score
                best_ucb_point = candidates[idx].copy()
                best_ucb_index = offset + idx

        offset += len(candidates)

    result = {
        "n_candidates": resolution ** 4,
        "ei_point": best_ei_point,
        "ei_max": best_ei_score if compute_ei else None,
        "ei_flat_index": best_ei_index,
        "ucb_point": best_ucb_point,
        "ucb_max": best_ucb_score if compute_ucb else None,
        "ucb_flat_index": best_ucb_index,
    }

    if compute_ei and compute_ucb:
        cross_candidates = np.vstack([best_ei_point, best_ucb_point])

        mean_cross, std_cross = gp.predict(cross_candidates,return_std=True)

        ei_cross = expected_improvement(mean_cross, std_cross, best=best, xi=xi)
        ucb_cross = mean_cross + kappa * std_cross

        result.update({
            "candidate_distance": float(
                np.linalg.norm(best_ei_point - best_ucb_point)),
            "ei_at_ucb_point": float(ei_cross[1]),
            "ei_retention_at_ucb_point": (
                float(ei_cross[1] / best_ei_score)
                if best_ei_score > 1e-12
                else np.nan),
            "ucb_at_ei_point": float(ucb_cross[0]),
            "ucb_normalised_gap_at_ei_point": float(
                (best_ucb_score - ucb_cross[0])
                / max(abs(best_ucb_score), 1e-12))})

    return result


def original_grid_blocks(
    dimension: int,
    resolution: int,
    *,
    block_size: int = 250_000,
    lower: float | np.ndarray | None = None,
    upper: float | np.ndarray | None = None,
):
    """Yield the historical default-meshgrid Cartesian grid in fixed-size blocks.

    The candidate order is exactly the same as:

        axes = [np.linspace(lower[i], upper[i], resolution) for i in range(dimension)]
        mesh = np.meshgrid(*axes)  # default indexing="xy"
        grid = np.column_stack([axis.ravel() for axis in mesh])

    Unlike materialising ``grid``, memory use is O(block_size * dimension).
    Runtime is still O(resolution ** dimension), so exhaustive grids remain
    unsuitable at large dimension/resolution combinations.
    """
    dimension = int(dimension)
    resolution = int(resolution)
    block_size = int(block_size)

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    lower = (
        np.zeros(dimension)
        if lower is None
        else np.broadcast_to(np.asarray(lower, dtype=float), (dimension,))
    )
    upper = (
        np.ones(dimension)
        if upper is None
        else np.broadcast_to(np.asarray(upper, dtype=float), (dimension,))
    )

    if np.any(lower > upper):
        raise ValueError("lower bounds cannot exceed upper bounds")

    axes = [
        np.linspace(lower[i], upper[i], resolution)
        for i in range(dimension)
    ]

    total = resolution ** dimension
    mesh_shape = (resolution,) * dimension

    for start in range(0, total, block_size):
        stop = min(start + block_size, total)
        flat_indices = np.arange(start, stop, dtype=np.int64)

        # np.meshgrid(..., indexing="xy") swaps the first two array axes.
        mesh_indices = np.column_stack(
            np.unravel_index(flat_indices, mesh_shape)
        )
        coordinate_indices = mesh_indices.copy()

        if dimension >= 2:
            coordinate_indices[:, [0, 1]] = coordinate_indices[:, [1, 0]]

        candidates = np.empty(
            (len(flat_indices), dimension),
            dtype=float,
        )

        for axis_index in range(dimension):
            candidates[:, axis_index] = axes[axis_index][
                coordinate_indices[:, axis_index]
            ]

        yield candidates


def validate_original_grid_ordering(
    max_dimension: int = 6,
    resolution: int = 3,
) -> bool:
    """Verify the streamed ordering against np.meshgrid on tiny grids."""
    for dimension in range(1, int(max_dimension) + 1):
        axes = [
            np.linspace(0.0, 1.0, resolution)
            for _ in range(dimension)
        ]
        mesh = np.meshgrid(*axes)
        reference = np.column_stack(
            [axis.ravel() for axis in mesh]
        )
        streamed = np.vstack(
            list(
                original_grid_blocks(
                    dimension,
                    resolution,
                    block_size=7,
                )
            )
        )
        if not np.array_equal(reference, streamed):
            return False

    return True


def exhaustive_cartesian_search(
    gp,
    dimension: int,
    resolution: int,
    *,
    best: float | None = None,
    kappa: float | None = None,
    xi: float = 0.0,
    compute_ei: bool = False,
    compute_ucb: bool = False,
    block_size: int = 250_000,
    lower: float | np.ndarray | None = None,
    upper: float | np.ndarray | None = None,
) -> dict:
    """Exhaustively maximise EI and/or UCB on a Cartesian grid.

    The full historical candidate set is scanned in original meshgrid order.
    Strict ``>`` updates preserve the first global maximiser, matching
    ``np.argmax`` tie-breaking on the fully materialised grid.
    """
    if not compute_ei and not compute_ucb:
        raise ValueError("request EI and/or UCB")
    if compute_ei and best is None:
        raise ValueError("'best' is required when computing EI")
    if compute_ucb and kappa is None:
        raise ValueError("'kappa' is required when computing UCB")

    best_ei_score = -np.inf
    best_ucb_score = -np.inf
    best_ei_point = None
    best_ucb_point = None
    best_ei_index = None
    best_ucb_index = None

    offset = 0

    for candidates in original_grid_blocks(
        dimension,
        resolution,
        block_size=block_size,
        lower=lower,
        upper=upper,
    ):
        mean, std = gp.predict(candidates, return_std=True)

        if compute_ei:
            scores = expected_improvement(
                mean,
                std,
                best=best,
                xi=xi,
            )
            index = int(np.argmax(scores))
            score = float(scores[index])

            if score > best_ei_score:
                best_ei_score = score
                best_ei_point = candidates[index].copy()
                best_ei_index = offset + index

        if compute_ucb:
            scores = upper_confidence_bound(
                mean,
                std,
                kappa=kappa,
            )
            index = int(np.argmax(scores))
            score = float(scores[index])

            if score > best_ucb_score:
                best_ucb_score = score
                best_ucb_point = candidates[index].copy()
                best_ucb_index = offset + index

        offset += len(candidates)

    result = {
        "n_candidates": int(resolution) ** int(dimension),
        "ei_point": best_ei_point,
        "ei_max": best_ei_score if compute_ei else None,
        "ei_flat_index": best_ei_index,
        "ucb_point": best_ucb_point,
        "ucb_max": best_ucb_score if compute_ucb else None,
        "ucb_flat_index": best_ucb_index,
    }

    if compute_ei and compute_ucb:
        result["candidate_distance"] = float(
            np.linalg.norm(best_ei_point - best_ucb_point)
        )

    return result


def local_sobol_candidates(
    centre: np.ndarray,
    radius: float | np.ndarray,
    observed_inputs: np.ndarray | None = None,
    *,
    power: int = 14,
    seed: int = 123,
    min_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and optionally de-duplicate Sobol candidates in a local box."""
    centre = np.asarray(centre, dtype=float).reshape(-1)
    radius = np.broadcast_to(
        np.asarray(radius, dtype=float),
        centre.shape,
    )

    lower, upper = search_region(centre, radius)
    candidates = sobol_candidates(
        dimension=centre.size,
        power=power,
        seed=seed,
        lower=lower,
        upper=upper,
    )

    if observed_inputs is not None and min_distance > 0.0:
        candidates = filter_candidates(
            candidates,
            np.asarray(observed_inputs, dtype=float),
            min_distance=min_distance,
        )

    if len(candidates) == 0:
        raise RuntimeError("all local Sobol candidates were filtered out")

    return candidates, lower, upper


# Optional compatibility wrappers for notebooks that still use old names.
def _original_grid_blocks_4d(resolution, **kwargs):
    yield from original_grid_blocks(4, resolution, **kwargs)


def _original_grid_blocks_5d(resolution, **kwargs):
    yield from original_grid_blocks(5, resolution, **kwargs)


def exhaustive_cartesian_search_4d(gp, resolution, **kwargs):
    return exhaustive_cartesian_search(
        gp,
        4,
        resolution,
        **kwargs)


def exhaustive_cartesian_search_5d(gp, resolution, **kwargs):
    return exhaustive_cartesian_search(
        gp,
        5,
        resolution,
        **kwargs)

""" The full candidate array is no longer materialised. `exhaustive_cartesian_search()` scans the same historical `np.meshgrid(...).ravel()` ordering in fixed-size blocks, preserving the candidate set and `np.argmax` tie ordering while reducing peak memory usage.
"""