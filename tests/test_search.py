import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.gaussian_process.kernels import RBF

from src.acquisition import upper_confidence_bound
from src.search import (
    cartesian_grid,
    distance_to_n_closest_point,
    exhaustive_cartesian_search,
    filter_candidates,
    furthest_candidate,
    local_sobol_candidates,
    nearest_observation_distance,
    original_grid_blocks,
    random_candidates,
    run_search_strategy,
    search_region,
    sobol_candidates,
    validate_original_grid_ordering,
)


class DeterministicGP:
    """Small test double with a predictable posterior."""

    def predict(self, candidates, return_std=True):
        candidates = np.asarray(candidates)
        mean = candidates.sum(axis=1)
        std = np.full(len(candidates), 0.1)
        return (mean, std) if return_std else mean


def test_distance_to_n_closest_point_sums_nearest_distances():
    domain = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    distance = distance_to_n_closest_point(0.0, 0.0, 2, domain)

    assert distance == pytest.approx(1.0)


def test_cartesian_grid_shape_bounds_and_order():
    grid = cartesian_grid(
        dimension=2,
        points_per_axis=3,
        lower=np.array([0.0, 10.0]),
        upper=np.array([1.0, 12.0]),
    )

    x1 = np.linspace(0.0, 1.0, 3)
    x2 = np.linspace(10.0, 12.0, 3)
    mesh = np.meshgrid(x1, x2)
    reference = np.column_stack([axis.ravel() for axis in mesh])

    assert grid.shape == (9, 2)
    assert_array_equal(grid, reference)


def test_search_region_clips_to_unit_cube():
    lower, upper = search_region(np.array([0.1, 0.9]), radius=0.25)

    assert_allclose(lower, [0.0, 0.65])
    assert_allclose(upper, [0.35, 1.0])


@pytest.mark.xfail(
    strict=True,
    reason="Known regression: search_region currently ignores upper_radius.",
)
def test_search_region_supports_asymmetric_radius():
    lower, upper = search_region(
        np.array([0.5, 0.5]),
        radius=np.array([0.1, 0.2]),
        upper_radius=np.array([0.3, 0.4]),
    )

    assert_allclose(lower, [0.4, 0.3])
    assert_allclose(upper, [0.8, 0.9])


def test_random_candidates_are_reproducible_and_bounded():
    lower = np.array([0.2, 0.4])
    upper = np.array([0.3, 0.8])

    first = random_candidates(2, 20, lower, upper, seed=7)
    second = random_candidates(2, 20, lower, upper, seed=7)

    assert_allclose(first, second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)


def test_sobol_candidates_are_reproducible_and_scaled():
    lower = np.array([0.2, 0.4])
    upper = np.array([0.3, 0.8])

    first = sobol_candidates(2, power=4, seed=11, lower=lower, upper=upper)
    second = sobol_candidates(2, power=4, seed=11, lower=lower, upper=upper)

    assert first.shape == (16, 2)
    assert_allclose(first, second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)


def test_filter_candidates_removes_points_within_min_distance():
    candidates = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 1.0]])
    observed = np.array([[0.0, 0.0]])

    filtered = filter_candidates(candidates, observed, min_distance=0.25)

    assert_allclose(filtered, [[1.0, 1.0]])


def test_nearest_observation_distance_and_furthest_candidate():
    candidates = np.array([[0.1, 0.0], [0.5, 0.0], [1.0, 1.0]])
    observed = np.array([[0.0, 0.0]])

    distances = nearest_observation_distance(candidates, observed)
    point, distance, index = furthest_candidate(candidates, observed)

    assert_allclose(distances, [0.1, 0.5, np.sqrt(2.0)])
    assert_allclose(point, [1.0, 1.0])
    assert distance == pytest.approx(np.sqrt(2.0))
    assert index == 2


def test_furthest_candidate_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        furthest_candidate(np.empty((0, 2)), np.array([[0.0, 0.0]]))


def test_original_grid_blocks_match_materialised_meshgrid():
    dimension = 3
    resolution = 3
    axes = [np.linspace(0.0, 1.0, resolution) for _ in range(dimension)]
    mesh = np.meshgrid(*axes)
    reference = np.column_stack([axis.ravel() for axis in mesh])

    streamed = np.vstack(list(original_grid_blocks(dimension, resolution, block_size=5)))

    assert_array_equal(streamed, reference)


def test_original_grid_blocks_validates_arguments():
    with pytest.raises(ValueError, match="dimension"):
        list(original_grid_blocks(0, 3))
    with pytest.raises(ValueError, match="resolution"):
        list(original_grid_blocks(2, 1))
    with pytest.raises(ValueError, match="block_size"):
        list(original_grid_blocks(2, 3, block_size=0))
    with pytest.raises(ValueError, match="lower bounds"):
        list(original_grid_blocks(2, 3, lower=[0.5, 0.0], upper=[0.4, 1.0]))


def test_validate_original_grid_ordering():
    assert validate_original_grid_ordering(max_dimension=5, resolution=3)


def test_exhaustive_cartesian_search_finds_ucb_maximum():
    result = exhaustive_cartesian_search(
        DeterministicGP(),
        dimension=2,
        resolution=3,
        kappa=2.0,
        compute_ucb=True,
        block_size=2,
    )

    assert result["n_candidates"] == 9
    assert_allclose(result["ucb_point"], [1.0, 1.0])
    assert result["ucb_max"] == pytest.approx(2.2)
    assert result["ei_point"] is None


def test_exhaustive_cartesian_search_validates_required_parameters():
    gp = DeterministicGP()

    with pytest.raises(ValueError, match="request EI"):
        exhaustive_cartesian_search(gp, 2, 3)
    with pytest.raises(ValueError, match="best"):
        exhaustive_cartesian_search(gp, 2, 3, compute_ei=True)
    with pytest.raises(ValueError, match="kappa"):
        exhaustive_cartesian_search(gp, 2, 3, compute_ucb=True)


def test_local_sobol_candidates_returns_box_and_respects_filter():
    centre = np.array([0.5, 0.5])
    observed = np.array([[0.5, 0.5]])

    candidates, lower, upper = local_sobol_candidates(
        centre,
        radius=0.2,
        observed_inputs=observed,
        power=5,
        seed=3,
        min_distance=0.05,
    )

    assert_allclose(lower, [0.3, 0.3])
    assert_allclose(upper, [0.7, 0.7])
    assert np.all(candidates >= lower)
    assert np.all(candidates <= upper)
    assert np.all(nearest_observation_distance(candidates, observed) > 0.05)


def test_run_search_strategy_returns_one_of_supplied_candidates():
    inputs = np.array([[0.0], [0.5], [1.0]])
    outputs = np.array([0.0, 1.0, 0.0])
    candidates = np.array([[0.25], [0.5], [0.75]])

    next_point, scores = run_search_strategy(
        inputs,
        outputs,
        candidates,
        kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
        acquisition_function=upper_confidence_bound,
        gp_kwargs={"n_restarts_optimizer": 0, "random_state": 0},
        kappa=1.0,
    )

    assert scores.shape == (3,)
    assert any(np.allclose(next_point, candidate) for candidate in candidates)
