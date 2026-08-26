import numpy as np
from numpy.testing import assert_allclose

from src.acquisition import (
    decaying_kappa,
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)


def test_upper_confidence_bound_matches_definition():
    mean = np.array([0.2, 1.0, -0.5])
    std = np.array([0.1, 0.4, 0.2])

    scores = upper_confidence_bound(mean, std, kappa=1.5)

    assert_allclose(scores, mean + 1.5 * std)


def test_expected_improvement_at_zero_improvement():
    mean = np.array([1.0])
    std = np.array([2.0])

    score = expected_improvement(mean, std, best=1.0, xi=0.0)

    expected = 2.0 / np.sqrt(2.0 * np.pi)
    assert_allclose(score, [expected])


def test_expected_improvement_is_zero_for_zero_std():
    scores = expected_improvement(
        mean=np.array([0.0, 2.0]),
        std=np.array([0.0, 1e-14]),
        best=1.0,
    )

    assert_allclose(scores, [0.0, 0.0])


def test_probability_of_improvement_is_half_at_threshold():
    score = probability_of_improvement(
        mean=np.array([1.0]),
        std=np.array([0.5]),
        best=1.0,
        xi=0.0,
    )

    assert_allclose(score, [0.5])


def test_probability_of_improvement_is_zero_for_zero_std():
    scores = probability_of_improvement(
        mean=np.array([2.0, 0.0]),
        std=np.array([0.0, 1e-14]),
        best=1.0,
    )

    assert_allclose(scores, [0.0, 0.0])


def test_decaying_kappa_matches_inverse_square_root_schedule():
    assert decaying_kappa(iteration=1, scale=2.5) == 2.5
    assert_allclose(decaying_kappa(iteration=4, scale=2.5), 1.25)
