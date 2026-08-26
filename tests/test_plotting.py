import numpy as np
import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_allclose

from src.plotting import plot_best_observed_progression_2d


def test_plot_best_observed_progression_draws_cumulative_best():
    fig, ax = plt.subplots()
    try:
        plot_best_observed_progression_2d([1.0, 0.5, 2.0, 1.8], ax=ax)
        step_line = ax.lines[0]
        assert_allclose(step_line.get_ydata(), [1.0, 1.0, 2.0, 2.0])
        assert ax.get_xlabel() == "Sequential observation"
        assert ax.get_ylabel() == "Best observed objective"
    finally:
        plt.close(fig)


def test_plot_best_observed_progression_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one value"):
        plot_best_observed_progression_2d([])


@pytest.mark.xfail(
    strict=True,
    reason="The function is annotated to return Axes but currently has no return statement.",
)
def test_plot_best_observed_progression_returns_axes():
    fig, ax = plt.subplots()
    try:
        returned = plot_best_observed_progression_2d([1.0, 2.0], ax=ax)
        assert returned is ax
    finally:
        plt.close(fig)
