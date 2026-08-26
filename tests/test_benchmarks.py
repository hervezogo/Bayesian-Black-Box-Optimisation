import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.benchmarks import compute_best_so_far, summarise_progression


def sample_results():
    return pd.DataFrame({
        "function": [2, 1, 1, 2, 1, 2],
        "week": [1, 2, 1, 3, 3, 2],
        "y": [5.0, 2.0, 1.0, 6.0, 1.5, 4.0],
    })


def test_compute_best_so_far_sorts_and_computes_progression():
    result = compute_best_so_far(sample_results())

    f1 = result[result["function"] == 1]
    assert list(f1["week"]) == [1, 2, 3]
    assert_allclose(f1["best_so_far"], [1.0, 2.0, 2.0])
    assert_allclose(f1["absolute_improvement"], [0.0, 1.0, 1.0])
    assert_allclose(f1["relative_improvement"], [0.0, 1.0, 1.0])


def test_compute_best_so_far_handles_zero_initial_value():
    frame = pd.DataFrame({"function": [1, 1], "week": [1, 2], "y": [0.0, 2.0]})

    result = compute_best_so_far(frame)

    assert result["relative_improvement"].isna().all()


def test_compute_best_so_far_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        compute_best_so_far(pd.DataFrame({"function": [1], "y": [1.0]}))


def test_summarise_progression_returns_one_row_per_function():
    summary = summarise_progression(sample_results()).set_index("function")

    assert list(summary.index) == [1, 2]
    assert summary.loc[1, "initial_best"] == pytest.approx(1.0)
    assert summary.loc[1, "final_best"] == pytest.approx(2.0)
    assert summary.loc[1, "best_week"] == 2
    assert summary.loc[1, "n_queries"] == 3
    assert summary.loc[2, "initial_best"] == pytest.approx(5.0)
    assert summary.loc[2, "final_best"] == pytest.approx(6.0)
    assert summary.loc[2, "best_week"] == 3
