import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import src.data as data_module
from src.data import extract, historical_record, load_initial_data, update_data


def test_extract_returns_numeric_inputs_and_outputs():
    frame = pd.DataFrame({
        "x1": [0.1, 0.2],
        "x2": [0.3, 0.4],
        "y": [1.0, 2.0],
    })

    inputs, outputs = extract(frame)

    assert_allclose(inputs, [[0.1, 0.3], [0.2, 0.4]])
    assert_allclose(outputs, [1.0, 2.0])


def test_update_data_appends_new_observation():
    frame = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y": [1.0]})

    updated = update_data(frame, np.array([0.3, 0.4]), 2.0)

    assert len(updated) == 2
    assert_allclose(updated.iloc[-1].to_numpy(dtype=float), [0.3, 0.4, 2.0])
    # The original frame is not mutated when an append occurs.
    assert len(frame) == 1


def test_update_data_skips_existing_point_value_pair():
    frame = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y": [1.0]})

    updated = update_data(frame, np.array([0.1, 0.2]), 1.0)

    assert len(updated) == 1


def test_update_data_allows_same_point_with_different_value():
    frame = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y": [1.0]})

    updated = update_data(frame, np.array([0.1, 0.2]), 1.5)

    assert len(updated) == 2


def test_update_data_rejects_wrong_dimension():
    frame = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y": [1.0]})

    with pytest.raises(ValueError, match="dimensions"):
        update_data(frame, np.array([0.3]), 2.0)


def test_update_data_rejects_non_finite_values():
    frame = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y": [1.0]})

    with pytest.raises(ValueError, match="finite"):
        update_data(frame, np.array([np.nan, 0.4]), 2.0)

    with pytest.raises(ValueError, match="finite"):
        update_data(frame, np.array([0.3, 0.4]), np.inf)


def test_load_initial_data_builds_expected_dataframe(tmp_path, monkeypatch):
    function_dir = tmp_path / "function_3"
    function_dir.mkdir()
    np.save(function_dir / "initial_inputs.npy", np.array([[0.1, 0.2], [0.3, 0.4]]))
    np.save(function_dir / "initial_outputs.npy", np.array([[1.5], [2.5]]))
    monkeypatch.setattr(data_module, "INITIAL_DATA_DIR", tmp_path)

    frame = load_initial_data(3)

    assert list(frame.columns) == ["x1", "x2", "y"]
    assert_allclose(frame.to_numpy(), [[0.1, 0.2, 1.5], [0.3, 0.4, 2.5]])


def test_historical_record_reads_query_and_result(tmp_path, monkeypatch):
    queries_file = tmp_path / "queries.csv"
    results_file = tmp_path / "results.csv"
    pd.DataFrame({
        "Function": ["F_2"],
        "Week_4": ["[0.25,\u00a00.75]"],
    }).to_csv(queries_file, index=False)
    pd.DataFrame({
        "Function": ["F_2"],
        "Week_4": [3.2],
    }).to_csv(results_file, index=False)
    monkeypatch.setattr(data_module, "QUERIES_FILE", queries_file)
    monkeypatch.setattr(data_module, "RESULTS_FILE", results_file)

    point, value = historical_record(2, 4)

    assert_allclose(point, [0.25, 0.75])
    assert value == 3.2
