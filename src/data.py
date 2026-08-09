"""Data loading and observation management."""

import numpy as np
import pandas as pd
from .config import INITIAL_DATA_DIR


### Initial Data

def load_initial_data(function_id: int) -> pd.DataFrame:
    
    """Load the initial observations for one challenge function."""

    function_dir = INITIAL_DATA_DIR / f"function_{function_id}"
    inputs = np.load(function_dir / "initial_inputs.npy")
    outputs = np.load(function_dir / "initial_outputs.npy").reshape(-1)

    input_columns = [f"x{i + 1}" for i in range(inputs.shape[1])]
    data = pd.DataFrame(inputs, columns=input_columns)
    data["y"] = outputs
    
    return data


### Extract inputs and outputs from data observations

def extract(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    
    """Extract optimisation input and outputs from cumulated observed data"""
    
    inputs = data.drop(columns="y").to_numpy(dtype=float)
    outputs = data["y"].to_numpy(dtype=float)

    return inputs, outputs
    

### Observed Data

def update_data(
    data: pd.DataFrame,
    new_point: np.ndarray,
    new_value: float,
) -> pd.DataFrame:
    
    """Append one observed black-box evaluation if not already present to data"""

    new_point = np.asarray(new_point, dtype=float).reshape(-1)
    input_columns = [column for column in data.columns if column != "y"]

    if new_point.size != len(input_columns):
        raise ValueError(f"point has {new_point.size} dimensions; expected {len(input_columns)}")

    if not np.all(np.isfinite(new_point)) or not np.isfinite(new_value):
        raise ValueError("point and value must be finite")

    inputs, outputs = extract(data)
    exists = np.any(np.all(np.isclose(inputs, new_point), axis=1) & np.isclose(outputs, new_value))

    if exists:
        print("Point already exists, skipping addition.")
        return data

    new_row = pd.DataFrame([[*new_point, float(new_value)]],columns=[*input_columns, "y"])
    data = pd.concat([data, new_row], ignore_index=True)

    print("Point added!")

    return data