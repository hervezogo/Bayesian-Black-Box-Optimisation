"""Simple baselines and result-table utilities."""

import numpy as np
import pandas as pd



def compute_best_so_far(results: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative-best and improvement columns to a long-form results table."""
    required = {"function", "week", "y"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = results.copy().sort_values(["function", "week"])
    frame["best_so_far"] = frame.groupby("function")["y"].cummax()
    initial = frame.groupby("function")["best_so_far"].transform("first")
    frame["absolute_improvement"] = frame["best_so_far"] - initial
    denominator = initial.abs().replace(0.0, np.nan)
    frame["relative_improvement"] = frame["absolute_improvement"] / denominator
    return frame.reset_index(drop=True)


def summarise_progression(results: pd.DataFrame) -> pd.DataFrame:
    """Return one compact performance row per challenge function."""
    frame = compute_best_so_far(results)
    rows: list[dict[str, float | int]] = []
    for function, group in frame.groupby("function", sort=True):
        group = group.sort_values("week")
        best_row = group.loc[group["y"].idxmax()]
        relative = group.iloc[-1]["relative_improvement"]
        rows.append(
            {
                "function": int(function),
                "initial_best": float(group.iloc[0]["best_so_far"]),
                "final_best": float(group.iloc[-1]["best_so_far"]),
                "best_week": int(best_row["week"]),
                "absolute_improvement": float(group.iloc[-1]["absolute_improvement"]),
                "relative_improvement": float(relative) if pd.notna(relative) else np.nan,
                "n_queries": int(len(group)),
            }
        )
    return pd.DataFrame(rows)
