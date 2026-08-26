"""Plotting helpers shared by the optimisation notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy.interpolate import Rbf, griddata


def plot_data_exploration(data: pd.DataFrame):
    
    """Plot 2D and 3D views of two-dimensional observations."""

    fig = plt.figure(figsize=(11, 4))

    ## # 2D-Data Exploration
    
    ax1 = fig.add_subplot(1, 2, 1)
    scatter = ax1.scatter(
        data["x1"],
        data["x2"],
        c=data["y"],
        cmap="viridis",
        s=50)

    colorbar = fig.colorbar(scatter, ax=ax1)
    colorbar.set_label("Output y", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    ax1.tick_params(axis="both", labelsize=8)
    ax1.set_xlabel("Input x1")
    ax1.set_ylabel("Input x2")
    ax1.set_title("Scatter plot of inputs colored by output")

    ## # 3D-Data Exploration
    
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        data["x1"],
        data["x2"],
        data["y"],
        c=data["y"],
        cmap="viridis",
        s=100)

    ax2.tick_params(axis="both", labelsize=7)
    ax2.set_xlabel("Input x1")
    ax2.set_ylabel("Input x2")
    ax2.set_zlabel("Output y")
    ax2.set_title("3D Scatter of Inputs vs Output")

    plt.show()
    

def plot_surrogate_landscapes(
    grid: np.ndarray,
    gp_values: np.ndarray,
    data: pd.DataFrame,
    elev: float = 30,
    azim: float = -60):
    
    """Plot the GP posterior mean as a 2D contour
    and the thin-plate RBF interpolation as a 3D surface.
    """

    fig = plt.figure(figsize=(9, 4))

    # 1. GP Posterior Mean: 2D contour

    ax1 = fig.add_subplot(1, 2, 1)

    grid_n = int(np.sqrt(grid.shape[0]))

    if grid_n**2 != grid.shape[0]:
        raise ValueError("grid must contain a square number of points.")

    xx1 = grid[:, 0].reshape(grid_n, grid_n)
    xx2 = grid[:, 1].reshape(grid_n, grid_n)
    gp_surface = gp_values.reshape(grid_n, grid_n)

    contour = ax1.contourf(xx1, xx2, gp_surface, levels=20, cmap="viridis")

    ax1.scatter(data["x1"], data["x2"], c="red", s=50, label="Observed")

    incumbents = np.array([[0.702637, 0.926564],[0.714286, 0.000000]])

    ax1.scatter(incumbents[:, 0], incumbents[:, 1], c="orange", edgecolors="black",
            s=100, marker="*", label="Incumbents", zorder=5)

    ax1.set_title("GP Posterior Mean", fontsize=9)
    ax1.set_xlabel(r"$x_1$", fontsize=8)
    ax1.set_ylabel(r"$x_2$", fontsize=8)
    ax1.tick_params(axis="both", labelsize=7)
    ax1.legend(loc="upper left", fontsize=7)
    
    cbar1 = fig.colorbar(contour, ax=ax1)
    cbar1.set_label("GP posterior mean", fontsize=7)
    cbar1.ax.tick_params(labelsize=6)

    # 2. Thin-Plate RBF Interpolation: 3D surface

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    x1 = data["x1"].to_numpy()
    x2 = data["x2"].to_numpy()
    y = data["y"].to_numpy()

    # Synthetic boundary-hugging points used only for visualisation.
    n_edge = 20
    t = np.linspace(0.0, 1.0, n_edge)

    xb = np.concatenate([t,t,np.zeros_like(t),np.ones_like(t)])
    yb = np.concatenate([np.zeros_like(t), np.ones_like(t), t,t])

    boundary_points = np.unique(np.column_stack([xb, yb]),axis=0)
    xb = boundary_points[:, 0]
    yb = boundary_points[:, 1]

    # Illustrative boundary values used to reduce edge extrapolation.
    # They are not observations and are not used by the GP optimisation model.
    y_boundary = np.full(len(boundary_points), -8e-4)

    x1_aug = np.concatenate([x1, xb])
    x2_aug = np.concatenate([x2, yb])
    y_aug = np.concatenate([y, y_boundary])

    rbf = Rbf(x1_aug, x2_aug, y_aug, function="thin_plate",smooth=0)

    rbf_grid_n = 500
    xg = np.linspace(0.0, 1.0, rbf_grid_n)
    yg = np.linspace(0.0, 1.0, rbf_grid_n)

    X1, X2 = np.meshgrid(xg, yg)
    Z = rbf(X1, X2)

    surface = ax2.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.85,linewidth=0)

    ax2.scatter(x1, x2, y, c="red", s=35, edgecolor="black", label="Observed")

    ax2.set_title("Illustrative Thin-Plate RBF Interpolation", fontsize=9)

    ax2.set_xlabel(r"$x_1$", fontsize=8)
    ax2.set_ylabel(r"$x_2$", fontsize=8)
    ax2.set_zlabel("y", fontsize=8)

    ax2.tick_params(axis="both", labelsize=6)
    ax2.legend(fontsize=7)

    ax2.view_init(elev=elev,azim=azim)

    cbar2 = fig.colorbar(
        surface,
        ax=ax2,
        shrink=0.7,
        pad=0.1)

    cbar2.set_label("Interpolated y", fontsize=7)
    cbar2.ax.tick_params(labelsize=6)

    fig.tight_layout()

    plt.show()

    
    

def plot_acquisition_2d(
    grid: np.ndarray, 
    acquisition_values: np.ndarray,
    data: pd.DataFrame,
    incumbent: np.ndarray,
    next_point: np.ndarray,
    acquisition_type: str,
    ax: Axes | None = None):

    """Plot contour surfaces for next point"""
    if ax is None:
        fig, ax = plt.subplots(1, 1,figsize=(5,3))
    else:
        fig = ax.figure
        
    grid_n = int(np.sqrt(grid.shape[0])) ## points per axis in grid
    
    xx1 = grid[:, 0].reshape(grid_n, grid_n)
    xx2 = grid[:, 1].reshape(grid_n, grid_n)
    v = acquisition_values.reshape(grid_n,grid_n)

    contour = ax.contourf(xx1, xx2, v, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, label="UCB")
    cbar.ax.tick_params(labelsize=12)
    
    ax.scatter(data['x1'], data['x2'], c='red', s=90, label='Observed')
    ax.scatter(next_point[0], next_point[1], c='black', s=150, label='Next')
    ax.scatter(incumbent[0], incumbent[1], c='white', s=150, label='Incumbent')

    ax.set_title(f'Next Point: {next_point[0]:.6f}, {next_point[1]:.6f}', fontsize=12)
    ax.set_xticks(np.linspace(0,1,5))
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_xlabel("x1", fontsize=15)
    ax.set_ylabel("x2", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=14, loc="upper left")
    
    #fig.suptitle(f'Bayesian Optimization: GP + {acquisition_type}', fontsize=8)
    
    
def plot_pairwise_observations_2d(
    inputs,
    outputs,
    title="Initial observations",
):
    
    """
    Plot pairwise projections of the initial observations,
    highlighting the best observed point.
    """
    dimension = inputs.shape[1]
    pairs = list(combinations(range(dimension), 2))
    best_idx = np.argmax(outputs)

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(5.3 * len(pairs), 4.5),
    )

    axes = np.atleast_1d(axes)

    for ax, (i, j) in zip(axes, pairs):
        scatter = ax.scatter(
            inputs[:, i],
            inputs[:, j],
            c=outputs,
            s=70)

        ax.scatter(
            inputs[best_idx, i],
            inputs[best_idx, j],
            marker="*",
            s=220,
            label="Initial incumbent")

        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    fig.colorbar(
        scatter,
        ax=axes.tolist(),
        label="Observed objective")

    fig.suptitle(title)
    plt.show()




def plot_rbf_surface_3d(data: pd.DataFrame, elev: int = 22, azim: int =255):

    """Plot 2D visualisation of RBF Surface from data input"""
    
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']

    # Add synthetic boundary hugging points
    n_edge = 20
    t = np.linspace(0, 1, n_edge)

    # four edges of the unit square
    xb = np.concatenate([t, t, np.zeros_like(t), np.ones_like(t)])
    yb = np.concatenate([np.zeros_like(t), np.ones_like(t), t, t])

    # Add synthetic negative boundary values to visually constrain the RBF surface.
    # This reduces edge extrapolation and improves the contrast of promising interior regions. 
    # These points are illustrative only and are not used in the optimisation model.
    boundary_points = np.unique(np.column_stack([xb, yb]),axis=0)
    xb = boundary_points[:, 0]
    yb = boundary_points[:, 1]
    y_boundary = np.full(len(boundary_points), -8e-4)
    
    # combine original + boundary points
    x1_aug = np.concatenate([x1, xb])
    x2_aug = np.concatenate([x2, yb])
    y_aug  = np.concatenate([y, y_boundary])

    # exact interpolation on augmented dataset
    rbf = Rbf(x1_aug, x2_aug, y_aug, function='thin_plate', smooth=0)

    # grid
    grid_n = 500
    xg = np.linspace(0, 1, grid_n)
    yg = np.linspace(0, 1, grid_n)
    X1, X2 = np.meshgrid(xg, yg)
    Z = rbf(X1, X2)

    # plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.85, linewidth=0)
     
    ax.scatter(x1, x2, y, c='red', s=50, edgecolor='k', label='Original points')
    #ax.scatter(xb, yb, y_boundary, c='orange', s=25, label='Boundary control points')

    ax.set_title("Illustrative Thin Plate RBF Surface Interpolation")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.legend()

    ax.view_init(elev, azim)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
    cbar.set_label("Interpolated y", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


def plot_global_to_local_refinement_2d(
    X_grid: np.ndarray,
    ei: np.ndarray,
    X_local_grid: np.ndarray,
    mu_surface: np.ndarray,
    data: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    next_point: np.ndarray,
):
    """Plot the transition from global EI search to local GP refinement."""

    inputs = data[["x1", "x2"]].to_numpy()
    outputs = data["y"].to_numpy()
    x_best = inputs[np.argmax(outputs)]

    # Global grid
    global_grid_size = int(np.sqrt(len(X_grid)))
    xx1 = X_grid[:, 0].reshape(global_grid_size, global_grid_size)
    xx2 = X_grid[:, 1].reshape(global_grid_size, global_grid_size)

    # Local grid
    local_grid_size = int(np.sqrt(len(X_local_grid)))
    xx1_local = X_local_grid[:, 0].reshape(local_grid_size, local_grid_size)
    xx2_local = X_local_grid[:, 1].reshape(local_grid_size, local_grid_size)

    # Observations inside trust region
    local_mask = np.all(
        (inputs >= lower) & (inputs <= upper),
        axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # LEFT PANEL: Global overview
    ax = axes[0]

    ax.contourf(
        xx1,
        xx2,
        ei.reshape(global_grid_size, global_grid_size),
        levels=20,
        cmap="viridis",
    )

    ax.scatter(
        inputs[:, 0], inputs[:, 1],
        c="red", s=35, label="Observed"
    )

    ax.scatter(
        x_best[0], x_best[1],
        c="orange", edgecolors="black",
        s=80, marker="*", label="Current best"
    )

    ax.scatter(
        next_point[0], next_point[1],
        c="white", edgecolors="black",
        s=120, marker="*", label="Next point"
    )

    rect = Rectangle(
        (lower[0], lower[1]),
        upper[0] - lower[0],
        upper[1] - lower[1],
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xlabel("x1", fontsize=7)
    ax.set_ylabel("x2", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title("Global view (from previous EI)", fontsize=8)
    ax.legend(fontsize=6, loc="best")

    # RIGHT PANEL: Local zoom
    ax = axes[1]

    contour = ax.contourf(
        xx1_local,
        xx2_local,
        mu_surface,
        levels=20,
        cmap="viridis",
    )

    ax.scatter(
        inputs[local_mask, 0],
        inputs[local_mask, 1],
        c="red", s=40,
        label="Observed",
        zorder=3,
    )

    ax.scatter(
        x_best[0], x_best[1],
        c="orange", edgecolors="black",
        s=80, marker="*",
        label="Current best",
        zorder=4,
    )

    ax.scatter(
        next_point[0], next_point[1],
        c="white", edgecolors="black",
        s=120, marker="*",
        label="Next point",
        zorder=5,
    )

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_xticks(np.linspace(lower[0], upper[0], 5))
    ax.set_yticks(np.linspace(lower[1], upper[1], 5))
    ax.set_xlabel("x1", fontsize=7)
    ax.set_ylabel("x2", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title("Local zoom: GP posterior mean", fontsize=8)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Predicted objective", fontsize=7)

    fig.suptitle("Bayesian Optimization: Local PI search", fontsize=8)
    fig.tight_layout()

    plt.tight_layout()
    plt.show()


def plot_local_refinement_2d(
    data,
    incumbent,
    candidate,
    title,
    candidate_label="Proposed manual refinement",
    xlim=(0.60, 0.86),
    ylim=(0.00, 0.32),
    ax: Axes | None = None):
    
    """Visualise a manual local refinement from the current incumbent."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    local_data = data[
        data["x1"].between(*xlim)
        & data["x2"].between(*ylim)]

    scatter = ax.scatter(
        local_data["x1"],
        local_data["x2"],
        c=local_data["y"],
        label="Observed evaluations",
        cmap="viridis",
        s=110,
        edgecolor="black")

    for _, row in local_data.iterrows():
        ax.annotate(
            f'{row["y"]:.3f}',
            (row["x1"], row["x2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8)

    ax.scatter(
        incumbent[0],
        incumbent[1],
        marker="*",
        s=280,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        label="Current incumbent")

    ax.scatter(
        candidate[0],
        candidate[1],
        marker="X",
        s=180,
        facecolor="none",
        edgecolor="red",
        linewidth=2,
        label=candidate_label)

    ax.annotate(
        "",
        xy=candidate,
        xytext=incumbent,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.8,
            color="red"))

    fig.colorbar(
        scatter,
        ax=ax,
        label="Observed objective value")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, loc="upper right")



def plot_best_observed_progression_2d(
    values: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str = "Best observed progression",
) -> Axes:
    """Plot cumulative best objective value across sequential observations."""
    y = np.asarray(values, dtype=float).reshape(-1)
    if len(y) == 0:
        raise ValueError("at least one value is required")
    best = np.maximum.accumulate(y)
    observations = np.arange(1, len(best) + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.step(observations, best, where="post")
    ax.scatter(observations, best, s=24)
    ax.set(
        xlabel="Sequential observation",
        ylabel="Best observed objective",
        title=title,
    )
    ax.grid(alpha=0.25)
    


def plot_initial_landscape_3d(
    data,
    x3_fixed=None,
    grid_resolution=50,
    scatter_title="3D Scatter of Initial Observations",
    surface_title=None,
    suptitle="Function 3: Initial exploratory visualisation",
):
    """
    Plot:
    1. A 3D scatter of the observed initial points coloured by y
    2. A 3D interpolated surface over (x1, x2) with x3 fixed

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain columns: 'x1', 'x2', 'x3', 'y'.
    x3_fixed : float or None, default=None
        Value of x3 used for the surface slice.
        If None, the median observed x3 is used.
    grid_resolution : int, default=50
        Number of grid points per axis for the interpolated surface.
    scatter_title : str
        Title of the 3D scatter plot.
    surface_title : str or None
        Title of the interpolated surface plot.
        If None, a default title is created.
    suptitle : str
        Overall figure title.
    """
    if x3_fixed is None:
        x3_fixed = np.median(data["x3"])

    x1_lin = np.linspace(0, 1, grid_resolution)
    x2_lin = np.linspace(0, 1, grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)

    points = data[["x1", "x2", "x3"]].values
    values = data["y"].values

    grid_z = griddata(
        points,
        values,
        (X1_grid, X2_grid, x3_fixed * np.ones_like(X1_grid)),
        method="linear",
    )

    fig = plt.figure(figsize=(12, 5))

    # Left panel: 3D scatter of actual observations
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax1.scatter(
        data["x1"],
        data["x2"],
        data["x3"],
        c=data["y"],
        cmap="viridis",
        s=100,
    )
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("x3")
    ax1.set_title(scatter_title)
    fig.colorbar(sc1, ax=ax1, shrink=0.7, label="Observed objective")

    # Right panel: interpolated surface coloured by interpolated y
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(
        X1_grid,
        X2_grid,
        grid_z,
        cmap="viridis",
        edgecolor="k",
        alpha=0.85,
    )
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    if surface_title is None:
        surface_title = f"Interpolated surface at x3 = {x3_fixed:.2f}"

    ax2.set_title(surface_title)
    fig.colorbar(surf, ax=ax2, shrink=0.7, label="Interpolated objective")

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_search_path_3d(
    inputs,
    outputs,
    query_inputs,
    n_initial=15,
    function_id=3):
    """Plot and save the 3D sequential search path."""

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Initial observations
    ax.scatter(
        inputs[:n_initial, 0],
        inputs[:n_initial, 1],
        inputs[:n_initial, 2],
        s=45,
        color="red",
        alpha=0.6,
        label="Initial observations")

    # Sequential queries
    ax.plot(
        query_inputs[:, 0],
        query_inputs[:, 1],
        query_inputs[:, 2],
        marker="o",
        linewidth=1.2,
        label="Sequential queries")

    # Week labels
    for week, point in enumerate(query_inputs, start=1):
        ax.text(
            point[0],
            point[1],
            point[2],
            f"W{week}",
            fontsize=8)

    # Final incumbent
    best_idx = np.argmax(outputs)

    ax.scatter(
        inputs[best_idx, 0],
        inputs[best_idx, 1],
        inputs[best_idx, 2],
        marker="*",
        s=260,
        label="Final incumbent")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(f"Function {function_id}: global exploration to local refinement")
    ax.legend()

    plt.tight_layout()

    # Save automatically
    plt.savefig(
        f"function_{function_id}.png",
        dpi=300,
        bbox_inches="tight")

    plt.show()

    
def plot_sequential_query_performance(
    performance: pd.DataFrame,
    function_id: int, 
    yscale: str = "symlog",
    plot_type: str = "line"):
    
    """Plot weekly query outputs and the evolution of the incumbent."""

    fig, ax = plt.subplots(figsize=(12, 5.5))

    weeks = performance["query"]
    query_output = performance["query_output"]
    best_so_far = performance["best_so_far"]

    previous_incumbent = (
        performance["best_so_far"] - performance["improvement"]
    )

    # ------------------------------------------------------------
    # Weekly query outputs
    #
    # Step-wise representation:
    # W1 value is held from W1 -> W2,
    # W2 value from W2 -> W3, etc.
    # ------------------------------------------------------------
    ax.step(
        weeks,
        query_output,
        where="post",
        linewidth=1,
        linestyle="--",
        alpha=0.85,
        label="Weekly query",
        zorder=3,
    )

    ax.scatter(
        weeks,
        query_output,
        s=45,
        zorder=4,
    )

    # ------------------------------------------------------------
    # Best-so-far trajectory
    #
    # Starts at W1 with the incumbent that existed before
    # the first sequential query.
    # ------------------------------------------------------------
    initial_week = weeks.iloc[0]
    initial_incumbent = previous_incumbent.iloc[0]

    incumbent_weeks = np.r_[
        initial_week,
        weeks.to_numpy(),
    ]

    incumbent_values = np.r_[
        initial_incumbent,
        best_so_far.to_numpy(),
    ]

    ax.step(
        incumbent_weeks,
        incumbent_values,
        where="post",
        linewidth=1,
        color="green",
        #linestyle="--",
        label="Best so far",
        zorder=2,
    )

    # Mark initial incumbent, changes, and final incumbent
    incumbent_points = np.r_[
        True,
        np.diff(incumbent_values) != 0,
    ]

    incumbent_points[-1] = True

    ax.scatter(
        incumbent_weeks[incumbent_points],
        incumbent_values[incumbent_points],
        s=38,
        color="green",
        zorder=5,
    )

    # ------------------------------------------------------------
    # Highlight overall best observed query
    # ------------------------------------------------------------
    best_idx = query_output.idxmax()
    best_week = performance.loc[best_idx, "query"]
    best_value = performance.loc[best_idx, "query_output"]

    ax.scatter(
        best_week,
        best_value,
        s=100,
        color="firebrick",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label="Best observed query",
    )

    ax.annotate(
        f"best={best_value:.4g} @ W{int(best_week)}",
        xy=(best_week, best_value),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
        color="firebrick",
    )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    ax.set_yscale(yscale)

    ax.set_xticks(weeks)
    ax.set_xticklabels(
        [f"W{int(w)}" for w in weeks]
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Objective value")
    ax.set_title(
        "Sequential Query Performance and Best-So-Far Evolution"
    )

    ax.grid(True, alpha=0.20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )

    plt.tight_layout()
    fig.savefig(f"summary_f{function_id}.png", dpi=300,bbox_inches="tight")
    plt.show()