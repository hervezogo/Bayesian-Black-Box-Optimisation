"""Plotting helpers shared by the optimisation notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy.interpolate import Rbf


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
    azim: float = -60,
):
    
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
        pad=0.1,
    )

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
    cbar.ax.tick_params(labelsize=10)
    
    ax.scatter(data['x1'], data['x2'], c='red', s=90, label='Observed')
    ax.scatter(next_point[0], next_point[1], c='black', s=150, label='Next')
    ax.scatter(incumbent[0], incumbent[1], c='white', s=150, label='Incumbent')

    ax.set_title(f'Next Point: {next_point[0]:.6f}, {next_point[1]:.6f}', fontsize=12)
    ax.set_xticks(np.linspace(0,1,5))
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_xlabel("x1", fontsize=15)
    ax.set_ylabel("x2", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=10, loc="upper left")
    
    #fig.suptitle(f'Bayesian Optimization: GP + {acquisition_type}', fontsize=8)
    
    


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
    ylim=(0.00, 0.32)
):
    
    """Visualise a manual local refinement from the current incumbent using observed objective values."""

    
    local_data = data[ data["x1"].between(*xlim) & data["x2"].between(*ylim)]

    plt.figure(figsize=(8, 5))

    scatter = plt.scatter(
        local_data["x1"],
        local_data["x2"],
        c=local_data["y"],
        label="Observed evaluations",
        cmap="viridis",
        s=110,
        edgecolor="black")

    for _, row in local_data.iterrows():
        plt.annotate(
            f'{row["y"]:.3f}',
            (row["x1"], row["x2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    plt.scatter(
        incumbent[0],
        incumbent[1],
        marker="*",
        s=280,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        label="Current incumbent"
    )

    plt.scatter(
        candidate[0],
        candidate[1],
        marker="X",
        s=180,
        facecolor="none",
        edgecolor="red",
        linewidth=2,
        label=candidate_label
    )

    plt.annotate(
        "",
        xy=candidate,
        xytext=incumbent,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.8,
            color="red"
        )
    )

    plt.colorbar(scatter, label="Observed objective value")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


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
    return ax


def plot_sequential_query_performance(
    performance: pd.DataFrame,
    title: str = "Sequential Query Performance vs Previous Incumbent",
):
    
    """Plot queried outputs against the previous incumbent."""

    fig, ax = plt.subplots(figsize=(9, 4))

    weeks = performance["query"]

    ax.plot(
        weeks,
        performance["query_output"],
        marker="o",
        label="Queried output",
    )

    previous_incumbent = (
        performance["best_so_far"] - performance["improvement"]
    )

    ax.plot(
        weeks,
        previous_incumbent,
        linestyle="--",
        label="Previous incumbent",
    )

    ax.set_yscale("symlog", linthresh=1e-18)
    ax.set_xlabel("Week")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.set_xticks(weeks)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

