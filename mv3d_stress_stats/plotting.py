"""
Visualization utilities for stress statistics and ensemble analysis.
"""

from __future__ import annotations
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from mv3d_stress_stats.ensemble import EnsembleAnalysisResult
from mv3d_stress_stats.joint import JointDistribution2D
from mv3d_stress_stats.correlation import SpatialCorrelationResult


def plot_ensemble_pdf(
    result: EnsembleAnalysisResult,
    show_individual: bool = True,
    show_kde: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Axes:
    """Plot ensemble PDF alongside individual RVE spatial PDFs.
    
    Args:
        result: EnsembleAnalysisResult from ensemble.analyze().
        show_individual: Whether to plot faint lines for per-RVE histograms.
        show_kde: Whether to plot ensemble KDE curve if available.
        ax: Optional matplotlib axes.
        figsize: Figure size if new axes created.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    x = result.bin_centers
    
    if show_individual:
        for i, h in enumerate(result.histograms):
            label = "Individual RVEs" if i == 0 else None
            ax.plot(x, h, color="gray", alpha=0.35, linewidth=1.0, label=label)
            
    # Ensemble PDF (stepped or line)
    ax.plot(
        x,
        result.ensemble_pdf,
        color="crimson",
        linewidth=2.2,
        label=f"Ensemble Mixture PDF ({result.quantity_name})",
    )
    
    if show_kde and result.ensemble_kde is not None and result.kde_eval_points is not None:
        ax.plot(
            result.kde_eval_points,
            result.ensemble_kde,
            color="navy",
            linestyle="--",
            linewidth=1.8,
            label="Ensemble KDE",
        )
        
    ax.set_xlabel(f"Quantity: {result.quantity_name}")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Ensemble PDF — {result.quantity_name} (Load Step {result.load_step})")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    return ax


def plot_variance_decomposition(
    result: EnsembleAnalysisResult,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Axes:
    """Plot variance decomposition bar chart showing spatial vs realization variability."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    vd = result.variance_decomposition
    labels = ["Within-RVE\n(Spatial Heterogeneity)", "Between-RVE\n(Microstructure Realization)"]
    percentages = [vd.eta_within * 100.0, vd.eta_between * 100.0]
    colors = ["#2b5c8f", "#d95f02"]
    
    bars = ax.bar(labels, percentages, color=colors, width=0.55)
    ax.set_ylabel("Variance Contribution (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Variance Breakdown — {result.quantity_name}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        
    return ax


def plot_joint_density_2d(
    joint: JointDistribution2D,
    levels: Optional[Tuple[float, ...]] = (0.50, 0.90, 0.95),
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """Plot 2D joint density map with probability contour levels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    X, Y = np.meshgrid(joint.x_centers, joint.y_centers, indexing="ij")
    
    # Heatmap / pcolormesh
    pcm = ax.pcolormesh(
        joint.x_bins,
        joint.y_bins,
        joint.density.T,
        shading="flat",
        cmap="viridis",
    )
    plt.colorbar(pcm, ax=ax, label="Probability Density")
    
    # Overlay contours
    if levels is not None and len(levels) > 0:
        c_vals = [joint.find_contour_level(p) for p in levels]
        c_vals_sorted = sorted(c_vals)
        cs = ax.contour(
            X, Y, joint.density,
            levels=c_vals_sorted,
            colors="white",
            linewidths=1.5,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")
        
    ax.set_xlabel(joint.x_name)
    ax.set_ylabel(joint.y_name)
    ax.set_title(f"Joint Density: p({joint.x_name}, {joint.y_name})")
    
    return ax


def plot_spatial_autocorrelation(
    corr_res: SpatialCorrelationResult,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (7, 4.5),
) -> plt.Axes:
    """Plot radially averaged spatial autocorrelation function R(r)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    ax.plot(corr_res.lags, corr_res.autocorrelation_radial, "o-", color="#1f77b4", markersize=4, label="R(r)")
    ax.axhline(1.0 / np.e, color="crimson", linestyle="--", label=f"1/e threshold (λ = {corr_res.correlation_length:.2f})")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    
    ax.set_xlabel("Lag distance r")
    ax.set_ylabel("Autocorrelation R(r)")
    ax.set_title(f"Spatial Autocorrelation (N_points={corr_res.n_points}, N_eff={corr_res.n_effective:.1f})")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    return ax


def plot_summary_dashboard(
    ensemble_result: EnsembleAnalysisResult,
    joint_dist: JointDistribution2D,
    corr_result: SpatialCorrelationResult,
    figsize: Tuple[int, int] = (15, 12),
) -> plt.Figure:
    """Generate a 4-panel publication-quality overview dashboard.
    
    Panels:
        (0, 0): Ensemble PDF with individual RVE curves and KDE
        (0, 1): Variance decomposition breakdown (% within vs % between)
        (1, 0): 2D Stress-state probability density in invariant space with contours
        (1, 1): Radially averaged 3D spatial autocorrelation and effective sample size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Ensemble PDF
    plot_ensemble_pdf(ensemble_result, ax=axes[0, 0])
    
    # 2. Variance breakdown
    plot_variance_decomposition(ensemble_result, ax=axes[0, 1])
    
    # 3. 2D Joint invariant density
    plot_joint_density_2d(joint_dist, ax=axes[1, 0])
    
    # 4. Spatial Autocorrelation
    plot_spatial_autocorrelation(corr_result, ax=axes[1, 1])
    
    plt.tight_layout()
    return fig

