"""
Visualization utilities for stress statistics and ensemble analysis.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from mv3d_stress_stats.ensemble import EnsembleAnalysisResult
from mv3d_stress_stats.joint import JointDistribution2D
from mv3d_stress_stats.correlation import SpatialCorrelationResult
from mv3d_stress_stats.directional import DirectionalHistogram
from mv3d_stress_stats.multiload import MultiloadAnalysisResult


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


def plot_loading_pdf_comparison(
    multiload_result: MultiloadAnalysisResult,
    load_steps: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot comparative PDFs and CDFs across multiple loading conditions.
    
    Panels:
        Left: Overlaid PDFs p(Q | L) for each loading condition L (distinguishing normal vs shear).
        Right: Cumulative Distribution Functions F(Q | L).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    steps = list(load_steps) if load_steps is not None else list(multiload_result.cases.keys())
    
    # Palette: distinct colors for different categories
    normal_colors = ["#1f77b4", "#2ca02c", "#9467bd"]  # Blue, Green, Purple
    shear_colors = ["#ff7f0e", "#d62728", "#8c564b"]   # Orange, Red, Brown
    other_colors = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    norm_idx = 0
    shear_idx = 0
    other_idx = 0
    
    for ls in steps:
        if ls not in multiload_result.results:
            continue
        res = multiload_result.results[ls]
        case = multiload_result.cases[ls]
        
        if case.category == "normal":
            color = normal_colors[norm_idx % len(normal_colors)]
            norm_idx += 1
            ls_style = "-"
        elif case.category in ("shear", "multiaxial_shear"):
            color = shear_colors[shear_idx % len(shear_colors)]
            shear_idx += 1
            ls_style = "--"
        else:
            color = other_colors[other_idx % len(other_colors)]
            other_idx += 1
            ls_style = "-."
            
        lbl = f"{case.label} ({case.category})"
        
        # Plot PDF: use KDE if available, else binned PDF
        if res.ensemble_kde is not None and res.kde_eval_points is not None:
            ax1.plot(res.kde_eval_points, res.ensemble_kde, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
            ax1.plot(res.bin_centers, res.ensemble_pdf, color=color, alpha=0.20, linewidth=1.0)
        else:
            ax1.plot(res.bin_centers, res.ensemble_pdf, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
            
        # Plot CDF
        dx = np.diff(res.bin_edges)
        cdf = np.cumsum(res.ensemble_pdf * dx)
        ax2.plot(res.bin_centers, cdf, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
        
    q_name = multiload_result.quantity
    q_label = r"\sigma_{\mathrm{VM}}" if q_name == "von_mises" else q_name
    ax1.set_xlabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(f"PDF Comparison: $p({q_label} \\mid L)$ across Loading Modes")
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.set_xlabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax2.set_ylabel(f"Cumulative Probability $F({q_label} \\mid L)$")
    ax2.set_title(f"CDF Comparison: Stress Accumulation across Loading Modes")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    return fig, (ax1, ax2)



def plot_directional_histogram(
    dir_hist: DirectionalHistogram,
    title: Optional[str] = None,
    color: str = "#2b5c8f",
    show_mean_line: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> plt.Axes:
    """Plot polar rose diagram of directional stress orientations.
    
    Args:
        dir_hist: DirectionalHistogram dataclass from compute_directional_histogram().
        title: Plot title.
        color: Wedge fill color.
        show_mean_line: If True, draws radial lines indicating circular mean orientation.
        ax: Matplotlib polar axes (must have projection='polar' if passed).
        figsize: Figure size.
        
    Returns:
        Matplotlib Axes with polar projection.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        
    d_theta = dir_hist.bin_edges[1] - dir_hist.bin_edges[0]
    
    ax.bar(
        dir_hist.bin_centers,
        dir_hist.frequencies,
        width=d_theta,
        bottom=0.0,
        color=color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.75,
    )
    
    if show_mean_line:
        max_r = float(np.max(dir_hist.frequencies)) * 1.15 if len(dir_hist.frequencies) > 0 else 1.0
        ax.plot([dir_hist.mean_direction, dir_hist.mean_direction], [0, max_r], color="crimson", linewidth=2.2, linestyle="-", label=f"Mean: {dir_hist.mean_direction_degrees:.1f}°")
        if dir_hist.symmetric:
            opp_mean = (dir_hist.mean_direction + np.pi) % (2.0 * np.pi)
            ax.plot([opp_mean, opp_mean], [0, max_r], color="crimson", linewidth=2.2, linestyle="-")
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), frameon=True, fontsize=8)
        
    ax.set_theta_zero_location("E")  # 0 rad along East (+X axis)
    ax.set_theta_direction(1)        # counterclockwise
    
    plane_str = dir_hist.plane.upper()
    disp_str = f"R={dir_hist.mean_resultant_length:.3f}, Disp={dir_hist.circular_dispersion:.3f}"
    
    if title is None:
        ax.set_title(f"Principal Stress Orientation ({plane_str})\n{disp_str}", va="bottom", fontsize=10, pad=15)
    else:
        ax.set_title(f"{title}\n{disp_str}", va="bottom", fontsize=10, pad=15)
        
    ax.grid(True, linestyle=":", alpha=0.6)
    return ax


def plot_multiload_directional_grid(
    histograms: Dict[str, DirectionalHistogram],
    cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Plot a grid of polar rose diagrams for multiple loading conditions."""
    n = len(histograms)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (5 * cols, 5 * rows)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw={"projection": "polar"})
    ax_list = np.atleast_1d(axes).ravel()
    
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728", "#8c564b"]
    
    for i, (label, dh) in enumerate(histograms.items()):
        c = colors[i % len(colors)]
        plot_directional_histogram(dh, title=label, color=c, ax=ax_list[i])
        
    # Hide unused subplots
    for j in range(i + 1, len(ax_list)):
        ax_list[j].set_visible(False)
        
    plt.tight_layout()
    return fig


def plot_loading_dashboard(
    multiload_result: MultiloadAnalysisResult,
    directional_histograms: Dict[str, DirectionalHistogram],
    load_steps: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Generate a comprehensive multi-loading & directional stress dashboard.
    
    Panels:
        (0, 0): Comparative PDFs p(sigma_VM | L) across canonical loading modes
        (0, 1): Comparative CDFs F(sigma_VM | L) and stress thresholds
        (1, 0): Directional rose diagram under normal tension (e.g. eps_x)
        (1, 1): Directional rose diagram under shear loading (e.g. eps_xy)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    
    ax_pdf = fig.add_subplot(gs[0, 0])
    ax_cdf = fig.add_subplot(gs[0, 1])
    ax_rose1 = fig.add_subplot(gs[1, 0], projection="polar")
    ax_rose2 = fig.add_subplot(gs[1, 1], projection="polar")
    
    steps = list(load_steps) if load_steps is not None else list(multiload_result.cases.keys())
    
    normal_colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    shear_colors = ["#ff7f0e", "#d62728", "#8c564b"]
    
    norm_i = 0
    shear_i = 0
    for ls in steps:
        if ls not in multiload_result.results:
            continue
        res = multiload_result.results[ls]
        case = multiload_result.cases[ls]
        
        if case.category == "normal":
            color = normal_colors[norm_i % len(normal_colors)]
            norm_i += 1
            ls_style = "-"
        else:
            color = shear_colors[shear_i % len(shear_colors)]
            shear_i += 1
            ls_style = "--"
            
        lbl = f"{case.label} ({case.category})"
        if res.ensemble_kde is not None and res.kde_eval_points is not None:
            ax_pdf.plot(res.kde_eval_points, res.ensemble_kde, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
        else:
            ax_pdf.plot(res.bin_centers, res.ensemble_pdf, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
            
        dx = np.diff(res.bin_edges)
        cdf = np.cumsum(res.ensemble_pdf * dx)
        ax_cdf.plot(res.bin_centers, cdf, color=color, linestyle=ls_style, linewidth=2.0, label=lbl)
        
    q_name = multiload_result.quantity
    q_label = r"\sigma_{\mathrm{VM}}" if q_name == "von_mises" else q_name
    ax_pdf.set_xlabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax_pdf.set_ylabel("Probability Density")
    ax_pdf.set_title(f"A. Probability Density $p({q_label} \\mid L)$ across Loading Modes")
    ax_pdf.legend(frameon=True, fontsize=9)
    ax_pdf.grid(True, linestyle=":", alpha=0.6)
    
    ax_cdf.set_xlabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax_cdf.set_ylabel(f"Cumulative Distribution $F({q_label} \\mid L)$")
    ax_cdf.set_title(f"B. Cumulative Probability $F({q_label} \\mid L)$")
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.legend(frameon=True, fontsize=9)
    ax_cdf.grid(True, linestyle=":", alpha=0.6)
    
    # Bottom rose diagrams: select one normal tension and one shear if available
    keys = list(directional_histograms.keys())
    norm_key = next((k for k in keys if "normal" in k.lower() or "eps_x" in k.lower()), keys[0] if keys else None)
    shear_key = next((k for k in keys if "shear" in k.lower() or "xy" in k.lower()), keys[1] if len(keys) > 1 else None)
    if norm_key is None and len(keys) >= 1:
        norm_key = keys[0]
    if shear_key is None and len(keys) >= 2:
        shear_key = keys[1]

    if norm_key and norm_key in directional_histograms:
        plot_directional_histogram(directional_histograms[norm_key], title=f"C. Directional Histogram: {norm_key}", color="#1f77b4", ax=ax_rose1)
    if shear_key and shear_key in directional_histograms:
        plot_directional_histogram(directional_histograms[shear_key], title=f"D. Directional Histogram: {shear_key}", color="#ff7f0e", ax=ax_rose2)
        
    plt.tight_layout()
    return fig



