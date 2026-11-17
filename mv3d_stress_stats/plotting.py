"""
Visualization utilities for stress statistics and ensemble analysis.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from mv3d_stress_stats.ensemble import EnsembleAnalysisResult
from mv3d_stress_stats.joint import JointDistribution2D
from mv3d_stress_stats.correlation import SpatialCorrelationResult
from mv3d_stress_stats.directional import (
    DirectionalHistogram,
    PairPlaneSectionResult,
    PairPlaneRaySectionResult,
    PairPlaneCartesianSectionResult,
)
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
    use_kde: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """Plot 2D joint density map with probability contour levels.
    
    Args:
        joint: JointDistribution2D instance.
        levels: Probability levels for contour superlevel sets (e.g. 0.50, 0.90, 0.95).
        use_kde: If True and joint.kde_density is available, plots the KDE density field.
        ax: Optional matplotlib axes.
        figsize: Figure size when creating a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    X, Y = np.meshgrid(joint.x_centers, joint.y_centers, indexing="ij")
    
    target_density = joint.kde_density if (use_kde and joint.kde_density is not None) else joint.density
    method_name = "KDE" if (use_kde and joint.kde_density is not None) else "Histogram"
    
    # Heatmap / pcolormesh
    pcm = ax.pcolormesh(
        joint.x_bins,
        joint.y_bins,
        target_density.T,
        shading="flat",
        cmap="viridis",
    )
    plt.colorbar(pcm, ax=ax, label=f"Probability Density ({method_name})")
    
    # Overlay contours
    if levels is not None and len(levels) > 0:
        c_vals = [joint.find_contour_level(p, use_kde=use_kde) for p in levels]
        c_vals_sorted = sorted(c_vals)
        cs = ax.contour(
            X, Y, target_density,
            levels=c_vals_sorted,
            colors="white",
            linewidths=1.5,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")
        
    ax.set_xlabel(joint.x_name)
    ax.set_ylabel(joint.y_name)
    title_suffix = " [Normalized]" if joint.is_normalized else ""
    ax.set_title(f"Joint Density ({method_name}): p({joint.x_name}, {joint.y_name}){title_suffix}")
    
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



def plot_directional_kde_field(
    multiload_result: MultiloadAnalysisResult,
    cut_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Visualize how p(Q | L) changes across loading directions.

    Requires multiload_result to come from analyze_multiload(..., kde=True,
    shared_range=True) so all directions share one Q grid.

    Panels:
        Left:  full field -- direction (categorical) x Q x density, as a heatmap.
        Right: explicit 1D cuts p(Q | L) at the requested directions
               (default: all of them), pulled directly from the same field.
    """
    labels, q_grid, field = multiload_result.directional_kde_field()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: full field as a heatmap, directions on x as categorical ticks.
    # 'nearest' shading (discrete blocks) is used deliberately: unlike a
    # continuous scatter-angle sweep, adjacent columns here (e.g. eps_x next
    # to eps_xy) are unrelated loading modes, so interpolating between them
    # would visually imply a continuity that doesn't exist.
    im = ax1.pcolormesh(
        np.arange(len(labels)), q_grid, field.T, shading="nearest", cmap="viridis"
    )
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    q_name = multiload_result.quantity
    ax1.set_ylabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax1.set_title(r"Full field: $p(Q \mid L)$ across directions")
    fig.colorbar(im, ax=ax1, label="density")

    # Right: explicit cuts
    cuts = list(cut_labels) if cut_labels is not None else labels
    cmap = plt.get_cmap("plasma")
    for j, lbl in enumerate(cuts):
        if lbl not in labels:
            continue
        i = labels.index(lbl)
        color = cmap(j / max(len(cuts) - 1, 1))
        ax2.plot(q_grid, field[i], color=color, linewidth=2.0, label=lbl)
    ax2.set_xlabel(f"{q_name.replace('_', ' ').title()} (Pa)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title(r"Cuts: $p(Q \mid L)$ at selected directions")
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


def plot_pair_plane_sections(
    section: PairPlaneSectionResult,
    density_type: str = "joint",
    cmap: str = "plasma",
    show_marginal: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (9, 5),
) -> plt.Axes:
    """Plot PDF sections in a pair plane with angle on x-axis and PDF on y-axis.
    
    Args:
        section: PairPlaneSectionResult containing angular sections across radial levels.
        density_type: 'joint' for joint PDF f(r, theta) (in Pa^-2 or normalized^-2),
                      or 'conditional' for conditional angular PDF p(theta | R=r) (in rad^-1).
        cmap: Matplotlib colormap for mapping stress radius levels.
        show_marginal: If True, overlays the overall angular marginal density p(theta).
        ax: Optional matplotlib axes.
        figsize: Figure dimensions if ax is None.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    densities = section.conditional_densities if density_type == "conditional" else section.section_densities
    n_radii = len(section.radii)
    
    # Colormap mapping for radial stress levels
    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=float(np.min(section.radii)), vmax=float(np.max(section.radii)))
    
    # Plot each radial section curve: x = angle, y = PDF value, color = radius
    for k in range(n_radii):
        r_val = section.radii[k]
        col = colormap(norm(r_val))
        r_unit = "" if section.is_normalized else " Pa"
        lbl = f"{section.r_labels[k]}: r={r_val:.2e}{r_unit}"
        ax.plot(
            section.angles_deg,
            densities[k, :],
            color=col,
            linewidth=2.0,
            label=lbl,
        )
        
    # Optional overlay of 1D marginal angular PDF p(theta)
    if show_marginal:
        if density_type == "conditional":
            ax.plot(
                section.angles_deg,
                section.angular_pdf,
                color="black",
                linestyle="--",
                linewidth=2.2,
                label=r"Marginal $p(\theta)$",
            )
        else:
            # Scale marginal peak height to match joint density scale for shape comparison
            max_sec = float(np.max(densities)) if np.max(densities) > 0 else 1.0
            max_ang = float(np.max(section.angular_pdf)) if np.max(section.angular_pdf) > 0 else 1.0
            scaled_ang = section.angular_pdf * (max_sec / max_ang)
            ax.plot(
                section.angles_deg,
                scaled_ang,
                color="black",
                linestyle="--",
                linewidth=1.8,
                alpha=0.75,
                label=r"Marginal $p(\theta)$ (scaled)",
            )
            
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    if section.is_centered:
        cbar_label = r"Fluctuation Radius $\Delta r = \sqrt{\Delta x^2 + \Delta y^2}$ (Pa)"
    elif section.is_normalized:
        cbar_label = r"Radius $r/R_{\mathrm{macro}}$"
    else:
        cbar_label = r"Radius $r = \sqrt{x^2 + y^2}$ (Pa)"
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Configure axes
    major_ticks = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    tick_labels = ["-180°", "-135°", "-90°", "-45°", "0°", "+45°", "+90°", "+135°", "+180°"]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-180, 180)
    
    c1, c2 = section.plane[0].upper(), section.plane[1].upper()
    delta_str = r"\Delta " if section.is_centered else ""
    ax.set_xlabel(
        rf"Fluctuation Angle $\theta = \arctan2({delta_str}{c2}, {delta_str}{c1})$"
        if section.is_centered
        else rf"Angle $\theta = \arctan2({c2}, {c1})$",
        fontsize=11,
    )
    
    if density_type == "conditional":
        ax.set_ylabel(r"Conditional Angular PDF $p(\theta \mid R=r)$ ($\mathrm{rad}^{-1}$)", fontsize=11)
    else:
        y_unit = "" if section.is_normalized else r" ($\mathrm{Pa}^{-2}$)"
        lbl_prefix = "Fluctuation Joint PDF" if section.is_centered else "Joint PDF"
        ax.set_ylabel(rf"{lbl_prefix} $f(r\cos\theta, r\sin\theta)${y_unit}", fontsize=11)
        
    title_prefix = "Fluctuation PDF Sections" if section.is_centered else "PDF Sections"
    title_suffix = " [Normalized]" if section.is_normalized else ""
    ax.set_title(f"{title_prefix} in Pair Plane {section.plane_label}{title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    
    return ax


def plot_pair_planes_grid(
    sections: Dict[str, PairPlaneSectionResult],
    density_type: str = "joint",
    cmap: str = "plasma",
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot multi-panel grid of angular PDF sections across component pair planes.
    
    Args:
        sections: Dictionary mapping plane labels to PairPlaneSectionResult objects
                  (e.g. from compute_all_pair_plane_sections).
        density_type: 'joint' or 'conditional'.
        cmap: Matplotlib colormap for radial section levels.
        ncols: Number of subplot columns (default: 3).
        figsize: Optional figure size.
        
    Returns:
        (Figure, 2D array of Axes).
    """
    keys = list(sections.keys())
    n_plots = len(keys)
    nrows = int(np.ceil(n_plots / ncols))
    
    if figsize is None:
        figsize = (5 * ncols, 3.8 * nrows)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    ax_flat = np.atleast_1d(axes).ravel()
    colormap = plt.get_cmap(cmap)
    
    for i, key in enumerate(keys):
        ax = ax_flat[i]
        sec = sections[key]
        densities = sec.conditional_densities if density_type == "conditional" else sec.section_densities
        n_radii = len(sec.radii)
        
        norm = mcolors.Normalize(vmin=float(np.min(sec.radii)), vmax=float(np.max(sec.radii)))
        for k in range(n_radii):
            col = colormap(norm(sec.radii[k]))
            ax.plot(sec.angles_deg, densities[k, :], color=col, linewidth=1.8, label=sec.r_labels[k])
            
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["-180°", "-90°", "0°", "+90°", "+180°"])
        ax.set_xlim(-180, 180)
        c1, c2 = sec.plane[0].upper(), sec.plane[1].upper()
        delta_str = r"\Delta " if sec.is_centered else ""
        ax.set_xlabel(
            rf"$\theta = \arctan2({delta_str}{c2}, {delta_str}{c1})$"
            if sec.is_centered
            else rf"$\theta = \arctan2({c2}, {c1})$",
            fontsize=10,
        )
        ax.set_ylabel("PDF Value", fontsize=10)
        title_prefix = "Centered " if sec.is_centered else "Plane "
        ax.set_title(f"{title_prefix}{sec.plane_label}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=7, loc="upper right")
        
    for j in range(n_plots, len(ax_flat)):
        ax_flat[j].set_visible(False)
        
    plt.tight_layout()
    return fig, axes


def plot_pair_planes_angular_comparison(
    sections: Dict[str, PairPlaneSectionResult],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    colormap: str = "tab10",
) -> plt.Axes:
    """Compare marginal angular distributions p(theta) across multiple pair planes in one figure.
    
    Args:
        sections: Dictionary of PairPlaneSectionResult objects.
        ax: Optional matplotlib axes.
        figsize: Figure size if ax is None.
        colormap: Discrete colormap for distinguishing planes.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    cmap = plt.get_cmap(colormap)
    keys = list(sections.keys())
    any_centered = any(sec.is_centered for sec in sections.values())
    delta_str = r"\Delta " if any_centered else ""
    
    for i, key in enumerate(keys):
        sec = sections[key]
        col = cmap(i % 10)
        ax.plot(sec.angles_deg, sec.angular_pdf, color=col, linewidth=2.0, label=sec.plane_label)
        ax.fill_between(sec.angles_deg, 0, sec.angular_pdf, color=col, alpha=0.08)
        
    major_ticks = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    tick_labels = ["-180°", "-135°", "-90°", "-45°", "0°", "+45°", "+90°", "+135°", "+180°"]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-180, 180)
    ax.set_xlabel(rf"Orientation Angle $\theta = \arctan2({delta_str}y, {delta_str}x)$", fontsize=11)
    ax.set_ylabel(r"Angular PDF $p(\theta)$ ($\mathrm{rad}^{-1}$)", fontsize=11)
    title_suffix = " [Centered]" if any_centered else ""
    ax.set_title(f"Angular Probability Density Comparison Across Pair Planes{title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    
    return ax


def plot_pair_plane_polar_heatmap(
    section: PairPlaneSectionResult,
    cmap: str = "viridis",
    show_section_lines: bool = True,
    show_mean_line: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (9, 5.5),
) -> plt.Axes:
    """Plot 2D unrolled polar heatmap: angle on x-axis, radius on y-axis, color = PDF value.
    
    Shows the continuous probability density landscape f(r, theta) across polar coordinates.
    When centered (default), displays horizontal lines across all angle intervals [-180, 180]
    corresponding to discrete section radii (e.g. q10, q25, q50, q75, q90) and the mean radius.
    
    Args:
        section: PairPlaneSectionResult instance.
        cmap: Colormap for PDF density surface.
        show_section_lines: If True, draws horizontal dashed reference lines across
                            all angles at the discrete section radii r_k.
        show_mean_line: If True, draws a horizontal solid reference line across all angles
                        at the mean radius r_mean.
        ax: Optional matplotlib axes.
        figsize: Figure size if ax is None.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    c1, c2 = section.plane[0].upper(), section.plane[1].upper()
    delta_str = r"\Delta " if section.is_centered else ""
    
    # Choose grid: use high-resolution polar_grid if available, else discrete section_densities
    if section.polar_grid_density is not None and section.polar_grid_radii is not None:
        Theta, R = np.meshgrid(section.angles_deg, section.polar_grid_radii)
        Z = section.polar_grid_density
        r_max = float(section.polar_grid_radii[-1])
        r_min = float(section.polar_grid_radii[0])
    else:
        Theta, R = np.meshgrid(section.angles_deg, section.radii)
        Z = section.section_densities
        r_max = float(section.radii[-1])
        r_min = float(section.radii[0])
        
    pcm = ax.pcolormesh(Theta, R, Z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar_unit = "" if section.is_normalized else r" ($\mathrm{Pa}^{-2}$)"
    cbar_label = f"Joint Fluctuation PDF{cbar_unit}" if section.is_centered else f"Joint PDF{cbar_unit}"
    cbar.set_label(cbar_label, fontsize=10)
    
    r_unit = "" if section.is_normalized else " Pa"
    
    # Draw horizontal dashed lines across all angle intervals at section radii
    if show_section_lines and section.radii is not None:
        for r_k, lbl in zip(section.radii, section.r_labels):
            ax.axhline(
                r_k,
                color="white",
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
                label=f"{lbl}: {r_k:.2e}{r_unit}",
            )
            
    # Draw horizontal solid line across all angle intervals at mean radius
    if show_mean_line and section.r_mean is not None:
        ax.axhline(
            section.r_mean,
            color="cyan",
            linestyle="-",
            linewidth=1.8,
            alpha=0.85,
            label=f"Mean: {section.r_mean:.2e}{r_unit}",
        )
        
    if (show_section_lines and section.radii is not None) or (show_mean_line and section.r_mean is not None):
        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=8,
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
        )
        
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xticklabels(["-180°", "-135°", "-90°", "-45°", "0°", "+45°", "+90°", "+135°", "+180°"])
    ax.set_xlim(-180, 180)
    ax.set_ylim(r_min, r_max)
    
    ax.set_xlabel(
        rf"Fluctuation Angle $\theta = \arctan2({delta_str}{c2}, {delta_str}{c1})$"
        if section.is_centered
        else rf"Angle $\theta = \arctan2({c2}, {c1})$",
        fontsize=11,
    )
    
    r_unit_axis = "" if section.is_normalized else " (Pa)"
    if section.is_centered:
        r_axis_lbl = rf"Fluctuation Radius $\Delta r = \sqrt{{({delta_str}{c1})^2 + ({delta_str}{c2})^2}}${r_unit_axis}"
    else:
        r_axis_lbl = rf"Stress Radius $r = \sqrt{{{c1}^2 + {c2}^2}}${r_unit_axis}"
    ax.set_ylabel(r_axis_lbl, fontsize=11)
    
    title_prefix = "Polar Fluctuation PDF Density Landscape: Centered" if section.is_centered else "Polar PDF Density Landscape"
    title_suffix = " [Normalized]" if section.is_normalized else ""
    ax.set_title(f"{title_prefix} {section.plane_label}{title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.3, color="white")
    
    return ax


def plot_pair_plane_ray_sections(
    ray_section: PairPlaneRaySectionResult,
    cmap: str = "tab10",
    show_peaks: bool = True,
    show_macro_line: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (9, 5.5),
) -> plt.Axes:
    """Plot PDF curves along radial ray sections at fixed angles in a stress pair plane.
    
    Plots f(r*cos(theta), r*sin(theta)) vs r for fixed angles theta (e.g. 0°, 30°, 45°, 60°, 90°).
    
    Args:
        ray_section: PairPlaneRaySectionResult instance.
        cmap: Colormap for distinguishing different ray angles (default: 'tab10').
        show_peaks: If True, marks the peak location along each ray with a scatter marker.
        show_macro_line: If True, draws a vertical line at the macroscopic radius.
        ax: Optional matplotlib axes.
        figsize: Figure dimensions if ax is None.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colormap = plt.get_cmap(cmap)
    n_rays = len(ray_section.angles_deg)
    
    for m in range(n_rays):
        col = colormap(m % 10)
        lbl = ray_section.angle_labels[m]
        ax.plot(
            ray_section.radii,
            ray_section.ray_densities[m, :],
            color=col,
            linewidth=2.2,
            label=lbl,
        )
        if show_peaks and ray_section.peak_radii is not None and ray_section.peak_densities is not None:
            r_pk = ray_section.peak_radii[m]
            d_pk = ray_section.peak_densities[m]
            if d_pk > 0:
                ax.plot(r_pk, d_pk, "o", color=col, markersize=5.5)
                
    if show_macro_line:
        if ray_section.is_normalized:
            ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.4, alpha=0.85, label=r"Macro Radius $R_{\mathrm{macro}} = 1.0$")
        elif ray_section.macro_radius is not None:
            ax.axvline(ray_section.macro_radius, color="gray", linestyle=":", linewidth=1.4, alpha=0.85, label=f"Macro R: {ray_section.macro_radius:.2e} Pa")
            
    c1, c2 = ray_section.plane[0].upper(), ray_section.plane[1].upper()
    delta_str = r"\Delta " if ray_section.is_centered else ""
    
    if ray_section.is_normalized:
        x_label = rf"Normalized Stress Radius $r / R_{{\mathrm{{macro}}}} = \sqrt{{{delta_str}\tilde{{{c1}}}^2 + {delta_str}\tilde{{{c2}}}^2}}$ (dimensionless)"
        y_label = r"Joint PDF $f(r\cos\theta, r\sin\theta)$"
    else:
        x_label = rf"Stress Radius $r = \sqrt{{{delta_str}{c1}^2 + {delta_str}{c2}^2}}$ (Pa)"
        y_label = r"Joint PDF $f(r\cos\theta, r\sin\theta)$ ($\mathrm{Pa}^{-2}$)"
        
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    
    norm_str = " [Normalized]" if ray_section.is_normalized else ""
    prefix = "Fluctuation " if ray_section.is_centered else ""
    ax.set_title(f"{prefix}Ray Sections in Pair Plane {ray_section.plane_label} at Fixed Angles{norm_str}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(left=0.0)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    
    return ax


def plot_pair_planes_ray_grid(
    ray_sections: Dict[str, PairPlaneRaySectionResult],
    ncols: int = 3,
    cmap: str = "tab10",
    show_peaks: bool = True,
    show_macro_line: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot multi-panel grid of radial ray sections across component pair planes.
    
    Args:
        ray_sections: Dict mapping plane labels to PairPlaneRaySectionResult objects.
        ncols: Number of columns (default: 3).
        cmap: Colormap for ray angles.
        show_peaks: If True, marks peaks.
        show_macro_line: If True, draws macro reference line.
        figsize: Figure dimensions.
        
    Returns:
        (Figure, 2D array of Axes).
    """
    keys = list(ray_sections.keys())
    n_plots = len(keys)
    nrows = int(np.ceil(n_plots / ncols))
    
    if figsize is None:
        figsize = (5.2 * ncols, 3.8 * nrows)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    ax_flat = np.atleast_1d(axes).ravel()
    colormap = plt.get_cmap(cmap)
    
    for i, key in enumerate(keys):
        ax = ax_flat[i]
        res = ray_sections[key]
        n_rays = len(res.angles_deg)
        
        for m in range(n_rays):
            col = colormap(m % 10)
            ax.plot(res.radii, res.ray_densities[m, :], color=col, linewidth=1.8, label=res.angle_labels[m])
            if show_peaks and res.peak_radii is not None and res.peak_densities is not None:
                r_pk = res.peak_radii[m]
                d_pk = res.peak_densities[m]
                if d_pk > 0:
                    ax.plot(r_pk, d_pk, "o", color=col, markersize=4.5)
                    
        if show_macro_line:
            if res.is_normalized:
                ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label="Macro (1.0)")
            elif res.macro_radius is not None:
                ax.axvline(res.macro_radius, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
                
        c1, c2 = res.plane[0].upper(), res.plane[1].upper()
        delta_str = r"\Delta " if res.is_centered else ""
        if res.is_normalized:
            ax.set_xlabel(rf"$r / R_{{\mathrm{{macro}}}}$ ({delta_str}{c1}, {delta_str}{c2})", fontsize=10)
        else:
            ax.set_xlabel(rf"Radius $r$ (Pa)", fontsize=10)
        ax.set_ylabel("PDF Value", fontsize=10)
        ax.set_title(f"Plane {res.plane_label}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(left=0.0)
        ax.legend(frameon=True, fontsize=7, loc="upper right")
        
    for j in range(n_plots, len(ax_flat)):
        ax_flat[j].set_visible(False)
        
    plt.tight_layout()
    return fig, axes


def plot_pair_plane_ray_dashboard(
    ray_section: PairPlaneRaySectionResult,
    cmap_rays: str = "tab10",
    figsize: Tuple[int, int] = (13, 5.5),
) -> plt.Figure:
    """Two-panel dashboard: 2D plane with radial rays (left) and 1D ray PDF profiles (right).
    
    Shows the geometric configuration of rays cutting through the 2D stress space alongside
    the exact probability density evaluated along each cut.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    colormap = plt.get_cmap(cmap_rays)
    n_rays = len(ray_section.angles_deg)
    
    c1, c2 = ray_section.plane[0].upper(), ray_section.plane[1].upper()
    delta_str = r"\Delta " if ray_section.is_centered else ""
    r_max = float(ray_section.radii[-1])
    
    # Left panel: Ray lines in (X, Y) space
    th_arc = np.linspace(0, np.pi/2, 100)
    for r_ref in [0.5, 1.0, 1.5]:
        if r_ref <= r_max:
            ax_left.plot(r_ref * np.cos(th_arc), r_ref * np.sin(th_arc), color="lightgray", linestyle="--", linewidth=1.0)
            ax_left.text(r_ref * np.cos(np.radians(15)), r_ref * np.sin(np.radians(15)), f"r={r_ref:g}", color="gray", fontsize=8)
            
    for m in range(n_rays):
        col = colormap(m % 10)
        rad = ray_section.angles_rad[m]
        # Ray line from 0 to r_max
        ax_left.plot([0, r_max * np.cos(rad)], [0, r_max * np.sin(rad)], color=col, linewidth=2.4, label=ray_section.angle_labels[m])
        # Mark peak position along ray
        if ray_section.peak_radii is not None and ray_section.peak_densities is not None:
            r_pk = ray_section.peak_radii[m]
            if ray_section.peak_densities[m] > 0:
                ax_left.plot(r_pk * np.cos(rad), r_pk * np.sin(rad), "o", color=col, markersize=6.5)
                
    # Mark macro state if available
    if ray_section.macro_center is not None:
        mx, my = ray_section.macro_center
        if ray_section.is_normalized and ray_section.macro_radius and ray_section.macro_radius > 0:
            mx_plot, my_plot = mx / ray_section.macro_radius, my / ray_section.macro_radius
        else:
            mx_plot, my_plot = mx, my
        ax_left.plot(mx_plot, my_plot, marker="*", markersize=14, color="crimson", markeredgecolor="black", label=r"Macro Mean $(\bar{X}, \bar{Y})$")
        
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.set_xlim(-0.05 * r_max, r_max * 1.05)
    ax_left.set_ylim(-0.05 * r_max, r_max * 1.05)
    
    if ray_section.is_normalized:
        ax_left.set_xlabel(rf"Normalized {delta_str}{c1} ($x / R_{{\mathrm{{macro}}}}$)", fontsize=11)
        ax_left.set_ylabel(rf"Normalized {delta_str}{c2} ($y / R_{{\mathrm{{macro}}}}$)", fontsize=11)
    else:
        ax_left.set_xlabel(rf"{delta_str}{c1} (Pa)", fontsize=11)
        ax_left.set_ylabel(rf"{delta_str}{c2} (Pa)", fontsize=11)
        
    ax_left.set_title(f"Ray Geometry in Plane {ray_section.plane_label}", fontsize=12, fontweight="bold")
    ax_left.grid(True, linestyle=":", alpha=0.5)
    ax_left.legend(frameon=True, fontsize=8, loc="upper right")
    
    # Right panel: 1D Ray Profiles
    plot_pair_plane_ray_sections(
        ray_section,
        cmap=cmap_rays,
        show_peaks=True,
        show_macro_line=True,
        ax=ax_right,
    )
    
    plt.tight_layout()
    return fig


def plot_pair_plane_cartesian_sections(
    cart_section: PairPlaneCartesianSectionResult,
    density_type: str = "joint",
    cmap: str = "plasma",
    show_peaks: bool = True,
    show_macro_line: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (9, 5.5),
) -> plt.Axes:
    """Plot parallel Cartesian cross-section profiles through the 2D stress PDF landscape.
    
    Args:
        cart_section: PairPlaneCartesianSectionResult instance.
        density_type: 'joint' for joint PDF f(x, y_k) or 'conditional' for p(x | y_k).
        cmap: Colormap for slice levels.
        show_peaks: If True, marks peak positions.
        show_macro_line: If True, draws macro mean line for profile axis.
        ax: Optional matplotlib axes.
        figsize: Figure dimensions if ax is None.
        
    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    densities = (
        cart_section.conditional_densities
        if density_type == "conditional"
        else cart_section.section_densities
    )
    n_slices = len(cart_section.slice_levels)
    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(
        vmin=float(np.min(cart_section.slice_levels)),
        vmax=float(np.max(cart_section.slice_levels)),
    )

    for k in range(n_slices):
        lvl = cart_section.slice_levels[k]
        col = colormap(norm(lvl))
        lbl = cart_section.slice_labels[k]
        is_median = "q50" in lbl
        lw = 2.6 if is_median else 1.8
        ax.plot(
            cart_section.profile_grid,
            densities[k, :],
            color=col,
            linewidth=lw,
            label=lbl,
        )
        if show_peaks and cart_section.peak_densities[k] > 0:
            pk_pos = cart_section.peak_positions[k]
            pk_val = (
                np.max(densities[k, :])
                if density_type == "conditional"
                else cart_section.peak_densities[k]
            )
            ax.plot(pk_pos, pk_val, "o", color=col, markersize=5.0)

    if show_macro_line and cart_section.macro_center is not None:
        mx, my = cart_section.macro_center
        c_x, c_y = cart_section.plane
        macro_val = mx if cart_section.profile_component == c_x.upper() else my
        if cart_section.is_normalized and cart_section.macro_radius and cart_section.macro_radius > 0:
            macro_val = macro_val / cart_section.macro_radius
        val_str = f"{macro_val:.3f}" if cart_section.is_normalized else f"{macro_val:.2e} Pa"
        ax.axvline(
            macro_val,
            color="gray",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label=rf"Macro $\bar{{{cart_section.profile_component}}} = {val_str}$",
        )

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar_unit = "" if cart_section.is_normalized else " (Pa)"
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"Slice Level {cart_section.slice_component}{cbar_unit}", fontsize=10)

    x_unit = "" if cart_section.is_normalized else " (Pa)"
    norm_txt = " [Normalized]" if cart_section.is_normalized else ""
    x_lbl = (
        rf"Normalized {cart_section.profile_component} (${cart_section.profile_component.lower()} / R_{{\mathrm{{macro}}}}$)"
        if cart_section.is_normalized
        else rf"{cart_section.profile_component}{x_unit}"
    )
    ax.set_xlabel(x_lbl, fontsize=11)

    if density_type == "conditional":
        ax.set_ylabel(rf"Conditional PDF $p({cart_section.profile_component} \mid {cart_section.slice_component}=lvl)$", fontsize=11)
        type_str = "Conditional"
    else:
        y_unit = "" if cart_section.is_normalized else r" ($\mathrm{Pa}^{-2}$)"
        ax.set_ylabel(rf"Joint PDF $f({cart_section.profile_component}, {cart_section.slice_component}=lvl)${y_unit}", fontsize=11)
        type_str = "Joint"

    ax.set_title(
        f"{type_str} Cartesian Sections in Plane {cart_section.plane_label} Sliced Along {cart_section.slice_component}{norm_txt}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    return ax


def plot_pair_planes_cartesian_grid(
    cart_sections: Dict[str, PairPlaneCartesianSectionResult],
    density_type: str = "joint",
    ncols: int = 3,
    cmap: str = "plasma",
    show_peaks: bool = True,
    show_macro_line: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot multi-panel grid of parallel Cartesian cross-sections across component pair planes.
    
    Args:
        cart_sections: Dict mapping plane labels to PairPlaneCartesianSectionResult.
        density_type: 'joint' or 'conditional'.
        ncols: Number of columns (default: 3).
        cmap: Colormap for slice levels.
        show_peaks: If True, marks peaks.
        show_macro_line: If True, draws macro reference line.
        figsize: Figure dimensions.
        
    Returns:
        (Figure, 2D array of Axes).
    """
    keys = list(cart_sections.keys())
    n_plots = len(keys)
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        figsize = (5.2 * ncols, 3.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    ax_flat = np.atleast_1d(axes).ravel()
    colormap = plt.get_cmap(cmap)

    for i, key in enumerate(keys):
        ax = ax_flat[i]
        sec = cart_sections[key]
        densities = (
            sec.conditional_densities
            if density_type == "conditional"
            else sec.section_densities
        )
        n_slices = len(sec.slice_levels)
        norm = mcolors.Normalize(
            vmin=float(np.min(sec.slice_levels)),
            vmax=float(np.max(sec.slice_levels)),
        )

        for k in range(n_slices):
            lvl = sec.slice_levels[k]
            col = colormap(norm(lvl))
            is_median = "q50" in sec.slice_labels[k]
            lw = 2.2 if is_median else 1.5
            ax.plot(
                sec.profile_grid,
                densities[k, :],
                color=col,
                linewidth=lw,
                label=sec.slice_labels[k],
            )
            if show_peaks and sec.peak_densities[k] > 0:
                pk_val = (
                    np.max(densities[k, :])
                    if density_type == "conditional"
                    else sec.peak_densities[k]
                )
                ax.plot(sec.peak_positions[k], pk_val, "o", color=col, markersize=4.0)

        if show_macro_line and sec.macro_center is not None:
            mx, my = sec.macro_center
            c_x, _ = sec.plane
            macro_val = mx if sec.profile_component == c_x.upper() else my
            if sec.is_normalized and sec.macro_radius and sec.macro_radius > 0:
                macro_val = macro_val / sec.macro_radius
            ax.axvline(macro_val, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        norm_unit = "" if sec.is_normalized else " (Pa)"
        ax.set_xlabel(f"{sec.profile_component}{norm_unit}", fontsize=10)
        ax.set_ylabel("PDF Value", fontsize=10)
        ax.set_title(f"Plane {sec.plane_label} [Cut: {sec.slice_component}]", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=7, loc="upper right")

    for j in range(n_plots, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_pair_plane_cartesian_dashboard(
    cart_section: PairPlaneCartesianSectionResult,
    density_type: str = "joint",
    cmap_surface: str = "Blues",
    cmap_slices: str = "plasma",
    figsize: Tuple[int, int] = (14, 5.5),
) -> plt.Figure:
    """Two-panel dashboard: 2D joint density map with parallel cut lines (left) and 1D profiles (right).
    
    Directly visualizes the 2D stress probability mountain alongside the 1D elevation profiles
    at each parallel cross-section level.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    colormap = plt.get_cmap(cmap_slices)
    norm = mcolors.Normalize(
        vmin=float(np.min(cart_section.slice_levels)),
        vmax=float(np.max(cart_section.slice_levels)),
    )

    c_x, c_y = cart_section.plane[0].upper(), cart_section.plane[1].upper()

    # Left panel: 2D contour map + horizontal/vertical slice lines
    if (
        cart_section.grid_x is not None
        and cart_section.grid_y is not None
        and cart_section.density_surface is not None
    ):
        Gx, Gy = np.meshgrid(cart_section.grid_x, cart_section.grid_y)
        cf = ax_left.contourf(Gx, Gy, cart_section.density_surface, levels=25, cmap=cmap_surface)
        ax_left.contour(Gx, Gy, cart_section.density_surface, levels=8, colors="gray", alpha=0.35, linewidths=0.7)
        cbar_left = fig.colorbar(cf, ax=ax_left, pad=0.02)
        cbar_left.set_label("2D Density f(X, Y)", fontsize=10)

    # Draw parallel slice lines across the 2D plane
    p_min = float(cart_section.profile_grid[0])
    p_max = float(cart_section.profile_grid[-1])

    for k, lvl in enumerate(cart_section.slice_levels):
        col = colormap(norm(lvl))
        lbl = cart_section.slice_labels[k]
        if cart_section.slice_axis == "y":
            ax_left.plot([p_min, p_max], [lvl, lvl], color=col, linewidth=2.0, label=lbl)
            ax_left.plot(cart_section.peak_positions[k], lvl, "o", color=col, markersize=6.0)
        else:
            ax_left.plot([lvl, lvl], [p_min, p_max], color=col, linewidth=2.0, label=lbl)
            ax_left.plot(lvl, cart_section.peak_positions[k], "o", color=col, markersize=6.0)

    if cart_section.macro_center is not None:
        mx, my = cart_section.macro_center
        if cart_section.is_normalized and cart_section.macro_radius and cart_section.macro_radius > 0:
            mx, my = mx / cart_section.macro_radius, my / cart_section.macro_radius
        ax_left.plot(
            mx,
            my,
            marker="*",
            markersize=14,
            color="crimson",
            markeredgecolor="black",
            label=r"Macro Mean $(\bar{X}, \bar{Y})$",
        )

    norm_txt = " [Normalized]" if cart_section.is_normalized else ""
    ax_left.set_xlabel(f"Normalized {c_x}" if cart_section.is_normalized else f"{c_x} (Pa)", fontsize=11)
    ax_left.set_ylabel(f"Normalized {c_y}" if cart_section.is_normalized else f"{c_y} (Pa)", fontsize=11)
    ax_left.set_title(f"2D PDF Landscape & Parallel Cut Lines {cart_section.plane_label}{norm_txt}", fontsize=12, fontweight="bold")
    ax_left.grid(True, linestyle=":", alpha=0.5)
    ax_left.legend(frameon=True, fontsize=8, loc="upper right")

    # Right panel: 1D Profiles
    plot_pair_plane_cartesian_sections(
        cart_section,
        density_type=density_type,
        cmap=cmap_slices,
        show_peaks=True,
        show_macro_line=True,
        ax=ax_right,
    )

    plt.tight_layout()
    return fig







# ---------------------------------------------------------------------------
# Node-level distributions reconstructed by superposition (localization.py)
# ---------------------------------------------------------------------------
def plot_node_distribution(
    nd,
    ax: Optional[plt.Axes] = None,
    show_individual: bool = True,
    show_cdf: bool = True,
    reference_values: Optional[np.ndarray] = None,
    reference_label: str = "FE reference",
    unit_scale: float = 1e-6,
    unit_label: str = "MPa",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Axes:
    """Plot p(Q | node) from LocalizationBasis.node_distribution().

    Gray: per-realization p(Q | R=r, node); red: realization-weighted mixture;
    dotted: quantiles q05/q50/q95; green dash-dot: Q of the homogenized macro
    stress; optional black steps: an independent reference sample (e.g. the FE
    stress field of a combined load step) histogrammed on the same bins.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    x = nd.bin_centers * unit_scale
    dens = 1.0 / unit_scale
    if show_individual:
        for i, h in enumerate(nd.rve_pdfs):
            ax.plot(x, h * dens, color="gray", alpha=0.18, lw=0.8,
                    label="single RVE $p(Q|R=r)$" if i == 0 else None)
    ax.plot(x, nd.pdf * dens, color="crimson", lw=2.2, label="ensemble $p(Q|node)$")
    if reference_values is not None:
        ref = np.asarray(reference_values).ravel()
        h, _ = np.histogram(ref, bins=nd.bin_edges)
        h = h / (ref.size * np.diff(nd.bin_edges))
        ax.step(x, h * dens, where="mid", color="black", lw=1.2, ls="--", label=reference_label)
    qs = nd.quantiles([0.05, 0.5, 0.95])
    for qv, lab in zip(qs, ["q05", "q50", "q95"]):
        ax.axvline(qv * unit_scale, color="navy", ls=":", lw=1.0)
        ax.text(qv * unit_scale, ax.get_ylim()[1] * 0.97, lab, color="navy",
                ha="center", va="top", fontsize=8)
    qm = nd.stats.get("q_of_macro_stress")
    if qm is not None:
        ax.axvline(qm * unit_scale, color="green", ls="-.", lw=1.2, label="Q(macro stress)")
    ax.set_xlabel(f"{nd.quantity} [{unit_label}]")
    ax.set_ylabel(f"probability density [1/{unit_label}]")
    ax.grid(True, ls=":", alpha=0.6)
    if show_cdf:
        ax2 = ax.twinx()
        ax2.plot(nd.bin_edges[1:] * unit_scale, nd.cdf, color="darkorange", lw=1.2, label="CDF")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("CDF", color="darkorange")
    eps = nd.macro_strain
    ax.set_title(title or (
        "macro strain [x,y,z,γxy,γyz,γxz] = [" + ", ".join(f"{e:.2e}" for e in eps) + "]"
    ), fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    return ax


def plot_superposition_validation(
    basis,
    file_path: str,
    load_steps: Sequence[int],
    quantities: Sequence[str] = ("von_mises", "principal_1", "hydrostatic_stress"),
    bins: int = 60,
    figsize_per_cell: Tuple[float, float] = (4.2, 3.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """Grid: rows = FE load steps, cols = quantities. Each cell overlays the FE
    distribution of that load step with the distribution reconstructed from the
    six unit cases (they should coincide), and the old scalar-rescaling
    estimate eta*Q_macro taken from the first active unit direction."""
    import h5py
    from mv3d_stress_stats.invariants import resolve_quantity

    sets = basis.metadata["set_ids"]
    factor = 2.0 if basis.metadata.get("shear_in_file") == "tensor" else 1.0
    unit_steps = basis.metadata["unit_load_steps"]
    fig, axes = plt.subplots(len(load_steps), len(quantities), squeeze=False,
                             figsize=(figsize_per_cell[0] * len(quantities),
                                      figsize_per_cell[1] * len(load_steps)))
    with h5py.File(file_path, "r") as f:
        for i, ls in enumerate(load_steps):
            eps = f[f"{sets[0]}/ls_{ls}/eps_as_loading"][:].astype(np.float64)
            eps[3:] *= factor
            fe = [f[f"{s}/ls_{ls}/results"][:, 7:13].astype(np.float64) for s in sets]
            k0 = int(np.flatnonzero(eps)[0])
            base = [f[f"{s}/ls_{unit_steps[k0]}/results"][:, 7:13].astype(np.float64) for s in sets]
            for j, qn in enumerate(quantities):
                q = resolve_quantity(qn)
                truth = np.concatenate([q(S) for S in fe])
                nd = basis.node_distribution(qn, macro_strain=eps, bins=bins)
                eta = np.concatenate([q(S) / q(S.mean(0)) for S in base])
                q_macro = np.mean([q(S.mean(0)) for S in fe])
                ax = axes[i, j]
                x = nd.bin_centers * 1e-6
                h_fe, _ = np.histogram(truth, bins=nd.bin_edges, density=True)
                ax.fill_between(x, h_fe * 1e6, step="mid", color="lightgray", label="FE (ANSYS)")
                ax.plot(x, nd.pdf * 1e6, color="crimson", lw=1.8, label="superposition")
                resc = eta * q_macro
                h_r, _ = np.histogram(resc, bins=nd.bin_edges)
                ax.plot(x, h_r / (resc.size * np.diff(nd.bin_edges)) * 1e6, color="navy",
                        ls="--", lw=1.2, label="scalar rescale (old)")
                lbl = "+".join(n for n, e in zip(["εx", "εy", "εz", "γxy", "γyz", "γxz"], eps) if e)
                ax.set_title(f"ls_{ls} ({lbl}) — {qn}", fontsize=9)
                ax.tick_params(labelsize=7)
                if i == len(load_steps) - 1:
                    ax.set_xlabel("MPa", fontsize=8)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------
# Pair planes of the local stress at a macro node (localization basis)
# --------------------------------------------------------------------------
STRESS_COMPONENT_LABELS = ["σxx", "σyy", "σzz", "σxy", "σyz", "σxz"]
ALL_STRESS_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]  # 15 pairs


def _hdr_levels(density: np.ndarray, cell_area: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    """Density thresholds c_p with  integral_{f >= c_p} f dA = p  (highest-density regions)."""
    f = density.ravel()
    order = np.argsort(f)[::-1]
    mass = np.cumsum(f[order] * cell_area.ravel()[order])
    mass /= mass[-1] if mass[-1] > 0 else 1.0
    return np.array([f[order][min(np.searchsorted(mass, p), f.size - 1)] for p in probs])


def _nice_step(raw: float) -> float:
    """1-2-5 rounded tick step >= raw."""
    e = 10.0 ** np.floor(np.log10(raw))
    return float(next(m * e for m in (1, 2, 5, 10) if m * e >= raw))


def _weighted_hist2d(x, y, w, xe, ye, smooth):
    h, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w)
    if smooth:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, smooth, mode="constant")
    area = np.outer(np.diff(xe), np.diff(ye))
    h = h / max(h.sum(), 1e-300) / area
    return h, area


def plot_node_pair_planes(
    basis,
    macro_strain: Optional[np.ndarray] = None,
    macro_stress: Optional[np.ndarray] = None,
    pairs: Sequence[Tuple[int, int]] = ALL_STRESS_PAIRS,
    fe_stress: Optional[Sequence[np.ndarray]] = None,
    rve_contours: int = 0,
    probs: Sequence[float] = (0.5, 0.9, 0.99),
    bins: int = 80,
    smooth: float = 1.0,
    ncols: int = 5,
    scale: float = 1e-6,
    unit_label: str = "MPa",
    title: Optional[str] = None,
    cell_size: Tuple[float, float] = (3.4, 3.4),
    common_scale: bool = True,
):
    """Joint densities p(σi, σj | node) of local stress components.

    Each panel: ensemble mixture density sum_r w_r p(σi, σj | R=r) as a shaded
    image (common bins, sample weight w_r / N_r), highest-density-region
    contours enclosing ``probs`` of the probability (crimson), the macro
    stress C_eff·ε (green star) and the ensemble mean (blue ×).

    fe_stress: optional list of (N, 6) FE stress fields of the same load case,
        one per realization in ``basis`` order — drawn as black dashed HDR
        contours on the same bins (they should coincide with the crimson ones).
    rve_contours: draw the outer HDR contour of this many single realizations
        (thin gray) to show realization-to-realization variability.
    smooth: Gaussian smoothing (in bins) applied identically to every density
        before contouring; 0 disables it.
    common_scale: every axis of every panel spans the same width (centered on
        the ensemble mean of that component) with equal aspect ratio, so the
        shapes of the clouds are directly comparable between pairs.
    """
    eps = basis._resolve_strain(macro_strain, macro_stress)
    sig = basis.local_stress(macro_strain=eps)                    # (M, N, 6)
    m, n = sig.shape[:2]
    w = np.repeat(basis.weights / n, n)
    flat = sig.reshape(-1, 6) * scale
    s_macro = basis.effective_stiffness() @ eps * scale
    s_mean = w @ flat
    fe_flat = fe_w = None
    if fe_stress is not None:
        if len(fe_stress) != m:
            raise ValueError("fe_stress must hold one field per realization of the basis")
        fe_flat = np.concatenate([np.asarray(s, float) for s in fe_stress]) * scale
        fe_w = np.concatenate([np.full(len(s), wr / len(s)) for s, wr in zip(fe_stress, basis.weights)])

    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)

    used = sorted({c for p in pairs for c in p})
    q_lo, q_hi = np.quantile(flat[:, used], [0.0005, 0.9995], axis=0)
    half = 1.12 * np.max(np.maximum(s_mean[used] - q_lo, q_hi - s_mean[used]))

    for k, (i, j) in enumerate(pairs):
        ax = axes.ravel()[k]
        x, y = flat[:, i], flat[:, j]
        if common_scale:
            xe = np.linspace(s_mean[i] - half, s_mean[i] + half, bins + 1)
            ye = np.linspace(s_mean[j] - half, s_mean[j] + half, bins + 1)
        else:
            lo_x, hi_x = np.quantile(x, [0.0005, 0.9995])
            lo_y, hi_y = np.quantile(y, [0.0005, 0.9995])
            pad_x = max(0.12 * (hi_x - lo_x), 1e-9)
            pad_y = max(0.12 * (hi_y - lo_y), 1e-9)
            xe = np.linspace(lo_x - pad_x, hi_x + pad_x, bins + 1)
            ye = np.linspace(lo_y - pad_y, hi_y + pad_y, bins + 1)
        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])

        h, area = _weighted_hist2d(x, y, w, xe, ye, smooth)
        ax.pcolormesh(xe, ye, h.T, cmap="Greys", shading="flat", rasterized=True)
        lv = np.unique(_hdr_levels(h, area, probs))
        ax.contour(xc, yc, h.T, levels=lv, colors="crimson", linewidths=1.3)

        if rve_contours:
            for r in np.linspace(0, m - 1, min(rve_contours, m)).astype(int):
                sr = sig[r] * scale
                hr, ar = _weighted_hist2d(sr[:, i], sr[:, j], None, xe, ye, smooth)
                ax.contour(xc, yc, hr.T, levels=_hdr_levels(hr, ar, [probs[-2] if len(probs) > 1 else probs[0]]),
                           colors="gray", linewidths=0.5, alpha=0.6)

        if fe_flat is not None:
            hf, af = _weighted_hist2d(fe_flat[:, i], fe_flat[:, j], fe_w, xe, ye, smooth)
            ax.contour(xc, yc, hf.T, levels=np.unique(_hdr_levels(hf, af, probs)),
                       colors="k", linewidths=1.0, linestyles="--")

        # mean of the local field == C_eff·eps exactly; hollow circle keeps both visible
        ax.plot(*s_mean[[i, j]], marker="o", mfc="none", mec="royalblue", ms=13, mew=1.8,
                ls="none", zorder=5)
        ax.plot(*s_macro[[i, j]], marker="*", color="limegreen", mec="k", ms=10, ls="none",
                zorder=6)
        ax.set_xlabel(f"{STRESS_COMPONENT_LABELS[i]} [{unit_label}]", fontsize=8)
        ax.set_ylabel(f"{STRESS_COMPONENT_LABELS[j]} [{unit_label}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, ls=":", alpha=0.4)
        if common_scale:
            from matplotlib.ticker import MultipleLocator

            ax.set_xlim(xe[0], xe[-1])
            ax.set_ylim(ye[0], ye[-1])
            ax.set_aspect("equal", adjustable="box")
            step = _nice_step(2 * half / 5)
            ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.yaxis.set_major_locator(MultipleLocator(step))

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="crimson", lw=1.3,
                      label="superposition HDR " + "/".join(f"{p:.0%}" for p in probs))]
    if fe_flat is not None:
        handles.append(Line2D([], [], color="k", ls="--", lw=1.0, label="FE (ANSYS) HDR"))
    if rve_contours:
        handles.append(Line2D([], [], color="gray", lw=0.5,
                              label=f"single RVE {probs[-2] if len(probs) > 1 else probs[0]:.0%} HDR"))
    handles += [Line2D([], [], marker="*", color="limegreen", mec="k", ms=10, ls="none",
                       label="macro stress C_eff·ε"),
                Line2D([], [], marker="o", mfc="none", mec="royalblue", ms=10, mew=1.8,
                       ls="none", label="ensemble mean")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.035, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes



def _node_samples(basis, macro_strain, macro_stress, scale):
    eps = basis._resolve_strain(macro_strain, macro_stress)
    sig = basis.local_stress(macro_strain=eps)
    n = sig.shape[1]
    w = np.repeat(basis.weights / n, n)
    flat = sig.reshape(-1, 6) * scale
    return flat, w, basis.effective_stiffness() @ eps * scale, w @ flat


def _half_width(flat, s_mean, comps):
    comps = sorted(comps)
    lo, hi = np.quantile(flat[:, comps], [0.0005, 0.9995], axis=0)
    return 1.12 * float(np.max(np.maximum(s_mean[comps] - lo, hi - s_mean[comps])))


def _pair_density(flat, w, s_mean, i, j, half, bins, smooth):
    xe = np.linspace(s_mean[i] - half, s_mean[i] + half, bins + 1)
    ye = np.linspace(s_mean[j] - half, s_mean[j] + half, bins + 1)
    h, area = _weighted_hist2d(flat[:, i], flat[:, j], w, xe, ye, smooth)
    return xe, ye, h, area


def _draw_density_surface(ax, flat, w, s_macro, s_mean, i, j, half, bins, smooth, probs,
                          zmax, cmap, view, unit_label, title):
    """One 3D panel. ``zmax=None`` -> the panel's own peak height."""
    xe, ye, h, area = _pair_density(flat, w, s_mean, i, j, half, bins, smooth)
    top = h.max() if zmax is None else zmax
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    floor = -0.25 * top
    ax.contour(X, Y, h, levels=np.unique(_hdr_levels(h, area, probs)), zdir="z",
               offset=floor, colors="crimson", linewidths=1.0)
    Z = np.where(h > 2e-3 * top, h, np.nan)  # drop the flat zero sheet that hides the floor
    ax.plot_surface(X, Y, Z, cmap=cmap, vmin=0, vmax=top, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.9)
    ix = int(np.clip(np.searchsorted(xe, s_macro[i]) - 1, 0, bins - 1))
    iy = int(np.clip(np.searchsorted(ye, s_macro[j]) - 1, 0, bins - 1))
    ax.plot([s_macro[i]] * 2, [s_macro[j]] * 2, [floor, h[ix, iy]], color="limegreen",
            lw=1.5, zorder=10)
    ax.plot([s_macro[i]], [s_macro[j]], [floor], marker="*", color="limegreen", mec="k",
            ms=10, ls="none", zorder=11)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_zlim(floor, top * 1.05)
    ax.set_box_aspect((1, 1, 0.75))
    ax.view_init(*view)
    ax.set_xlabel(f"{STRESS_COMPONENT_LABELS[i]} [{unit_label}]", fontsize=8, labelpad=2)
    ax.set_ylabel(f"{STRESS_COMPONENT_LABELS[j]} [{unit_label}]", fontsize=8, labelpad=2)
    ax.set_zlabel(f"p [1/{unit_label}²]", fontsize=8, labelpad=4)
    ax.tick_params(labelsize=6, pad=0)
    ax.zaxis.get_major_formatter().set_powerlimits((-2, 3))
    ax.set_title(title, fontsize=9)


def _surface_footer(fig, probs):
    fig.text(0.5, 0.005, "floor: HDR contours " + "/".join(f"{p:.0%}" for p in probs)
             + " (crimson); green star + line: macro stress C_eff·ε", ha="center", fontsize=8)


def plot_node_pair_surfaces(
    basis,
    macro_strain: Optional[np.ndarray] = None,
    macro_stress: Optional[np.ndarray] = None,
    pairs: Sequence[Tuple[int, int]] = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2, 5)),
    probs: Sequence[float] = (0.5, 0.9, 0.99),
    bins: int = 60,
    smooth: float = 1.5,
    ncols: int = 3,
    scale: float = 1e-6,
    unit_label: str = "MPa",
    shared_z: bool = True,
    view: Tuple[float, float] = (30.0, -55.0),
    cmap: str = "viridis",
    title: Optional[str] = None,
    cell_size: Tuple[float, float] = (5.0, 4.4),
):
    """3D surfaces of the joint density p(σi, σj | node) over the (σi, σj) plane.

    Same estimator as :func:`plot_node_pair_planes` (weighted ensemble
    histogram, Gaussian smoothing). Every in-plane axis spans the same width
    around the component mean; with ``shared_z`` all panels share one density
    axis, so peak heights and spreads are comparable between pairs. HDR
    contours (``probs``) are projected on the floor.
    """
    flat, w, s_macro, s_mean = _node_samples(basis, macro_strain, macro_stress, scale)
    half = _half_width(flat, s_mean, {c for p in pairs for c in p})
    zmax = None
    if shared_z:
        zmax = max(_pair_density(flat, w, s_mean, i, j, half, bins, smooth)[2].max()
                   for i, j in pairs)
    nrows = int(np.ceil(len(pairs) / ncols))
    fig = plt.figure(figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        _draw_density_surface(
            ax, flat, w, s_macro, s_mean, i, j, half, bins, smooth, probs, zmax, cmap, view,
            unit_label, f"p({STRESS_COMPONENT_LABELS[i]}, {STRESS_COMPONENT_LABELS[j]} | node)")
    if title:
        fig.suptitle(title, fontsize=11)
    _surface_footer(fig, probs)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def plot_pair_surfaces_vs_load(
    basis,
    macro_stresses: Sequence[Tuple[str, np.ndarray]],
    pairs: Sequence[Tuple[int, int]] = ((0, 1), (0, 3)),
    probs: Sequence[float] = (0.5, 0.9, 0.99),
    bins: int = 60,
    smooth: float = 1.5,
    scale: float = 1e-6,
    unit_label: str = "MPa",
    view: Tuple[float, float] = (30.0, -55.0),
    cmap: str = "viridis",
    title: Optional[str] = None,
    cell_size: Tuple[float, float] = (4.6, 4.0),
):
    """How p(σi, σj | node) changes with the macro load: rows = macro stress
    states (label, (6,) in Pa), columns = pairs. One in-plane width and one
    density axis for the whole grid, so shapes and peak heights compare
    directly between loads and pairs."""
    data = [(lbl,) + _node_samples(basis, None, np.asarray(s, float), scale)
            for lbl, s in macro_stresses]
    comps = {c for p in pairs for c in p}
    half = max(_half_width(flat, s_mean, comps) for _, flat, _, _, s_mean in data)
    zmax = max(_pair_density(flat, w, s_mean, i, j, half, bins, smooth)[2].max()
               for _, flat, w, _, s_mean in data for i, j in pairs)
    nrows, ncols = len(data), len(pairs)
    fig = plt.figure(figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    for r, (lbl, flat, w, s_macro, s_mean) in enumerate(data):
        for c, (i, j) in enumerate(pairs):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection="3d")
            _draw_density_surface(
                ax, flat, w, s_macro, s_mean, i, j, half, bins, smooth, probs, zmax, cmap,
                view, unit_label,
                f"{lbl}: p({STRESS_COMPONENT_LABELS[i]}, {STRESS_COMPONENT_LABELS[j]})")
    if title:
        fig.suptitle(title, fontsize=11)
    _surface_footer(fig, probs)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


# --------------------------------------------------------------------------
# 1D distribution vs macro load direction
# --------------------------------------------------------------------------
def plot_direction_sweep(
    sweep,
    ridge_angles: Optional[Sequence[float]] = None,
    scale: float = 1e-6,
    unit_label: str = "MPa",
    view: Tuple[float, float] = (28.0, -60.0),
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15.0, 11.0),
):
    """How the 1D distribution p(Q | node) changes while the macro load turns
    from pure component i (θ=0°) to pure component j (θ=90°).

    Panels: (a) 3D surface over (Q, θ); (b) density map with quantile lines and
    Q(macro); (c) PDFs at selected angles; (d) spread (CoV = std/|mean|) and
    between-RVE variance share vs θ. ``sweep`` is a
    :class:`~mv3d_stress_stats.localization.DirectionSweep`.
    """
    from matplotlib.lines import Line2D

    lab_i, lab_j = STRESS_COMPONENT_LABELS[sweep.component_i], STRESS_COMPONENT_LABELS[sweep.component_j]
    if sweep.kind == "strain":
        lab_i, lab_j = lab_i.replace("σ", "ε"), lab_j.replace("σ", "ε")
    th = sweep.thetas_deg
    if sweep.normalized:
        xs, xlabel = 1.0, f"η = {sweep.quantity} / Q(macro) [-]"
        qm = np.ones_like(th)
    else:
        xs, xlabel = scale, f"{sweep.quantity} [{unit_label}]"
        qm = sweep.q_macro * xs
    x = sweep.bin_centers * xs
    pdf = sweep.pdf / xs
    qv = sweep.quantiles * xs
    probs = list(np.round(sweep.probs, 6))
    zlabel = "p [-]" if sweep.normalized else f"p [1/{unit_label}]"

    fig = plt.figure(figsize=figsize)
    # (a) surface
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    X, T = np.meshgrid(x, th)
    top = pdf.max()
    Z = np.where(pdf > 2e-3 * top, pdf, np.nan)
    ax.plot_surface(X, T, Z, cmap=cmap, vmin=0, vmax=top, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.95)
    floor = -0.2 * top
    for p, ls in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
        if p in probs:
            ax.plot(qv[:, probs.index(p)], th, np.full_like(th, floor), color="k", ls=ls, lw=1)
    ax.plot(qm, th, np.full_like(th, floor), color="crimson", lw=1.5)
    ax.set_zlim(floor, top * 1.05)
    ax.view_init(*view)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("θ [deg]", fontsize=8)
    ax.set_zlabel(zlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.zaxis.get_major_formatter().set_powerlimits((-2, 3))
    ax.set_title("(a) density surface; floor: q05/q50/q95 (black), Q(macro) (red)", fontsize=9)

    # (b) map
    ax = fig.add_subplot(2, 2, 2)
    dth = np.diff(th).mean() if th.size > 1 else 1.0
    th_edges = np.concatenate([[th[0] - dth / 2], 0.5 * (th[1:] + th[:-1]), [th[-1] + dth / 2]])
    mesh = ax.pcolormesh(sweep.bin_edges * xs, th_edges, pdf, cmap=cmap, shading="flat",
                         rasterized=True)
    fig.colorbar(mesh, ax=ax, label=zlabel, pad=0.01)
    styles = {0.01: (":", 0.8), 0.05: ("--", 1.0), 0.25: ("-", 0.6), 0.5: ("-", 1.6),
              0.75: ("-", 0.6), 0.95: ("--", 1.0), 0.99: (":", 0.8)}
    for k, p in enumerate(probs):
        ls, lw = styles.get(p, ("-", 0.6))
        ax.plot(qv[:, k], th, color="white", ls=ls, lw=lw)
    ax.plot(qm, th, color="crimson", lw=1.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(f"θ [deg]   (0° = pure {lab_i}, 90° = pure {lab_j})", fontsize=9)
    ax.set_title("(b) density map, quantiles " + ", ".join(f"q{p:g}" for p in probs), fontsize=9)
    ax.legend(handles=[Line2D([], [], color="white", lw=1.6, label="median"),
                       Line2D([], [], color="white", ls="--", label="q05 / q95"),
                       Line2D([], [], color="crimson", lw=1.6, label="Q(macro)")],
              fontsize=8, loc="upper right", facecolor="0.3", labelcolor="white")

    # (c) curves at selected angles
    ax = fig.add_subplot(2, 2, 3)
    if ridge_angles is None:
        ridge_angles = [0, 15, 30, 45, 60, 75, 90]
    colors = plt.get_cmap("plasma")(np.linspace(0, 0.9, len(ridge_angles)))
    for a, c in zip(ridge_angles, colors):
        k = int(np.argmin(np.abs(th - a)))
        ax.plot(x, pdf[k], color=c, lw=1.5, label=f"θ = {th[k]:g}°")
        ax.axvline(qm[k], color=c, ls=":", lw=0.9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(zlabel, fontsize=9)
    ax.set_title("(c) p(Q | θ) at selected angles; dotted: Q(macro)", fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls=":", alpha=0.5)

    # (d) spread and between-RVE share
    ax = fig.add_subplot(2, 2, 4)
    # relative measures are meaningless where the mean / macro value ~ 0 (masked)
    ok = np.abs(sweep.mean) > 0.2 * sweep.std
    cov = np.where(ok, sweep.std / np.where(ok, np.abs(sweep.mean), 1.0), np.nan)
    ax.plot(th, cov * 100, color="navy", lw=1.8, label="CoV = std / |mean|")
    iq = np.full_like(th, np.nan)
    if 0.05 in probs and 0.95 in probs:
        width = sweep.quantiles[:, probs.index(0.95)] - sweep.quantiles[:, probs.index(0.05)]
        ref = np.ones_like(th) if sweep.normalized else np.abs(sweep.q_macro)
        okm = np.ones_like(th, dtype=bool) if sweep.normalized else ref > 0.2 * sweep.std
        iq = np.where(okm, width / np.where(okm, ref, 1.0), np.nan)
        ax.plot(th, iq * 100, color="teal", ls="--", lw=1.4, label="(q95 − q05) / |Q(macro)|")
    if not ok.all():
        ax.text(0.02, 0.02, "relative spread masked where |mean| < 0.2·std",
                transform=ax.transAxes, fontsize=7, color="gray")
    ax.set_xlabel("θ [deg]", fontsize=9)
    ax.set_ylabel("relative spread [%]", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(th, sweep.between_share * 100, color="darkorange", lw=1.4,
             label="between-RVE share of Var")
    ax2.set_ylabel("between-RVE share [%]", fontsize=9, color="darkorange")
    ax2.set_ylim(0, max(10.0, 1.2 * 100 * float(np.nanmax(sweep.between_share))))
    top_spread = np.nanmax(np.concatenate([cov * 100, iq * 100]))
    ax.set_ylim(0, 1.15 * top_spread if np.isfinite(top_spread) else None)
    ax2.tick_params(axis="y", colors="darkorange")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.set_title("(d) relative spread and realization variability vs θ", fontsize=9)

    mag = sweep.magnitude * (scale if sweep.kind == "stress" else 1.0)
    unit = unit_label if sweep.kind == "stress" else ""
    fig.suptitle(title or (f"{sweep.quantity}: macro load turns from pure {lab_i} to pure {lab_j}"
                           f" at |load| = {mag:g} {unit}"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
