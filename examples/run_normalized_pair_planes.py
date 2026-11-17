"""
Pipeline: Normalized RVE Ensemble & Pair-Plane PDF Section Analysis.

Workflow:
1. Load dataset and print descriptive statistics (SX, SY, SZ, SXY, SYZ, SXZ, von Mises, hydrostatic stress).
2. Normalize and join RVE realizations to a single unified KDE PDF (macro-load normalization).
3. Plot normalized pairs as angular sections and radial ray sections (0°, 30°, 45°, 60°, 90°) of PDF in canonical planes:
   (SX, SY), (SY, SZ), (SX, SZ), (SX, SXY), (SY, SYZ), (SZ, SXZ).

Usage:
    python examples/run_normalized_pair_planes.py [path_to_hdf5] [--load-step 1] [--max-rves 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from mv3d_stress_stats import (
    RVEEnsemble,
    RVESimulation,
    compute_all_pair_plane_sections,
    compute_pair_plane_sections,
    compute_all_pair_plane_ray_sections,
    compute_pair_plane_ray_sections,
    compute_all_pair_plane_cartesian_sections,
    compute_pair_plane_cartesian_sections,
    plot_pair_plane_polar_heatmap,
    plot_pair_plane_sections,
    plot_pair_planes_angular_comparison,
    plot_pair_planes_grid,
    plot_pair_plane_ray_sections,
    plot_pair_planes_ray_grid,
    plot_pair_plane_ray_dashboard,
    plot_pair_plane_cartesian_sections,
    plot_pair_planes_cartesian_grid,
    plot_pair_plane_cartesian_dashboard,
    von_mises,
    hydrostatic_stress,
)
from mv3d_stress_stats.directional import (
    PairPlaneSectionResult,
    PairPlaneRaySectionResult,
    PairPlaneCartesianSectionResult,
    CANONICAL_PAIR_PLANES,
)
from mv3d_stress_stats.reader import RVEReader


def resolve_dataset(custom_path: Optional[str] = None) -> Path:
    """Find available target HDF5 dataset."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {custom_path}")

    # Fallback search order
    candidates = [
        PROJECT_ROOT / "result-10-5.hdf5",
        PROJECT_ROOT / "ansys_angle000.00_r0.hdf5",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Search any hdf5 file in project root
    h5_files = list(PROJECT_ROOT.glob("*.hdf5"))
    if h5_files:
        return h5_files[0]

    raise FileNotFoundError(
        "No HDF5 dataset found in repository root. Please specify path as first argument."
    )


def build_rve_ensemble(
    h5_path: Path,
    load_step: int = 1,
    max_rves: Optional[int] = 1000,
) -> RVEEnsemble:
    """Construct an RVE ensemble from multi-set HDF5 or synthetic perturbation."""
    reader = RVEReader(h5_path)
    avail_sets = reader.set_ids
    n_sets = len(avail_sets)

    if n_sets > 1:
        if max_rves is not None and max_rves > 0:
            selected_sets = list(range(min(max_rves, n_sets)))
        else:
            selected_sets = list(range(n_sets))

        sims = [
            RVESimulation(h5_path, set_index=i, load_step=load_step, as_grid=True)
            for i in selected_sets
        ]
        ensemble = RVEEnsemble(sources=sims)
        print(f"    Constructed RVE Ensemble with {len(selected_sets)} realizations (from {n_sets} available sets in {h5_path.name}).")
        return ensemble
    else:
        # Single realization file -> create synthetic perturbed ensemble
        base_sim = RVESimulation(h5_path, set_index=0, load_step=load_step)
        rng = np.random.default_rng(42)
        mean_sx = float(np.mean(np.abs(base_sim.stress[:, 0])))
        noise_level = 0.08 * (mean_sx if mean_sx > 0 else 1e6)

        sim_r0 = base_sim
        sim_r1 = RVESimulation(base_sim.stress + rng.normal(0, noise_level, size=base_sim.stress.shape), realization_id="r1_perturbed")
        sim_r2 = RVESimulation(base_sim.stress + rng.normal(0, noise_level, size=base_sim.stress.shape), realization_id="r2_perturbed")

        ensemble = RVEEnsemble(
            sources=[sim_r0, sim_r1, sim_r2],
            weights=[0.34, 0.33, 0.33],
            rve_ids=["r0_baseline", "r1_perturbed", "r2_perturbed"],
        )
        print(f"    Constructed 3-realization synthetic RVE ensemble from baseline {h5_path.name}.")
        return ensemble


# =========================================================================
# STEP 1: Load Dataset & Print Descriptive Statistics
# =========================================================================

def print_descriptive_statistics(ensemble: RVEEnsemble) -> None:
    """Print detailed spatial and ensemble descriptive statistics for all stress quantities."""
    print("=" * 86)
    print(" STEP 1: DESCRIPTIVE STATISTICS OF STRESS TENSOR COMPONENTS & INVARIANTS")
    print("=" * 86)

    quantities = [
        ("SX", "sx"),
        ("SY", "sy"),
        ("SZ", "sz"),
        ("SXY", "sxy"),
        ("SYZ", "syz"),
        ("SXZ", "sxz"),
        ("von Mises (SEQV)", "von_mises"),
        ("Hydrostatic Stress", "hydrostatic_stress"),
    ]

    header = f"{'Quantity':<20} | {'Mean (Pa)':<11} | {'Std (Pa)':<11} | {'Min (Pa)':<10} | {'q25 (Pa)':<10} | {'Median':<10} | {'q75 (Pa)':<10} | {'Max (Pa)':<10}"
    print(header)
    print("-" * len(header))

    for label, q_key in quantities:
        res = ensemble.analyze(quantity=q_key, bins=50, kde=False)
        pts = res.bin_centers
        pdf = res.ensemble_pdf
        dx = np.diff(res.bin_edges)

        # Pooled ensemble moments
        mean_val = float(np.sum(pts * pdf * dx))
        var_val = float(np.sum(((pts - mean_val) ** 2) * pdf * dx))
        std_val = float(np.sqrt(max(var_val, 0.0)))

        # CDF and quantiles
        cdf = np.cumsum(pdf * dx)
        cdf = cdf / cdf[-1]

        def get_q(p_val: float) -> float:
            idx = np.searchsorted(cdf, p_val)
            return float(pts[min(idx, len(pts) - 1)])

        q25 = get_q(0.25)
        q50 = get_q(0.50)
        q75 = get_q(0.75)
        min_val = float(pts[0])
        max_val = float(pts[-1])

        print(f"{label:<20} | {mean_val:11.4e} | {std_val:11.4e} | {min_val:10.3e} | {q25:10.3e} | {q50:10.3e} | {q75:10.3e} | {max_val:10.3e}")

    # Homogenized Macroscopic Mean Stress Tensor
    first_sim = ensemble._get_realizations()[0]
    macro = first_sim.homogenized_stress()
    print("\n  Homogenized Macroscopic Stress Tensor bar(Sigma) (Load Step 1, Realization 0):")
    print(f"    SX  = {macro[0]:12.4e} Pa | SXY = {macro[3]:12.4e} Pa")
    print(f"    SY  = {macro[1]:12.4e} Pa | SYZ = {macro[4]:12.4e} Pa")
    print(f"    SZ  = {macro[2]:12.4e} Pa | SXZ = {macro[5]:12.4e} Pa")

    # Variance Decomposition
    vm_res = ensemble.analyze(quantity="von_mises", bins=60, kde=False)
    vd = vm_res.variance_decomposition
    print("\n  Variance Decomposition of Equivalent Stress sigma_VM:")
    print(f"    Within-RVE Variance  (V_within):  {vd.v_within:12.4e} Pa^2 ({vd.eta_within * 100:.1f}%) [Spatial Heterogeneity]")
    print(f"    Between-RVE Variance (V_between): {vd.v_between:12.4e} Pa^2 ({vd.eta_between * 100:.1f}%) [Microstructural Realization]")
    print(f"    Total Ensemble Variance (V_total): {vd.v_total:12.4e} Pa^2")


# =========================================================================
# STEP 2: Normalize and Join RVE to Single KDE PDF
# =========================================================================

def normalize_and_join_rve_kde(
    ensemble: RVEEnsemble,
    quantity: str = "von_mises",
    n_eval_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Normalize each realization by its macroscopic load and join to a single ensemble KDE PDF.
    
    Formula:
        tilde(sigma)_r(x) = sigma_r(x) / bar(sigma)_r
        p(tilde(sigma)) = sum_{r=1}^M w_r * p(tilde(sigma) | R=r)
    """
    print("\n" + "=" * 86)
    print(" STEP 2: NORMALIZE & JOIN RVE REALIZATIONS TO SINGLE KDE PDF")
    print("=" * 86)

    # Analyze with load normalization enabled
    ens_res = ensemble.analyze(quantity=quantity, bins=100, normalize=True, kde=True)

    eval_pts = ens_res.kde_eval_points if ens_res.kde_eval_points is not None else ens_res.bin_centers
    joined_pdf = ens_res.ensemble_kde if ens_res.ensemble_kde is not None else ens_res.ensemble_pdf

    dx = np.diff(eval_pts)
    int_check = float(np.sum(joined_pdf[:-1] * dx))
    norm_mean = float(np.sum(eval_pts[:-1] * joined_pdf[:-1] * dx))

    print(f"  Joined Single KDE PDF for Normalized {quantity}:")
    print(f"    - Macro normalization mode:     tilde(Q) = Q / bar(Q)_macro (dimensionless ~1.0)")
    print(f"    - Number of joined RVEs (M):    {len(ensemble._get_realizations())}")
    print(f"    - Joined KDE support range:     [{eval_pts[0]:.3f}, {eval_pts[-1]:.3f}]")
    print(f"    - Mean of normalized field:     {norm_mean:.4f} (expected ~1.0)")
    print(f"    - Total integral of joined PDF: {int_check:.4f}")

    meta = {
        "norm_mean": norm_mean,
        "integral": int_check,
        "q_min": float(eval_pts[0]),
        "q_max": float(eval_pts[-1]),
    }
    return eval_pts, joined_pdf, meta


# =========================================================================
# STEP 3: Plot Normalized Pairs as Sections of PDF in Planes
# =========================================================================

def plot_normalized_pair_planes(
    ensemble: RVEEnsemble,
    output_dir: Path,
    n_angles: int = 180,
    dpi: int = 300,
) -> Dict[str, Path]:
    """Compute and plot normalized pair plane PDF sections across canonical stress planes.
    
    Normalizes components by the plane macro radius R_macro = sqrt(X_macro^2 + Y_macro^2),
    preserving raw stress angles theta = arctan2(Y, X) while scaling radial sections
    to dimensionless values ~1.0.
    """
    print("\n" + "=" * 86)
    print(" STEP 3: COMPUTE & PLOT NORMALIZED PAIR-PLANE PDF SECTIONS")
    print("=" * 86)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: Dict[str, Path] = {}

    print("  Evaluating 2D KDE and angular sections across 6 canonical planes...")
    print("  Planes: (SX, SY), (SY, SZ), (SX, SZ), (SX, SXY), (SY, SYZ), (SZ, SXZ)")
    print("  Normalization: Macro-load (R_macro = sqrt(X_macro^2 + Y_macro^2)), centered=False")

    pair_sections = compute_all_pair_plane_sections(
        source=ensemble,
        planes=CANONICAL_PAIR_PLANES,
        n_angles=n_angles,
        normalize=True,
        centered=False,
    )

    for label, sec in pair_sections.items():
        print(f"    - Plane {label:12s}: Normalized radii q10..q90 = {np.round(sec.radii, 3).tolist()}, Macro R = {sec.macro_radius:.3e} Pa")

    # ---------------------------------------------------------------------
    # Plot 1: 6-Panel Multi-Plot Grid of PDF Sections (PDF vs Angle theta)
    # ---------------------------------------------------------------------
    print("\n  [1/10] Generating 6-panel grid of normalized PDF sections...")
    fig_grid, _ = plot_pair_planes_grid(pair_sections, ncols=3, figsize=(16, 9))
    grid_path = output_dir / "normalized_pair_planes_grid.png"
    fig_grid.savefig(grid_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_grid)
    saved_figures["grid"] = grid_path
    print(f"        Saved: {grid_path.name}")

    # ---------------------------------------------------------------------
    # Plot 2: Detailed Section Plot for Primary Plane (SX, SY)
    # ---------------------------------------------------------------------
    print("  [2/10] Generating detailed section plot for primary plane (SX, SY)...")
    sec_sx_sy = pair_sections["(SX, SY)"]
    fig_sec, ax_sec = plt.subplots(figsize=(9, 5.5))
    plot_pair_plane_sections(sec_sx_sy, ax=ax_sec, density_type="joint", cmap="plasma")
    sec_path = output_dir / "normalized_sections_sx_sy.png"
    fig_sec.savefig(sec_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_sec)
    saved_figures["sections_sx_sy"] = sec_path
    print(f"        Saved: {sec_path.name}")

    # ---------------------------------------------------------------------
    # Plot 3: 2D Unrolled Polar Heatmap with Horizontal Reference Lines
    # ---------------------------------------------------------------------
    print("  [3/10] Generating 2D unrolled polar heatmap for (SX, SY)...")
    fig_hm, ax_hm = plt.subplots(figsize=(9, 5.5))
    plot_pair_plane_polar_heatmap(
        sec_sx_sy,
        ax=ax_hm,
        cmap="viridis",
        show_section_lines=True,
        show_mean_line=True,
    )
    hm_path = output_dir / "normalized_polar_heatmap_sx_sy.png"
    fig_hm.savefig(hm_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_hm)
    saved_figures["polar_heatmap"] = hm_path
    print(f"        Saved: {hm_path.name}")

    # ---------------------------------------------------------------------
    # Plot 4: Comparative Angular Marginal PDF Across All 6 Planes
    # ---------------------------------------------------------------------
    print("  [4/10] Generating comparative angular PDF overlay...")
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    plot_pair_planes_angular_comparison(pair_sections, ax=ax_comp)
    comp_path = output_dir / "normalized_angular_comparison.png"
    fig_comp.savefig(comp_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_comp)
    saved_figures["angular_comparison"] = comp_path
    print(f"        Saved: {comp_path.name}")

    # ---------------------------------------------------------------------
    # Plot 5: 6-Panel Multi-Plot Grid of Ray Sections (0°, 30°, 45°, 60°, 90°)
    # ---------------------------------------------------------------------
    print("\n  [5/10] Evaluating & plotting ray sections at angles theta in {0, 30, 45, 60, 90} deg...")
    ray_sections = compute_all_pair_plane_ray_sections(
        source=ensemble,
        planes=CANONICAL_PAIR_PLANES,
        angles_deg=[0.0, 30.0, 45.0, 60.0, 90.0],
        normalize=True,
        centered=False,
    )
    for label, rsec in ray_sections.items():
        peak_info = ", ".join([f"{a:g} deg: r={r:.2f}" for a, r in zip(rsec.angles_deg, rsec.peak_radii)])
        print(f"    - Plane {label:12s}: Peaks [{peak_info}]")

    fig_ray_grid, _ = plot_pair_planes_ray_grid(ray_sections, ncols=3, figsize=(16, 9))
    ray_grid_path = output_dir / "normalized_ray_sections_grid.png"
    fig_ray_grid.savefig(ray_grid_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_ray_grid)
    saved_figures["ray_sections_grid"] = ray_grid_path
    print(f"        Saved: {ray_grid_path.name}")

    # ---------------------------------------------------------------------
    # Plot 6: Detailed Ray Sections for Primary Plane (SX, SY)
    # ---------------------------------------------------------------------
    print("  [6/10] Generating detailed ray sections plot for (SX, SY)...")
    ray_sx_sy = ray_sections["(SX, SY)"]
    fig_ray_sec, ax_ray_sec = plt.subplots(figsize=(9, 5.5))
    plot_pair_plane_ray_sections(ray_sx_sy, ax=ax_ray_sec)
    ray_sec_path = output_dir / "normalized_ray_sections_sx_sy.png"
    fig_ray_sec.savefig(ray_sec_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_ray_sec)
    saved_figures["ray_sections_sx_sy"] = ray_sec_path
    print(f"        Saved: {ray_sec_path.name}")

    # ---------------------------------------------------------------------
    # Plot 7: 2-Panel Ray Dashboard for Primary Plane (SX, SY)
    # ---------------------------------------------------------------------
    print("  [7/10] Generating 2-panel ray dashboard (geometry + profiles) for (SX, SY)...")
    fig_dash = plot_pair_plane_ray_dashboard(ray_sx_sy, figsize=(13, 5.5))
    dash_path = output_dir / "normalized_ray_dashboard_sx_sy.png"
    fig_dash.savefig(dash_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_dash)
    saved_figures["ray_dashboard_sx_sy"] = dash_path
    print(f"        Saved: {dash_path.name}")

    # ---------------------------------------------------------------------
    # Plot 8: 6-Panel Multi-Plot Grid of Parallel Cartesian Sections (Option 3)
    # ---------------------------------------------------------------------
    print("\n  [8/10] Evaluating & plotting parallel Cartesian sections at quantiles q10..q90...")
    cart_sections = compute_all_pair_plane_cartesian_sections(
        source=ensemble,
        planes=CANONICAL_PAIR_PLANES,
        slice_axis="y",
        percentiles=[10, 25, 50, 75, 90],
        normalize=True,
        include_2d_surface=True,
    )
    for label, csec in cart_sections.items():
        peak_info = ", ".join([f"{lvl:.2f}: pk={pk:.2f}" for lvl, pk in zip(csec.slice_levels, csec.peak_positions)])
        print(f"    - Plane {label:12s} [Cut {csec.slice_component}]: Peaks [{peak_info}]")

    fig_cart_grid, _ = plot_pair_planes_cartesian_grid(cart_sections, ncols=3, figsize=(16, 9))
    cart_grid_path = output_dir / "normalized_cartesian_sections_grid.png"
    fig_cart_grid.savefig(cart_grid_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_cart_grid)
    saved_figures["cartesian_sections_grid"] = cart_grid_path
    print(f"        Saved: {cart_grid_path.name}")

    # ---------------------------------------------------------------------
    # Plot 9: Detailed Cartesian Sections for Primary Plane (SX, SY)
    # ---------------------------------------------------------------------
    print("  [9/10] Generating detailed Cartesian sections plot for (SX, SY)...")
    cart_sx_sy = cart_sections["(SX, SY)"]
    fig_cart_sec, ax_cart_sec = plt.subplots(figsize=(9, 5.5))
    plot_pair_plane_cartesian_sections(cart_sx_sy, ax=ax_cart_sec)
    cart_sec_path = output_dir / "normalized_cartesian_sections_sx_sy.png"
    fig_cart_sec.savefig(cart_sec_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_cart_sec)
    saved_figures["cartesian_sections_sx_sy"] = cart_sec_path
    print(f"        Saved: {cart_sec_path.name}")

    # ---------------------------------------------------------------------
    # Plot 10: 2-Panel Cartesian Dashboard (2D Map + 1D Elevation Profiles)
    # ---------------------------------------------------------------------
    print("  [10/10] Generating 2-panel Cartesian dashboard (2D landscape + 1D profiles) for (SX, SY)...")
    fig_cart_dash = plot_pair_plane_cartesian_dashboard(cart_sx_sy, figsize=(14, 5.5))
    cart_dash_path = output_dir / "normalized_cartesian_dashboard_sx_sy.png"
    fig_cart_dash.savefig(cart_dash_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_cart_dash)
    saved_figures["cartesian_dashboard_sx_sy"] = cart_dash_path
    print(f"        Saved: {cart_dash_path.name}")

    return saved_figures


# =========================================================================
# Main Execution Entrypoint
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline: Normalized RVE Ensemble & Pair-Plane PDF Section Analysis"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Path to HDF5 dataset file (defaults to result-10-5.hdf5 or ansys_angle000.00_r0.hdf5)",
    )
    parser.add_argument(
        "--load-step",
        type=int,
        default=1,
        help="Load step index to analyze (default: 1)",
    )
    parser.add_argument(
        "--max-rves",
        type=int,
        default=10,
        help="Max number of RVE sets to load from multi-set HDF5 (default: 10, set 0 for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures" / "normalized_pair_planes"),
        help="Directory to save generated figures",
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=180,
        help="Number of angular evaluation points in [-180, +180] (default: 180)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure resolution DPI (default: 300)",
    )

    args = parser.parse_args()

    print("=" * 86)
    print("  MV3D STRESS ENSEMBLE STATISTICS: NORMALIZED PAIR-PLANE PIPELINE")
    print("=" * 86)

    # 1. Resolve Dataset & Build Ensemble
    h5_path = resolve_dataset(args.dataset)
    print(f"  Target Dataset: {h5_path}")
    max_rves_val = None if args.max_rves == 0 else args.max_rves
    ensemble = build_rve_ensemble(h5_path, load_step=args.load_step, max_rves=max_rves_val)

    # 2. Step 1: Print Descriptive Statistics
    print_descriptive_statistics(ensemble)

    # 3. Step 2: Normalize and Join RVE to Single KDE PDF
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_pts, joined_pdf, _ = normalize_and_join_rve_kde(ensemble, quantity="von_mises")

    # Plot normalized 1D KDE PDF
    fig_vm, ax_vm = plt.subplots(figsize=(8, 4.8))
    ax_vm.plot(eval_pts, joined_pdf, color="#1f77b4", linewidth=2.2, label=r"Joined Ensemble KDE $p(\tilde{\sigma}_{\mathrm{VM}})$")
    ax_vm.fill_between(eval_pts, 0, joined_pdf, color="#1f77b4", alpha=0.15)
    ax_vm.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label=r"Macro Mean $\tilde{\sigma} = 1.0$")
    ax_vm.set_xlabel(r"Macro-Normalized von Mises Stress $\tilde{\sigma}_{\mathrm{VM}} = \sigma_{\mathrm{VM}} / \bar{\sigma}_{\mathrm{VM}}$ (dimensionless)", fontsize=11)
    ax_vm.set_ylabel(r"Probability Density $p(\tilde{\sigma}_{\mathrm{VM}})$", fontsize=11)
    ax_vm.set_title(r"Joined Single Ensemble KDE PDF of Macro-Normalized von Mises Stress", fontsize=12, fontweight="bold")
    ax_vm.grid(True, linestyle=":", alpha=0.6)
    ax_vm.legend(frameon=True, fontsize=9)
    vm_pdf_path = out_dir / "normalized_ensemble_von_mises_pdf.png"
    fig_vm.savefig(vm_pdf_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_vm)
    print(f"  Saved joined 1D KDE PDF plot: {vm_pdf_path.name}")

    # 4. Step 3: Compute & Plot Normalized Pairs as Sections of PDF in Planes
    saved_figs = plot_normalized_pair_planes(
        ensemble=ensemble,
        output_dir=out_dir,
        n_angles=args.angles,
        dpi=args.dpi,
    )
    saved_figs["normalized_1d_pdf"] = vm_pdf_path

    print("\n" + "=" * 86)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" All figures saved to: {out_dir}")
    for name, path in saved_figs.items():
        print(f"   - {name:<22}: {path.name}")
    print("=" * 86)


if __name__ == "__main__":
    main()
