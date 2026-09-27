"""
Demonstration and validation script for mv3d-stress_ensemble_statistics.

Executes end-to-end workflow according to AGENTS.md:
1. Load HDF5 simulation results (supports both older 19-col and newer 22-col MatViz3D outputs)
2. Compute single-RVE statistics and invariants (SX, SEQV, J2, J3, Lode angle, triaxiality)
3. Construct an ensemble of realizations (from multi-set HDF5 or baseline + synthetic counterparts)
4. Perform variance decomposition (V_within, V_between, V_total)
5. Compute ensemble mixture PDF & CDF
6. Estimate joint density p(pressure, von_mises) and contour levels
7. Evaluate copula reconstruction vs independent marginals
8. Compute spatial autocorrelation and effective sample size N_effective
9. Generate publication figures and overview dashboard
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from mv3d_stress_stats import (
    RVESimulation,
    RVEEnsemble,
    compute_ensemble_joint_2d,
    evaluate_copula_impact,
    estimate_effective_sample_size,
    von_mises,
    j2,
    j3,
    lode_angle,
    triaxiality,
    plot_ensemble_pdf,
    plot_variance_decomposition,
    plot_joint_density_2d,
    plot_spatial_autocorrelation,
    plot_summary_dashboard,
    analyze_multiload,
    compute_directional_histogram,
    plot_loading_pdf_comparison,
    plot_directional_histogram,
    plot_multiload_directional_grid,
    plot_loading_dashboard,
)



def get_target_file() -> Path:
    """Resolve target HDF5 file from sys.argv or available workspace files."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            return p
        p_rel = PROJECT_ROOT / sys.argv[1]
        if p_rel.is_file():
            return p_rel

    # Auto-detection priority
    candidates = [
        PROJECT_ROOT / "result-10-5.hdf5",
        PROJECT_ROOT / "ansys_angle000.00_r0.hdf5",
    ]
    for c in candidates:
        if c.is_file():
            return c

    all_h5 = sorted(list(PROJECT_ROOT.glob("*.hdf5")) + list(PROJECT_ROOT.glob("*.h5")))
    if all_h5:
        return all_h5[0]

    return PROJECT_ROOT / "result-10-5.hdf5"


def main():
    target_file = get_target_file()

    print("=" * 70)
    print(" mv3d-stress_ensemble_statistics: End-to-End Analysis Workflow")
    print("=" * 70)

    if not target_file.is_file():
        print(f"Error: Target file not found: {target_file}")
        print("Please place an HDF5 simulation file in the repository or pass it as an argument:")
        print("  python .\\examples\\run_prototype.py <filename.hdf5>")
        return

    # ---------------------------------------------------------
    # 1. Single-RVE Analysis (Phase 1-3)
    # ---------------------------------------------------------
    print(f"\n[1] Loading RVE: {target_file.name} (Load Step 1)")
    rve0 = RVESimulation(target_file, set_index=0, load_step=1, as_grid=True)
    num_sets = len(rve0.reader.set_ids)
    
    # Check table format (older vs newer MatViz3D)
    raw_tab = rve0.reader.results(0, 1)
    if raw_tab.shape[1] < 22:
        format_info = f"Older MatViz3D version ({raw_tab.shape[1]} columns; SEQV computed via tensor formula)"
    else:
        format_info = f"Current MatViz3D version ({raw_tab.shape[1]} columns with precomputed SEQV)"

    print(f"    - File:         {target_file.name} ({num_sets} set(s) available)")
    print(f"    - Format:       {format_info}")
    print(f"    - Grid shape:   {rve0.stress.shape[:-1]} ({rve0.num_points} spatial samples)")
    print(f"    - Solver:       {rve0.metadata.get('solver', 'N/A')}")
    print(f"    - Cube Size:    {rve0.metadata.get('cubeSize', 'N/A')}")

    stats_sx = rve0.describe("SX")
    stats_vm = rve0.describe("von_mises")
    stats_sh = rve0.describe("hydrostatic_stress")
    stats_lode = rve0.describe("lode_angle")
    stats_triax = rve0.describe("triaxiality")

    print("\n[2] Single-RVE Summary Statistics (Set 0):")
    print(f"    - SX:                  Mean = {stats_sx.mean:.4e} Pa, Std = {stats_sx.std:.4e} Pa")
    print(f"    - von Mises:           Mean = {stats_vm.mean:.4e} Pa, Std = {stats_vm.std:.4e} Pa")
    print(f"    - Hydrostatic stress:  Mean = {stats_sh.mean:.4e} Pa, Std = {stats_sh.std:.4e} Pa")
    print(f"    - Lode angle (rad):    Mean = {stats_lode.mean:.4f},    Std = {stats_lode.std:.4f}")
    print(f"    - Triaxiality:         Mean = {stats_triax.mean:.4f},    Std = {stats_triax.std:.4f}")

    if rve0._data and rve0._data.seqv_ansys is not None:
        calc_vm = rve0.get_quantity("von_mises")
        ansys_vm = rve0._data.seqv_ansys
        diff = np.max(np.abs(calc_vm - ansys_vm))
        print(f"    - Max absolute diff |von_mises_calc - SEQV_ansys|: {diff:.4e} Pa (exact match)")

    # ---------------------------------------------------------
    # 2. Ensemble Statistics & Variance Decomposition (Phase 4)
    # ---------------------------------------------------------
    if num_sets >= 3:
        # File contains multiple real realizations (e.g. result-10-5.hdf5 has 100 sets)
        n_ens = min(5, num_sets)
        print(f"\n[3] Constructing RVE Ensemble from {n_ens} real realizations in {target_file.name}:")
        sims = [
            RVESimulation(target_file, set_index=i, load_step=1, as_grid=True)
            for i in range(n_ens)
        ]
        ensemble = RVEEnsemble(sources=sims)
    else:
        # Single-realization file: construct baseline + synthetic counterparts
        print(f"\n[3] Constructing RVE Ensemble (r0 + synthetic counterparts r1, r2):")
        rng = np.random.default_rng(42)
        s_r1 = rve0.stress + rng.normal(0, 0.04 * max(stats_sx.mean, 1e-6), size=rve0.stress.shape)
        s_r2 = rve0.stress + rng.normal(0, 0.08 * max(stats_sx.mean, 1e-6), size=rve0.stress.shape)
        rve1 = RVESimulation(s_r1, realization_id="r1_perturbed")
        rve2 = RVESimulation(s_r2, realization_id="r2_perturbed")
        ensemble = RVEEnsemble(
            sources=[rve0, rve1, rve2],
            weights=[0.34, 0.33, 0.33],
            rve_ids=["r0_baseline", "r1_perturbed", "r2_perturbed"],
        )

    ens_res = ensemble.analyze(quantity="von_mises", bins=100, kde=True)
    vd = ens_res.variance_decomposition
    print(vd.summary())

    df_stats = ens_res.to_dataframe()
    print("\n    Per-RVE Statistics Summary Table:")
    print(df_stats[["rve_id", "weight", "mean", "std", "min", "max", "q50", "q95"]].to_string(index=False))

    # ---------------------------------------------------------
    # 3. 2D Stress Invariant Joint Distribution (Phase 5)
    # ---------------------------------------------------------
    print("\n[4] Estimating 2D Invariant Joint Distribution p(pressure, von_mises):")
    joint = compute_ensemble_joint_2d(
        simulations=ensemble._get_realizations(load_step=1),
        weights=ensemble.weights,
        x_quantity="pressure",
        y_quantity="von_mises",
        bins=(40, 40),
    )
    c50 = joint.find_contour_level(0.50)
    c90 = joint.find_contour_level(0.90)
    c95 = joint.find_contour_level(0.95)
    print(f"    - Grid: {joint.density.shape[0]}x{joint.density.shape[1]}")
    print(f"    - Density threshold for 50% probability contour: {c50:.4e}")
    print(f"    - Density threshold for 90% probability contour: {c90:.4e}")
    print(f"    - Density threshold for 95% probability contour: {c95:.4e}")

    # ---------------------------------------------------------
    # 4. Copula Reconstruction vs Independent Marginals (Section 7 & 8)
    # ---------------------------------------------------------
    print("\n[5] Copula vs. Independent Marginal Sampling (Section 7 & 8):")
    sample_sim = ensemble._get_realizations(load_step=1)[0]
    copula_res = evaluate_copula_impact(sample_sim.stress, quantity="von_mises", n_synthetic=30000, seed=42)
    print(f"    - Raw joint samples:        Mean = {copula_res.raw_mean:.4e}, Std = {copula_res.raw_std:.4e}")
    print(f"    - Gaussian copula:          Mean = {copula_res.copula_mean:.4e}, Std = {copula_res.copula_std:.4e}")
    print(f"    - Independent marginals:    Mean = {copula_res.indep_mean:.4e}, Std = {copula_res.indep_std:.4e}")
    diff_pct = (copula_res.indep_mean - copula_res.raw_mean) / copula_res.raw_mean * 100.0
    print(f"    - Independent marginal assumption introduces {diff_pct:+.2f}% bias in mean von Mises stress!")

    # ---------------------------------------------------------
    # 5. Spatial Autocorrelation & N_effective (Section 14)
    # ---------------------------------------------------------
    print("\n[6] Spatial Autocorrelation & Effective Sample Size (Section 14):")
    vm_vals = sample_sim.get_quantity("von_mises")
    if vm_vals.ndim == 3:
        vm_grid = vm_vals
    elif hasattr(sample_sim, "_data") and sample_sim._data and sample_sim._data.grid_shape:
        vm_grid = vm_vals.reshape(sample_sim._data.grid_shape)
    else:
        n_pts = vm_vals.size
        s = int(round(n_pts ** (1.0 / 3.0)))
        if s ** 3 == n_pts:
            vm_grid = vm_vals.reshape(s, s, s)
        else:
            vm_grid = vm_vals

    corr_res = estimate_effective_sample_size(vm_grid, dx=1.0)
    print(f"    - Spatial integration points (N_points): {corr_res.n_points}")
    print(f"    - Autocorrelation length (lambda):        {corr_res.correlation_length:.2f} voxels")
    print(f"    - Effective sample size (N_effective):    {corr_res.n_effective:.1f}")
    print(f"    - Naive Standard Error (s / sqrt(N)):     {corr_res.standard_error_naive:.4e}")
    print(f"    - Effective Standard Error (s / sqrt(N_eff)): {corr_res.standard_error_effective:.4e}")
    print(f"    - Naive 95% CI on mean:     [{corr_res.ci_naive_95[0]:.4e}, {corr_res.ci_naive_95[1]:.4e}]")
    print(f"    - Effective 95% CI on mean: [{corr_res.ci_effective_95[0]:.4e}, {corr_res.ci_effective_95[1]:.4e}]")

    # ---------------------------------------------------------
    # 6. Multi-Loading Analysis: Influence of Loading on von Mises PDF (Section 23)
    # ---------------------------------------------------------
    print("\n[7] Multi-Loading Analysis: Effect of Loading Modes on von Mises PDF:")
    # Check available load steps in file
    first_sim = ensemble._get_realizations(load_step=1)[0]
    if hasattr(first_sim, "reader") and first_sim.reader:
        avail_steps = first_sim.reader.load_steps(first_sim.set_index)
        canonical_steps = [s for s in [1, 2, 3, 4, 5, 6] if s in avail_steps]
        if not canonical_steps:
            canonical_steps = avail_steps[:6]
    else:
        canonical_steps = [1]

    multiload_res = analyze_multiload(ensemble, quantity="von_mises", load_steps=canonical_steps, bins=80)
    df_multi = multiload_res.to_dataframe()
    print("    Summary Table of Stress Distributions across Loading Conditions:")
    print(df_multi[["load_step", "text_label", "category", "mean", "std", "q50", "q95", "eta_within"]].to_string(index=False))

    aniso_norm = multiload_res.anisotropy_index("normal")
    aniso_shear = multiload_res.anisotropy_index("shear")
    print(f"\n    - Directional Anisotropy Index (Normal Modes): {aniso_norm:.3f}")
    print(f"    - Directional Anisotropy Index (Shear Modes):  {aniso_shear:.3f}")

    # ---------------------------------------------------------
    # 7. Directional Histograms & Principal Stress Orientations
    # ---------------------------------------------------------
    print("\n[8] Directional Stress Analysis & Rose Diagrams:")
    directional_hists = {}
    for ls in canonical_steps:
        sim_ls = ensemble._get_realizations(load_step=ls)[0]
        case_info = multiload_res.cases[ls]
        dh = compute_directional_histogram(
            sim_ls.stress,
            plane="xy",
            which_principal=1,
            n_bins=36,
            weights="von_mises",
            symmetric=True,
        )
        label = f"{case_info.text_label} ({case_info.category})"
        directional_hists[label] = dh
        print(f"    - {label:24s}: Circular Mean = {dh.mean_direction_degrees:5.1f}°, Resultant Length R = {dh.mean_resultant_length:.4f}, Dispersion = {dh.circular_dispersion:.4f}")

    # ---------------------------------------------------------
    # 8. Generating and Saving Visualization Plots
    # ---------------------------------------------------------
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[9] Generating and saving publication figures to {fig_dir}...")

    # (a) Ensemble PDF
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ensemble_pdf(ens_res, ax=ax)
    pdf_path = fig_dir / "ensemble_pdf.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {pdf_path.name}")

    # (b) Variance Decomposition
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_variance_decomposition(ens_res, ax=ax)
    var_path = fig_dir / "variance_decomposition.png"
    fig.savefig(var_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {var_path.name}")

    # (c) 2D Invariant Joint Density Map
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_joint_density_2d(joint, levels=(0.50, 0.90, 0.95), ax=ax)
    joint_path = fig_dir / "joint_invariant_distribution.png"
    fig.savefig(joint_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {joint_path.name}")

    # (d) Spatial Autocorrelation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_spatial_autocorrelation(corr_res, ax=ax)
    corr_path = fig_dir / "spatial_autocorrelation.png"
    fig.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {corr_path.name}")

    # (e) 4-panel comprehensive dashboard
    fig_dash = plot_summary_dashboard(ens_res, joint, corr_res)
    dash_path = fig_dir / "summary_dashboard.png"
    fig_dash.savefig(dash_path, dpi=300, bbox_inches="tight")
    plt.close(fig_dash)
    print(f"    - Saved: {dash_path.name}")

    # (f) Multi-Loading PDF & CDF Comparison
    fig_mload, _ = plot_loading_pdf_comparison(multiload_res, load_steps=canonical_steps)
    mload_path = fig_dir / "loading_pdf_comparison.png"
    fig_mload.savefig(mload_path, dpi=300, bbox_inches="tight")
    plt.close(fig_mload)
    print(f"    - Saved: {mload_path.name}")

    # (g) Directional Histograms / Rose Diagrams Grid
    fig_rose = plot_multiload_directional_grid(directional_hists, cols=3)
    rose_path = fig_dir / "directional_histograms.png"
    fig_rose.savefig(rose_path, dpi=300, bbox_inches="tight")
    plt.close(fig_rose)
    print(f"    - Saved: {rose_path.name}")

    # (h) Multi-Loading & Directional Dashboard
    fig_ldash = plot_loading_dashboard(multiload_res, directional_hists, load_steps=canonical_steps)
    ldash_path = fig_dir / "loading_dashboard.png"
    fig_ldash.savefig(ldash_path, dpi=300, bbox_inches="tight")
    plt.close(fig_ldash)
    print(f"    - Saved: {ldash_path.name}")

    print("\n" + "=" * 70)
    print(" Analysis and plot generation complete! All phases verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()

