"""
Stage 1 of the ANSYS ACT tool: stress distribution at a macroscopic node.

1. Build the 6x6 localization basis from the unit load cases of every RVE.
2. Validate: every FE load step in the file is reproduced by superposition
   (point-wise error and distribution overlay), and compare with the old
   scalar-rescaling approach (eta = Q / Q_macro), which is wrong for
   multiaxial states.
3. Distribution of von Mises / principal / hydrostatic stress for an arbitrary
   macro node state given as macro STRESS (as ANSYS would export it).
4. Nodal batch statistics -> CSV (what the ACT extension contours).

Usage:
    python examples/run_node_distribution.py [result-10-5.hdf5]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mv3d_stress_stats import LocalizationBasis, plot_node_distribution, plot_superposition_validation
from mv3d_stress_stats.node_cli import write_stats_csv


def main() -> None:
    h5 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "result-10-5.hdf5"
    out = ROOT / "figures" / "node_distribution"
    out.mkdir(parents=True, exist_ok=True)

    basis = LocalizationBasis.from_file(h5)
    print(f"Basis: {basis.num_realizations} RVEs x {basis.num_points} points, "
          f"unit steps {basis.metadata['unit_load_steps']}, shear in file = {basis.metadata['shear_in_file']}")
    np.set_printoptions(precision=1, suppress=True, linewidth=120)
    print("C_eff [GPa] (engineering Voigt, from nodal fields):\n", basis.effective_stiffness() / 1e9)

    # --- 2. validation ---------------------------------------------------
    checks = basis.verify_against_file()
    for c in checks:
        print(f"  ls_{c.load_step:<3d} eps={np.array2string(c.macro_strain * 1e4, precision=1)}e-4  "
              f"max rel error {c.max_rel_error:.1e}")
    combined = [c.load_step for c in checks if np.count_nonzero(c.macro_strain) > 1][:4]
    if combined:
        fig, _ = plot_superposition_validation(basis, str(h5), combined)
        fig.savefig(out / "superposition_validation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  saved superposition_validation.png")

    # --- 3. one node given by macro stress --------------------------------
    sigma_node = np.array([60e6, 20e6, -10e6, 15e6, 0.0, 5e6])  # Pa, SX SY SZ SXY SYZ SXZ
    eps_node = basis.macro_strain_from_stress(sigma_node)
    print("Node macro stress [MPa]:", sigma_node / 1e6,
          "-> engineering strain:", np.array2string(eps_node, formatter={"float_kind": "{:.3e}".format}))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, qn in zip(axes, ["von_mises", "principal_1", "hydrostatic_stress"]):
        nd = basis.node_distribution(qn, macro_stress=sigma_node)
        plot_node_distribution(nd, ax=ax, title=qn)
        s = nd.stats
        print(f"  {qn:20s} Q(macro)={s['q_of_macro_stress']/1e6:8.2f}  mean={s['mean']/1e6:8.2f}  "
              f"q05={s['q05']/1e6:8.2f}  q95={s['q95']/1e6:8.2f}  q99={s['q99']/1e6:8.2f} MPa  "
              f"between-RVE share={s['eta_between']*100:.1f}%")
    fig.suptitle("Local stress distribution at a macro node, σ_macro = "
                 + np.array2string(sigma_node / 1e6, precision=0) + " MPa")
    fig.tight_layout()
    fig.savefig(out / "node_distribution_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 4. batch over nodes (synthetic macro field) -----------------------
    rng = np.random.default_rng(1)
    sig_nodes = sigma_node * rng.uniform(0.2, 1.2, size=(500, 1))
    ids = [str(i + 1) for i in range(len(sig_nodes))]
    probs, thr = [0.05, 0.5, 0.95, 0.99], [80e6]
    st = basis.nodal_statistics("von_mises", macro_stresses=sig_nodes, probs=probs, thresholds=thr)
    write_stats_csv(str(out / "node_stats_example.csv"), ids, st, probs, thr)
    print(f"  wrote node_stats_example.csv for {len(ids)} nodes")


if __name__ == "__main__":
    main()
