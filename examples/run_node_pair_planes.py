"""
Pair planes p(σi, σj | node) of the local stress reconstructed by the
6x6 localization superposition — a visual check of the node distributions.

Figures (figures/node_pair_planes/):
  validation_ls<k>.png   all 15 component pairs for FE load step k: FE (ANSYS)
                         HDR contours (black dashed) over the superposition ones
                         (crimson); they must coincide.
  node_pairs.png         an arbitrary macro node state (macro stress in MPa).
  node_pairs_rves.png    the same node, 6 main pairs, with single-RVE contours
                         (realization-to-realization variability).

Usage:
    python examples/run_node_pair_planes.py [result-10-5.hdf5]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np

from mv3d_stress_stats.localization import LocalizationBasis
from mv3d_stress_stats.plotting import (plot_node_pair_planes, plot_node_pair_surfaces,
                                      plot_pair_surfaces_vs_load)

LABELS = ["εx", "εy", "εz", "γxy", "γyz", "γxz"]
MAIN_PAIRS = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2, 5)]


def main(path: str) -> None:
    out = ROOT / "figures" / "node_pair_planes"
    out.mkdir(parents=True, exist_ok=True)
    basis = LocalizationBasis.from_file_cached(path)
    sets = basis.metadata["set_ids"]
    factor = 2.0 if basis.metadata.get("shear_in_file") == "tensor" else 1.0
    print(f"{basis.num_realizations} RVEs x {basis.num_points} points")

    with h5py.File(path, "r") as f:
        for ls in (7, 9, 20):
            if f"ls_{ls}" not in f[sets[0]]:
                continue
            eps = f[f"{sets[0]}/ls_{ls}/eps_as_loading"][:].astype(float)
            eps[3:] *= factor
            fe = [f[f"{s}/ls_{ls}/results"][:, 7:13].astype(float) for s in sets]
            lbl = " + ".join(n for n, e in zip(LABELS, eps) if e)
            fig, _ = plot_node_pair_planes(
                basis, macro_strain=eps, fe_stress=fe,
                title=f"ls_{ls} ({lbl}): FE vs superposition, {len(sets)} RVEs",
            )
            p = out / f"validation_ls{ls}.png"
            fig.savefig(p, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print("  saved", p.name)

    sigma_node = np.array([60.0, 20.0, -10.0, 15.0, 0.0, 5.0]) * 1e6
    t = f"node: σ_macro = {np.array2string(sigma_node / 1e6, precision=0)} MPa"
    fig, _ = plot_node_pair_planes(basis, macro_stress=sigma_node, title=t)
    fig.savefig(out / "node_pairs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_node_pair_planes(basis, macro_stress=sigma_node, pairs=MAIN_PAIRS, ncols=3,
                                   rve_contours=12, cell_size=(4.2, 4.2),
                                   title=t + " — single-RVE spread")
    fig.savefig(out / "node_pairs_rves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved node_pairs.png, node_pairs_rves.png")

    # 3D density surfaces: same pairs, same in-plane scale, shared density axis
    fig = plot_node_pair_surfaces(basis, macro_stress=sigma_node, pairs=MAIN_PAIRS,
                                  title=t + " — joint density surfaces")
    fig.savefig(out / "node_pair_surfaces.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # how the density changes with the loading direction
    loads = [("uniaxial σxx", [80, 0, 0, 0, 0, 0]),
             ("σxx + σxy", [60, 0, 0, 40, 0, 0]),
             ("pure shear σxy", [0, 0, 0, 80, 0, 0]),
             ("biaxial σxx = σyy", [60, 60, 0, 0, 0, 0])]
    fig = plot_pair_surfaces_vs_load(
        basis, [(lbl, np.array(st, float) * 1e6) for lbl, st in loads], pairs=[(0, 1), (0, 3)],
        title="p(σxx, σyy) and p(σxx, σxy) under different macro loads")
    fig.savefig(out / "pair_surfaces_vs_load.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved node_pair_surfaces.png, pair_surfaces_vs_load.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "result-10-5.hdf5"))
