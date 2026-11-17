"""
How the 1D local-stress distribution p(Q | node) changes while the macro
load turns at constant magnitude from one pure component to another:

    σ_macro(θ) = S · (cos θ · e_i + sin θ · e_j),   θ = 0° … 90°

Figures (figures/direction_sweep/):
  sweep_<i>_<j>_<quantity>.png        absolute values of Q
  sweep_<i>_<j>_<quantity>_eta.png    η = Q / Q(macro): change of shape only

Usage:
    python examples/run_direction_sweep.py [result-10-5.hdf5] [S_MPa]
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

from mv3d_stress_stats.localization import LocalizationBasis, direction_sweep
from mv3d_stress_stats.plotting import plot_direction_sweep

NAMES = ["sx", "sy", "sz", "sxy", "syz", "sxz"]
PATHS = [(0, 1), (0, 3)]                       # σx -> σy,  σx -> σxy
QUANTITIES = ["von_mises", "principal_1", "hydrostatic_stress"]


def main(path: str, magnitude_mpa: float = 100.0) -> None:
    out = ROOT / "figures" / "direction_sweep"
    out.mkdir(parents=True, exist_ok=True)
    basis = LocalizationBasis.from_file_cached(path)
    thetas = np.linspace(0.0, 90.0, 91)
    for i, j in PATHS:
        for q in QUANTITIES:
            tag = f"sweep_{NAMES[i]}_{NAMES[j]}_{q}"
            sw = direction_sweep(basis, q, i, j, magnitude_mpa * 1e6, thetas)
            fig = plot_direction_sweep(sw)
            fig.savefig(out / f"{tag}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            k = [0, 45, 90]
            print(f"{tag:42s} Q(macro) at 0/45/90° = "
                  + " / ".join(f"{sw.q_macro[a] / 1e6:6.1f}" for a in k)
                  + "  CoV % = " + " / ".join(f"{100 * sw.std[a] / abs(sw.mean[a]):5.1f}" for a in k))
            try:
                sw_n = direction_sweep(basis, q, i, j, magnitude_mpa * 1e6, thetas, normalize=True)
            except ValueError as exc:  # e.g. hydrostatic stress vanishes under pure shear
                print(f"  {tag}: no normalized plot ({exc})")
                continue
            fig = plot_direction_sweep(sw_n)
            fig.savefig(out / f"{tag}_eta.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "result-10-5.hdf5"),
         float(sys.argv[2]) if len(sys.argv) > 2 else 100.0)
