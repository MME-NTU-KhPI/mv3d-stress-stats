"""
Command-line bridge for ANSYS ACT (stage 1).

ACT extensions run IronPython, which cannot import numpy/h5py.  The intended
coupling is therefore file based: the ACT extension writes the macro state of
the selected nodes to CSV, starts CPython with this module via
System.Diagnostics.Process, and reads the resulting CSV back to contour it.

Input CSV (header required; ANSYS component order, engineering shear strain):
    node_id, X, Y, Z, XY, YZ, XZ

Output CSV (one row per node):
    node_id, q_macro, mean, std, min, max, q<p>..., P_gt_<t>...

Units: the basis is in the RVE solver units (Pa). With ``--stress-unit MPa``
(kPa, GPa) input macro stresses and thresholds are read in that unit and all
stress-like outputs are written in it (J2 as unit^2, J3 as unit^3; triaxiality
and Lode quantities are dimensionless). Strains are dimensionless.

The basis is cached as ``<hdf5 stem>.basis-<hash>.npz`` next to the HDF5 file
(or in ``--cache-dir``); use ``--no-cache`` to always rebuild it.

Example:
    python -m mv3d_stress_stats.node_cli --basis result-10-5.hdf5 ^
        --input nodes_strain.csv --kind strain --quantity von_mises ^
        --probs 0.05,0.5,0.95,0.99 --thresholds 2.5e7 --out node_stats.csv ^
        --plot-nodes 12,345 --plot-dir figures/nodes
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from mv3d_stress_stats.localization import LocalizationBasis

_COLS = ["X", "Y", "Z", "XY", "YZ", "XZ"]

STRESS_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}

# power of the stress unit carried by each registry quantity (default 1)
_QUANTITY_DIMENSION = {
    "j2": 2, "j3": 3,
    "triaxiality": 0, "lode_angle": 0, "lode_parameter": 0,
}


def quantity_unit_power(quantity: str) -> int:
    return _QUANTITY_DIMENSION.get(quantity.strip().lower(), 1)


def read_nodes_csv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        hdr = {h.strip().upper(): h for h in (reader.fieldnames or [])}
        missing = [c for c in ["NODE_ID"] + _COLS if c not in hdr]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; need node_id,{','.join(_COLS)}")
        ids, rows = [], []
        for row in reader:
            ids.append(row[hdr["NODE_ID"]].strip())
            rows.append([float(row[hdr[c]]) for c in _COLS])
    return ids, np.asarray(rows, dtype=np.float64).reshape(-1, 6)


def write_stats_csv(path: str, ids: Sequence[str], stats: dict,
                    probs: Sequence[float], thresholds: Sequence[float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = ["node_id", "q_macro", "mean", "std", "min", "max"]
    header += [f"q{p:g}" for p in probs] + [f"P_gt_{t:g}" for t in thresholds]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for j, nid in enumerate(ids):
            w.writerow(
                [nid] + [f"{stats[k][j]:.9g}" for k in ("q_macro", "mean", "std", "min", "max")]
                + [f"{v:.9g}" for v in stats["quantiles"][j]]
                + [f"{v:.6g}" for v in stats["exceedance"][j]]
            )


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()] if s else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", required=True, help="MatViz3D HDF5 with the 6 unit load cases")
    ap.add_argument("--sets", default="", help="comma-separated 0-based set indices (default all)")
    ap.add_argument("--input", required=True, help="CSV with node_id,X,Y,Z,XY,YZ,XZ")
    ap.add_argument("--kind", choices=["strain", "stress"], default="strain",
                    help="macro state in the CSV: engineering strain (EPEL) or stress (S)")
    ap.add_argument("--quantity", default="von_mises")
    ap.add_argument("--probs", default="0.05,0.5,0.95,0.99")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot-nodes", default="", help="node ids to plot p(Q|node) for")
    ap.add_argument("--plot-dir", default="node_plots")
    ap.add_argument("--stress-unit", choices=sorted(STRESS_UNITS), default="Pa",
                    help="unit of input stresses, thresholds and output stresses")
    ap.add_argument("--cache-dir", default="", help="folder for the .npz basis cache")
    ap.add_argument("--no-cache", action="store_true", help="always rebuild the basis from HDF5")
    ap.add_argument("--progress-chunk", type=int, default=2000,
                    help="nodes per progress line (statistics are computed in such chunks)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    say = lambda msg: print(f"[node_cli +{time.perf_counter() - t0:6.2f}s] {msg}", flush=True)
    say(f"python {sys.version.split()[0]}, numpy {np.__version__}")

    sets = [int(x) for x in args.sets.split(",") if x.strip()] or None
    say(f"loading basis {args.basis} (sets: {'all' if sets is None else len(sets)})")
    if args.no_cache:
        basis = LocalizationBasis.from_file(args.basis, set_indices=sets)
        say("basis built from HDF5 (cache disabled)")
    else:
        basis = LocalizationBasis.from_file_cached(args.basis, set_indices=sets,
                                                   cache_dir=args.cache_dir or None)
        say(f"basis ready ({'cache hit' if basis.metadata.get('_from_cache') else 'built from HDF5, cache written'})")
    say(f"basis: {basis.num_realizations} RVEs x {basis.num_points} points")
    ids, state = read_nodes_csv(args.input)
    probs, thr = _floats(args.probs), _floats(args.thresholds)
    say(f"read {len(ids)} nodes ({args.kind}) from {args.input}")

    unit = STRESS_UNITS[args.stress_unit]
    out_scale = unit ** quantity_unit_power(args.quantity)  # Pa^k -> unit^k
    state_si = state * unit if args.kind == "stress" else state
    key = "macro_stresses" if args.kind == "stress" else "macro_strains"
    parts = []
    n = len(ids)
    step = max(1, min(n, args.progress_chunk))
    for j0 in range(0, n, step):
        parts.append(basis.nodal_statistics(args.quantity, probs=probs,
                                            thresholds=[t * out_scale for t in thr],
                                            **{key: state_si[j0:j0 + step]}))
        done = min(n, j0 + step)
        say(f"{args.quantity}: {done}/{n} nodes ({100.0 * done / n:.0f}%)")
    stats = {k: np.concatenate([pt[k] for pt in parts], axis=0) for k in parts[0]} if parts else \
        basis.nodal_statistics(args.quantity, probs=probs, **{key: state_si})
    for k in ("q_macro", "mean", "std", "min", "max", "quantiles"):
        stats[k] = stats[k] / out_scale
    write_stats_csv(args.out, ids, stats, probs, thr)
    say(f"wrote {args.out} ({len(ids)} nodes)")

    wanted = [x.strip() for x in args.plot_nodes.split(",") if x.strip()]
    if wanted:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mv3d_stress_stats.plotting import plot_node_distribution

        out_dir = Path(args.plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for nid in wanted:
            if nid not in ids:
                print(f"  node {nid} not in input, skipped", file=sys.stderr)
                continue
            j = ids.index(nid)
            nd = basis.node_distribution(args.quantity, **(
                {"macro_strain": state[j]} if args.kind == "strain"
                else {"macro_stress": state[j] * unit}))
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_node_distribution(nd, ax=ax, title=f"node {nid}: {args.quantity}")
            p = out_dir / f"node_{nid}_{args.quantity}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            say(f"plot: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
