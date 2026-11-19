"""
Exact reconstruction of local (microscale) stress fields for an arbitrary
macroscopic loading state, by linear superposition of the RVE unit load cases.

Why this module exists
----------------------
For a linear-elastic RVE the local stress at every spatial point is a linear
function of the applied macroscopic strain:

    sigma_local(x) = A(x) : eps_macro                         (6x6 per point)

where column k of A(x) is the local stress field produced by the k-th unit
macro-strain load case.  MatViz3D stores exactly these unit load cases as
load steps ls_1..ls_6 (``eps_as_loading`` = one non-zero component each), and
the combined steps ls_7..ls_21 of the reference files are reproduced by
superposition to float32 round-off (~1e-7 relative).

The distribution of a scalar Q = g(sigma) for a combined loading state can
therefore NOT be obtained by rescaling or mixing the scalar PDFs of the unit
directions (g is non-linear: von Mises, principal stresses, ...).  The correct
path, used here, is

    eps_macro (or sigma_macro) at a macro node
        -> sigma_local^(r)(x) = A^(r)(x) eps_macro     for every RVE r, point x
        -> Q^(r)(x) = g(sigma_local^(r)(x))
        -> p(Q | node) = sum_r w_r p(Q | R=r, node)

Strain convention
-----------------
All macro strains in this module are **engineering Voigt strains in ANSYS
order**  [eps_x, eps_y, eps_z, gamma_xy, gamma_yz, gamma_xz]
(gamma = 2 * tensor shear), i.e. exactly what ANSYS reports as EPEL.
Stress order is [SX, SY, SZ, SXY, SYZ, SXZ].

MatViz3D stores the shear entries of ``eps_as_loading`` as TENSOR strains
(the nodal EpsXY columns of a unit shear case contain 2x the stored value).
The convention is detected automatically from the nodal strain columns and
can be forced with ``shear_in_file="tensor" | "engineering"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from mv3d_stress_stats.invariants import resolve_quantity

COMPONENTS = ("X", "Y", "Z", "XY", "YZ", "XZ")
_STRESS_COLS = slice(7, 13)
_STRAIN_COLS = slice(13, 19)

QuantityLike = Union[str, Callable[[np.ndarray], np.ndarray]]


# --------------------------------------------------------------------------
# Weighted mixture helpers
# --------------------------------------------------------------------------
def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probs: Sequence[float]
) -> np.ndarray:
    """Quantiles of a weighted empirical distribution (linear interpolation
    of the mid-point CDF, consistent with numpy's default for equal weights)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw_mid = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.asarray(probs, dtype=np.float64), cw_mid, v)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class NodeDistribution:
    """Distribution of a scalar quantity at one macroscopic node."""

    quantity: str
    macro_strain: np.ndarray            # (6,) engineering Voigt strain used
    values: np.ndarray                  # (M, N) local values per realization
    weights: np.ndarray                 # (M,) realization weights
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    rve_pdfs: np.ndarray                # (M, n_bins) p(Q | R=r, node)
    pdf: np.ndarray                     # (n_bins,) mixture p(Q | node)
    cdf: np.ndarray                     # (n_bins,) mixture CDF at right bin edges
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def sample_weights(self) -> np.ndarray:
        """Per-sample weights w_r / N_r giving the realization-weighted mixture."""
        m, n = self.values.shape
        return np.repeat(self.weights / n, n).reshape(m, n)

    def quantiles(self, probs: Sequence[float]) -> np.ndarray:
        return weighted_quantiles(self.values, self.sample_weights, probs)

    def exceedance(self, threshold: float) -> float:
        """P(Q > threshold | node) under the realization-weighted mixture."""
        return float(np.sum(self.weights * np.mean(self.values > threshold, axis=1)))


@dataclass
class SuperpositionCheck:
    """Validation of the localization basis against FE load steps."""

    load_step: int
    macro_strain: np.ndarray
    max_rel_error: float               # max |pred - FE| / max |FE| over all points/sets


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------
# Quantities that are linear (Q = l·σ) or the square root / identity of a
# quadratic form (Q² = σᵀPσ) in the stress. For them Q over all points of all
# RVEs is a single BLAS product with a per-point operator instead of building
# the (nodes, points, 6) local stress array. The operators are derived from the
# registry functions themselves (probing + verification), so they cannot
# drift from invariants.py.
# Macro states with |component| above this are treated as missing data (the
# ANSYS result reader returns ~1.8e308 where it has no value): NaN statistics.
_NO_DATA_LIMIT = 1e30

_LINEAR_QUANTITIES = {"sx", "sy", "sz", "sxy", "syz", "sxz", "i1", "trace",
                      "mean_stress", "hydrostatic_stress", "pressure"}
_QUADRATIC_QUANTITIES = {"j2": "identity", "von_mises": "sqrt", "seqv": "sqrt"}


def _probe_linear(q_fn) -> Optional[np.ndarray]:
    eye = np.eye(6)
    l = np.asarray(q_fn(eye), dtype=np.float64).reshape(6)
    r = np.random.default_rng(7).normal(size=(16, 6))
    ok = np.allclose(np.asarray(q_fn(r), dtype=np.float64), r @ l, rtol=1e-10, atol=1e-12)
    return l if ok else None


def _probe_quadratic(q_fn, kind: str) -> Optional[np.ndarray]:
    sq = (lambda v: v) if kind == "identity" else (lambda v: v ** 2)
    f = lambda s: sq(np.asarray(q_fn(s), dtype=np.float64))
    eye = np.eye(6)
    P = np.empty((6, 6))
    d = f(eye)
    for i in range(6):
        for j in range(6):
            P[i, j] = d[i] if i == j else 0.5 * (f(eye[i] + eye[j]) - d[i] - d[j])
    r = np.random.default_rng(7).normal(size=(16, 6))
    ok = np.allclose(f(r), np.einsum("ni,ij,nj->n", r, P, r), rtol=1e-10, atol=1e-12)
    return P if ok else None


class LocalizationBasis:
    """Per-point 6x6 stress-localization operators for an ensemble of RVEs.

    Attributes:
        A: (M, N, 6, 6) array. ``A[r, i] @ eps`` is the local stress (Pa) at
           point i of realization r for engineering macro strain ``eps``.
        weights: (M,) realization weights (sum to 1).
        coordinates: (N, 3) grid indices of the points (same for all sets).
        unit_steps: load step number used for each strain component.
    """

    def __init__(
        self,
        A: np.ndarray,
        weights: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        rve_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 4 or A.shape[-2:] != (6, 6):
            raise ValueError(f"A must have shape (M, N, 6, 6), got {A.shape}")
        self.A = A
        m = A.shape[0]
        if weights is None:
            weights = np.full(m, 1.0 / m)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (m,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be non-negative, shape (M,), with positive sum")
        self.weights = weights / weights.sum()
        self.coordinates = coordinates
        self.rve_ids = rve_ids or [f"rve_{i}" for i in range(m)]
        self.metadata = metadata or {}

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        set_indices: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        shear_in_file: str = "auto",
    ) -> "LocalizationBasis":
        """Build the basis from a MatViz3D HDF5 file.

        Each set is one RVE realization. In every set the six single-component
        unit load steps are located by their ``eps_as_loading`` vector (any
        order), so files with extra combined steps are handled.

        Args:
            file_path: HDF5 file.
            set_indices: 0-based indices of sets to use (default: all).
            weights: realization weights (default: equal).
            shear_in_file: how shear entries of ``eps_as_loading`` are stored:
                "tensor", "engineering", or "auto" (detected from nodal strains).
        """
        file_path = str(Path(file_path).resolve())
        with h5py.File(file_path, "r") as f:
            set_ids = sorted(
                [k for k in f.keys() if isinstance(f[k], h5py.Group) and k != "last_set"],
                key=lambda s: int(s) if s.isdigit() else s,
            )
            if set_indices is None:
                set_indices = list(range(len(set_ids)))
            chosen = [set_ids[i] for i in set_indices]

            unit_steps, shear_factor = _find_unit_steps(f[chosen[0]], shear_in_file)

            A_list = []
            coords_ref = None
            for sid in chosen:
                grp = f[sid]
                cols = []
                for k in range(6):
                    ls = unit_steps[k]
                    res = grp[f"ls_{ls}/results"][:]
                    eps = grp[f"ls_{ls}/eps_as_loading"][:].astype(np.float64)
                    amp = eps[k] * (shear_factor if k >= 3 else 1.0)  # engineering amplitude
                    if coords_ref is None:
                        coords_ref = res[:, 1:4].copy()
                    elif not np.array_equal(res[:, 1:4], coords_ref):
                        raise ValueError(
                            f"Set {sid} ls_{ls}: node ordering differs from the first set; "
                            "superposition across load steps requires identical node order."
                        )
                    cols.append(res[:, _STRESS_COLS].astype(np.float64) / amp)
                A_list.append(np.stack(cols, axis=-1))  # (N, 6 stress, 6 strain)

        meta = {
            "file_path": file_path,
            "set_ids": chosen,
            "unit_load_steps": unit_steps,
            "shear_in_file": "tensor" if shear_factor == 2.0 else "engineering",
            "strain_convention": "engineering Voigt [x, y, z, xy, yz, xz]",
        }
        return cls(
            np.stack(A_list, axis=0),
            weights=None if weights is None else np.asarray(weights, dtype=np.float64),
            coordinates=coords_ref,
            rve_ids=[f"{Path(file_path).stem}_s{s}" for s in chosen],
            metadata=meta,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Store the basis as a compressed-free .npz (fast to reload)."""
        import json

        np.savez(
            str(path),
            A=self.A,
            weights=self.weights,
            coordinates=np.zeros((0, 3)) if self.coordinates is None else self.coordinates,
            rve_ids=np.asarray(self.rve_ids),
            metadata=np.asarray(json.dumps(self.metadata, default=str)),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LocalizationBasis":
        import json

        with np.load(str(path), allow_pickle=False) as d:
            coords = d["coordinates"]
            return cls(
                d["A"],
                weights=d["weights"],
                coordinates=None if coords.size == 0 else coords,
                rve_ids=[str(s) for s in d["rve_ids"]],
                metadata=json.loads(str(d["metadata"])),
            )

    @classmethod
    def from_file_cached(
        cls,
        file_path: Union[str, Path],
        set_indices: Optional[Sequence[int]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "LocalizationBasis":
        """Like :meth:`from_file`, but reuse a .npz cache keyed by the HDF5
        file's size/mtime and the chosen sets. The cache lives next to the
        HDF5 file (or in ``cache_dir``); failures to write it are ignored."""
        import hashlib

        src = Path(file_path).resolve()
        st = src.stat()
        key = f"{src}|{st.st_size}|{st.st_mtime_ns}|{list(set_indices) if set_indices is not None else 'all'}"
        tag = hashlib.sha1(key.encode()).hexdigest()[:12]
        folder = Path(cache_dir) if cache_dir else src.parent
        cache = folder / f"{src.stem}.basis-{tag}.npz"
        if cache.exists():
            try:
                b = cls.load(cache)
                b.metadata["_from_cache"] = str(cache)
                return b
            except Exception:
                pass
        basis = cls.from_file(src, set_indices=set_indices)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            basis.save(tmp)
            tmp.replace(cache)
        except OSError:
            pass
        return basis

    # ------------------------------------------------------------ basics
    @property
    def num_realizations(self) -> int:
        return self.A.shape[0]

    @property
    def num_points(self) -> int:
        return self.A.shape[1]

    def rve_stiffness(self) -> np.ndarray:
        """(M, 6, 6) apparent stiffness of each realization: volume average of A."""
        return self.A.mean(axis=1)

    def effective_stiffness(self) -> np.ndarray:
        """(6, 6) ensemble effective stiffness C_eff = sum_r w_r <A^(r)>,
        engineering Voigt convention (same as ANSYS / MatViz3D C_matrix)."""
        return np.einsum("r,rij->ij", self.weights, self.rve_stiffness())

    def macro_strain_from_stress(self, macro_stress: np.ndarray) -> np.ndarray:
        """Engineering macro strain eps = C_eff^-1 sigma for (..., 6) macro stress."""
        s = np.asarray(macro_stress, dtype=np.float64)
        return np.linalg.solve(self.effective_stiffness(), s[..., None])[..., 0]

    def _resolve_strain(self, macro_strain, macro_stress) -> np.ndarray:
        if (macro_strain is None) == (macro_stress is None):
            raise ValueError("Provide exactly one of macro_strain or macro_stress.")
        if macro_strain is not None:
            eps = np.asarray(macro_strain, dtype=np.float64)
        else:
            eps = self.macro_strain_from_stress(macro_stress)
        if eps.shape[-1] != 6:
            raise ValueError("Macro strain/stress must have 6 Voigt components.")
        return eps

    def local_stress(
        self,
        macro_strain: Optional[np.ndarray] = None,
        macro_stress: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Exact local stress field (M, N, 6) for ONE macro state (6,)."""
        eps = self._resolve_strain(macro_strain, macro_stress)
        if eps.ndim != 1:
            raise ValueError("local_stress expects a single (6,) macro state.")
        return self.A @ eps

    # ------------------------------------------------------ validation
    def verify_against_file(
        self, file_path: Optional[Union[str, Path]] = None, shear_in_file: Optional[str] = None
    ) -> List[SuperpositionCheck]:
        """Predict every load step stored in the file by superposition and
        compare with the FE stress field. Requires the basis to have been built
        from the same file / sets."""
        file_path = str(file_path or self.metadata["file_path"])
        factor = 2.0 if (shear_in_file or self.metadata.get("shear_in_file")) == "tensor" else 1.0
        checks: List[SuperpositionCheck] = []
        with h5py.File(file_path, "r") as f:
            set_ids = self.metadata.get("set_ids")
            steps = sorted(
                int(k[3:]) for k in f[set_ids[0]].keys() if k.startswith("ls_")
            )
            for ls in steps:
                eps = f[f"{set_ids[0]}/ls_{ls}/eps_as_loading"][:].astype(np.float64)
                eps[3:] *= factor
                err, scale = 0.0, 0.0
                for r, sid in enumerate(set_ids):
                    fe = f[f"{sid}/ls_{ls}/results"][:, _STRESS_COLS].astype(np.float64)
                    pred = self.A[r] @ eps
                    err = max(err, float(np.abs(pred - fe).max()))
                    scale = max(scale, float(np.abs(fe).max()))
                checks.append(SuperpositionCheck(ls, eps, err / scale if scale > 0 else 0.0))
        return checks

    # ---------------------------------------------- one-node distribution
    def node_distribution(
        self,
        quantity: QuantityLike = "von_mises",
        macro_strain: Optional[np.ndarray] = None,
        macro_stress: Optional[np.ndarray] = None,
        bins: Union[int, np.ndarray] = 80,
        bin_range: Optional[Tuple[float, float]] = None,
    ) -> NodeDistribution:
        """Distribution p(Q | node) for one macro state, realization-weighted."""
        eps = self._resolve_strain(macro_strain, macro_stress)
        q_fn = resolve_quantity(quantity)
        values = np.asarray(q_fn(self.A @ eps), dtype=np.float64)  # (M, N)

        if isinstance(bins, (int, np.integer)):
            lo, hi = bin_range if bin_range is not None else (values.min(), values.max())
            if hi <= lo:  # constant field -> narrow window around the value
                pad = max(abs(lo) * 1e-6, 1e-12)
                lo, hi = lo - pad, hi + pad
            edges = np.linspace(lo, hi, int(bins) + 1)
        else:
            edges = np.asarray(bins, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        n = values.shape[1]
        rve_pdfs = np.empty((values.shape[0], len(centers)))
        for r in range(values.shape[0]):
            counts, _ = np.histogram(values[r], bins=edges)
            # divide by ALL points of the realization (not just in-range ones),
            # so mass outside a user bin_range is not silently re-normalized away
            rve_pdfs[r] = counts / (n * widths)
        pdf = self.weights @ rve_pdfs
        cdf = np.cumsum(pdf * widths)

        nd = NodeDistribution(
            quantity=quantity if isinstance(quantity, str) else getattr(quantity, "__name__", "custom"),
            macro_strain=eps,
            values=values,
            weights=self.weights,
            bin_edges=edges,
            bin_centers=centers,
            rve_pdfs=rve_pdfs,
            pdf=pdf,
            cdf=cdf,
        )
        rve_means = values.mean(axis=1)
        rve_vars = values.var(axis=1)
        mean = float(self.weights @ rve_means)
        v_within = float(self.weights @ rve_vars)
        v_between = float(self.weights @ (rve_means - mean) ** 2)
        q = nd.quantiles([0.01, 0.05, 0.5, 0.95, 0.99])
        nd.stats = {
            "mean": mean,
            "std": float(np.sqrt(v_within + v_between)),
            "min": float(values.min()),
            "max": float(values.max()),
            "q01": float(q[0]), "q05": float(q[1]), "q50": float(q[2]),
            "q95": float(q[3]), "q99": float(q[4]),
            "v_within": v_within,
            "v_between": v_between,
            "eta_between": v_between / (v_within + v_between) if v_within + v_between > 0 else 0.0,
            "q_of_macro_stress": float(np.asarray(q_fn(self.effective_stiffness() @ eps))),
        }
        return nd

    def _fast_evaluator(self, quantity, A: np.ndarray, cache_key):
        """Return f(eps (c,6)) -> Q (c, M*N) using a precomputed per-point
        operator, or None when ``quantity`` has no linear/quadratic form."""
        if not isinstance(quantity, str):
            return None
        key = quantity.strip().lower()
        cache = self.__dict__.setdefault("_fast_ops", {})
        if (key, cache_key) in cache:
            return cache[(key, cache_key)]
        q_fn = resolve_quantity(quantity)
        m, n = A.shape[:2]
        Ap = A.reshape(m * n, 6, 6)
        f = None
        if key in _LINEAR_QUANTITIES:
            l = _probe_linear(q_fn)
            if l is not None:
                L = np.ascontiguousarray(np.einsum("k,pkj->pj", l, Ap))      # (MN, 6)
                f = lambda e: e @ L.T
        elif key in _QUADRATIC_QUANTITIES:
            kind = _QUADRATIC_QUANTITIES[key]
            P = _probe_quadratic(q_fn, kind)
            if P is not None:
                K = np.einsum("pki,kl,plj->pij", Ap, P, Ap, optimize=True)  # (MN, 6, 6)
                K = 0.5 * (K + K.transpose(0, 2, 1))
                iu = np.triu_indices(6)
                mult = np.where(iu[0] == iu[1], 1.0, 2.0)
                Kt = np.ascontiguousarray((K[:, iu[0], iu[1]] * mult).T)    # (21, MN)

                def f(e, Kt=Kt, iu=iu, kind=kind):
                    ee = e[:, iu[0]] * e[:, iu[1]]                          # (c, 21)
                    v = ee @ Kt
                    return v if kind == "identity" else np.sqrt(np.maximum(v, 0.0))
        cache[(key, cache_key)] = f
        return f

    # ---------------------------------------------- many nodes (ACT field)
    def nodal_statistics(
        self,
        quantity: QuantityLike = "von_mises",
        macro_strains: Optional[np.ndarray] = None,
        macro_stresses: Optional[np.ndarray] = None,
        probs: Sequence[float] = (0.05, 0.5, 0.95, 0.99),
        thresholds: Sequence[float] = (),
        realization_subset: Optional[Sequence[int]] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Scalar distribution descriptors for every macro node.

        Args:
            macro_strains / macro_stresses: (Nn, 6) per-node macro state.
            probs: quantile levels to report.
            thresholds: values t for exceedance probabilities P(Q > t | node).
            realization_subset: optionally use only these realizations (speed).
            n_jobs: worker threads (NumPy releases the GIL in the heavy
                element-wise work); default min(4, CPU count). Memory per
                chunk is divided by the thread count.

        Returns:
            dict with arrays: "mean", "std", "min", "max" (Nn,),
            "quantiles" (Nn, len(probs)), "exceedance" (Nn, len(thresholds)),
            "q_macro" (Nn,) = Q evaluated on the homogenized macro stress.
        """
        if macro_stresses is not None:
            macro_stresses = np.atleast_2d(np.asarray(macro_stresses, dtype=np.float64))
            bad = ~(np.abs(macro_stresses) <= _NO_DATA_LIMIT).all(axis=1)  # inf, nan, reader sentinels
            macro_stresses = np.where(bad[:, None], 0.0, macro_stresses)
        else:
            macro_strains = np.atleast_2d(np.asarray(macro_strains, dtype=np.float64))
            bad = ~(np.abs(macro_strains) <= _NO_DATA_LIMIT).all(axis=1)  # inf, nan, reader sentinels
            macro_strains = np.where(bad[:, None], 0.0, macro_strains)
        eps_all = self._resolve_strain(macro_strains, macro_stresses)
        eps_all = np.atleast_2d(eps_all)
        q_fn = resolve_quantity(quantity)
        A, w = self.A, self.weights
        subset_key = None
        if realization_subset is not None:
            idx = np.asarray(realization_subset)
            A, w = A[idx], w[idx] / w[idx].sum()
            subset_key = tuple(int(i) for i in idx)
        fast = self._fast_evaluator(quantity, A, subset_key)
        m, n = A.shape[:2]
        sample_w = np.repeat(w / n, n)
        c_eff = self.effective_stiffness()

        nn = eps_all.shape[0]
        out = {
            "mean": np.empty(nn), "std": np.empty(nn),
            "min": np.empty(nn), "max": np.empty(nn),
            "quantiles": np.empty((nn, len(probs))),
            "exceedance": np.empty((nn, len(thresholds))),
            "q_macro": np.asarray(q_fn(eps_all @ c_eff.T), dtype=np.float64).reshape(nn),
        }
        thr = np.asarray(thresholds, dtype=np.float64)
        equal_w = np.allclose(w, w[0])
        if n_jobs is None:
            n_jobs = min(4, os.cpu_count() or 1)
        n_jobs = max(1, int(n_jobs))
        if fast is not None:
            chunk = max(1, int(2e7 // (m * n) // n_jobs))  # ~160 MB of Q values in total
        else:
            # component-major operator (6 comp, 6 strain, M*N): each stress
            # component of a chunk comes out as one contiguous (c, M*N) block
            Lt = np.ascontiguousarray(A.reshape(m * n, 6, 6).transpose(1, 2, 0))
            chunk = max(1, int(4e7 // (m * n * 6) // n_jobs))  # ~320 MB of stress in total

        def work(j0):
            e = eps_all[j0:j0 + chunk]                                  # (c, 6)
            if fast is not None:
                vals = fast(e)                                          # (c, M*N)
            else:
                comp = np.matmul(e[None], Lt)                           # (6, c, M*N)
                vals = np.asarray(q_fn(np.moveaxis(comp, 0, -1)),
                                  dtype=np.float64).reshape(len(e), m * n)
            mean = vals @ sample_w
            sl = slice(j0, j0 + len(e))
            out["mean"][sl] = mean
            out["std"][sl] = np.sqrt(((vals - mean[:, None]) ** 2) @ sample_w)
            out["min"][sl] = vals.min(axis=1)
            out["max"][sl] = vals.max(axis=1)
            if len(probs) == 0:
                pass
            elif equal_w:
                # 'hazen' == mid-point CDF interpolation used by weighted_quantiles
                out["quantiles"][sl] = np.quantile(vals, probs, axis=1, method="hazen").T
            else:
                out["quantiles"][sl] = np.array(
                    [weighted_quantiles(v, sample_w, probs) for v in vals]
                )
            if thr.size:
                out["exceedance"][sl] = np.stack(
                    [(vals > t) @ sample_w for t in thr], axis=1
                )

        starts = range(0, nn, chunk)
        if n_jobs == 1 or nn <= chunk:
            for j0 in starts:
                work(j0)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                for _ in pool.map(work, starts):  # re-raises worker errors
                    pass
        if bad.any():  # non-finite macro state (e.g. missing result data): no statistics
            for k in out:
                out[k][bad] = np.nan
        return out


# --------------------------------------------------------------------------
# 1D distribution along a rotating macro load direction
# --------------------------------------------------------------------------
@dataclass
class DirectionSweep:
    """p(Q | node) for macro states rotating between two Voigt directions.

    Macro state at angle θ:  s(θ) = magnitude · (cos θ · e_i + sin θ · e_j),
    in stress (``kind="stress"``, Pa) or engineering strain.

    Attributes:
        thetas_deg: (K,) angles.
        bin_edges / bin_centers: common bins of Q for all angles.
        pdf: (K, B) ensemble mixture density sum_r w_r p(Q | R=r, θ).
        quantiles: (K, len(probs)); probs listed in ``probs``.
        mean, std: (K,).
        q_macro: (K,) Q of the homogenized macro stress C_eff·ε(θ).
        between_share: (K,) fraction of Var(Q) from realization-to-realization.
    """
    quantity: str
    component_i: int
    component_j: int
    magnitude: float
    kind: str
    thetas_deg: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    pdf: np.ndarray
    probs: np.ndarray
    quantiles: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    q_macro: np.ndarray
    between_share: np.ndarray
    normalized: bool = False   # True -> values are eta = Q / q_macro(θ)

    def macro_states(self) -> np.ndarray:
        return _direction_states(self.component_i, self.component_j, self.magnitude,
                                 self.thetas_deg)


def _direction_states(i: int, j: int, magnitude: float, thetas_deg) -> np.ndarray:
    t = np.deg2rad(np.asarray(thetas_deg, dtype=np.float64))
    s = np.zeros((t.size, 6))
    s[:, i] += magnitude * np.cos(t)
    s[:, j] += magnitude * np.sin(t)
    return s


def direction_sweep(
    basis: "LocalizationBasis",
    quantity: str = "von_mises",
    component_i: int = 0,
    component_j: int = 1,
    magnitude: float = 100e6,
    thetas_deg: Sequence[float] = tuple(np.linspace(0.0, 90.0, 46)),
    kind: str = "stress",
    bins: int = 120,
    probs: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    value_range: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
) -> DirectionSweep:
    """Ensemble distribution of Q while the macro load turns from pure
    component i (θ=0°) to pure component j (θ=90°) at constant magnitude.

    Exact for linear-elastic RVEs (localization superposition). All angles
    share one set of bins so the densities are directly comparable.

    normalize: report eta = Q / Q(C_eff·ε(θ)) instead of Q, which removes the
        change of the macro value and leaves only the change of shape/spread.
        Raises if Q of the macro stress vanishes at some angle.
    """
    if kind not in ("stress", "strain"):
        raise ValueError("kind must be 'stress' or 'strain'")
    q_fn = resolve_quantity(quantity)
    states = _direction_states(component_i, component_j, magnitude, thetas_deg)
    eps = states if kind == "strain" else basis.macro_strain_from_stress(states)
    c_eff = basis.effective_stiffness()
    m, n = basis.A.shape[:2]
    A2 = np.ascontiguousarray(basis.A.reshape(m * n * 6, 6))
    chunk = max(1, int(2e7 // (m * n * 6)))  # ~160 MB of local stress per chunk
    vals = np.empty((len(eps), m, n))
    for a0 in range(0, len(eps), chunk):
        e = eps[a0:a0 + chunk]
        sig = (A2 @ e.T).reshape(m, n, 6, -1).transpose(3, 0, 1, 2)  # (c, M, N, 6)
        vals[a0:a0 + len(e)] = np.asarray(q_fn(sig), dtype=np.float64)
    w = basis.weights
    q_macro = np.asarray(q_fn(eps @ c_eff.T), dtype=np.float64)
    if normalize:
        tiny = 1e-9 * np.max(np.abs(vals))
        if np.any(np.abs(q_macro) <= tiny):
            raise ValueError(f"{quantity} of the macro stress is ~0 at some angle; "
                             "cannot normalize")
        vals = vals / q_macro[:, None, None]

    lo, hi = value_range if value_range is not None else (vals.min(), vals.max())
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    k = vals.shape[0]
    pdf = np.empty((k, bins))
    for a in range(k):
        h = np.stack([np.histogram(vals[a, r], bins=edges, density=True)[0] for r in range(m)])
        pdf[a] = w @ h

    rve_mean = vals.mean(axis=2)                                      # (K, M)
    rve_var = vals.var(axis=2)
    mean = rve_mean @ w
    within = rve_var @ w
    between = ((rve_mean - mean[:, None]) ** 2) @ w
    total = within + between
    flat = vals.reshape(k, -1)
    if np.allclose(w, w[0]):
        qs = np.quantile(flat, probs, axis=1, method="hazen").T
    else:
        sw = np.repeat(w / n, n)
        qs = np.array([weighted_quantiles(v, sw, probs) for v in flat])
    return DirectionSweep(
        quantity=str(quantity), component_i=component_i, component_j=component_j,
        magnitude=float(magnitude), kind=kind,
        thetas_deg=np.asarray(thetas_deg, dtype=np.float64),
        bin_edges=edges, bin_centers=0.5 * (edges[1:] + edges[:-1]), pdf=pdf,
        probs=np.asarray(probs, dtype=np.float64), quantiles=qs, mean=mean,
        std=np.sqrt(total), q_macro=q_macro, normalized=normalize,
        between_share=np.divide(between, total, out=np.zeros_like(total), where=total > 0),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _find_unit_steps(grp: h5py.Group, shear_in_file: str) -> Tuple[List[int], float]:
    """Locate the 6 single-component load steps and determine shear factor."""
    steps = sorted(int(k[3:]) for k in grp.keys() if k.startswith("ls_"))
    unit: Dict[int, int] = {}
    for ls in steps:
        eps = grp[f"ls_{ls}/eps_as_loading"][:].astype(np.float64)
        nz = np.flatnonzero(np.abs(eps) > 1e-3 * np.abs(eps).max()) if np.any(eps) else []
        if len(nz) == 1 and nz[0] not in unit:
            unit[int(nz[0])] = ls
    missing = [COMPONENTS[k] for k in range(6) if k not in unit]
    if missing:
        raise ValueError(
            f"Group '{grp.name}' lacks single-component unit load steps for {missing}; "
            "a full 6x6 localization basis needs all six."
        )
    unit_steps = [unit[k] for k in range(6)]

    if shear_in_file == "tensor":
        factor = 2.0
    elif shear_in_file == "engineering":
        factor = 1.0
    elif shear_in_file == "auto":
        ls = unit_steps[3]
        res = grp[f"ls_{ls}/results"][:]
        eps_stored = float(grp[f"ls_{ls}/eps_as_loading"][3])
        if res.shape[1] >= 19:
            gamma_nodal = float(np.mean(res[:, _STRAIN_COLS][:, 3]))
            ratio = gamma_nodal / eps_stored
            factor = 2.0 if abs(ratio - 2.0) < abs(ratio - 1.0) else 1.0
        else:  # no strain columns: MatViz3D convention is tensor shear
            factor = 2.0
    else:
        raise ValueError("shear_in_file must be 'auto', 'tensor' or 'engineering'")
    return unit_steps, factor
