"""
Ensemble statistical analysis of RVE simulations.

Combines spatial variability within each RVE with realization-to-realization
variability using mixture distributions and the law of total variance:
    Var(Q) = E_R[Var(Q|R)] + Var_R(E[Q|R])
           = V_within + V_between
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from mv3d_stress_stats.single_rve import (
    RVESimulation,
    RVEStatistics,
    compute_statistics,
    compute_histogram,
    compute_cdf,
    compute_kde,
)
from mv3d_stress_stats.invariants import resolve_quantity


@dataclass
class VarianceDecomposition:
    """Variance decomposition according to the law of total variance."""
    v_within: float     # E_R[Var(Q|R)]: within-RVE variance (spatial heterogeneity)
    v_between: float    # Var_R(E[Q|R]): between-RVE variance (realization uncertainty)
    v_total: float      # V_within + V_between
    eta_within: float   # V_within / V_total
    eta_between: float  # V_between / V_total
    mean_of_means: float
    std_of_means: float # sqrt(V_between)
    rve_means: np.ndarray
    rve_variances: np.ndarray

    def summary(self) -> str:
        pct_within = self.eta_within * 100.0
        pct_between = self.eta_between * 100.0
        return (
            f"Variance Decomposition:\n"
            f"  Total Variance (V_total):   {self.v_total:.6e}\n"
            f"  Within-RVE (V_within):      {self.v_within:.6e} ({pct_within:.2f}%)\n"
            f"  Between-RVE (V_between):    {self.v_between:.6e} ({pct_between:.2f}%)\n"
            f"  Mean of RVE Means:          {self.mean_of_means:.6e}\n"
            f"  Std of RVE Means:           {self.std_of_means:.6e}"
        )


@dataclass
class EnsembleAnalysisResult:
    """Complete result of an ensemble statistical analysis."""
    quantity_name: str
    load_step: int
    set_index: int
    weights: np.ndarray
    rve_stats: List[RVEStatistics]
    rve_ids: List[str]
    variance_decomposition: VarianceDecomposition
    # Common-bin histograms
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    histograms: List[np.ndarray]   # Per-RVE spatial PDFs
    ensemble_pdf: np.ndarray       # Weighted sum of per-RVE PDFs
    ensemble_cdf: np.ndarray       # Weighted sum of per-RVE CDFs
    # Optional KDE
    kde_eval_points: Optional[np.ndarray] = None
    rve_kdes: Optional[List[np.ndarray]] = None
    ensemble_kde: Optional[np.ndarray] = None
    # Reproducibility metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-RVE summary statistics as a pandas DataFrame."""
        rows = []
        for r_id, w, st in zip(self.rve_ids, self.weights, self.rve_stats):
            d = st.to_dict()
            d["rve_id"] = r_id
            d["weight"] = float(w)
            rows.append(d)
        df = pd.DataFrame(rows)
        # Move rve_id and weight to front
        cols = ["rve_id", "weight"] + [c for c in df.columns if c not in ("rve_id", "weight")]
        return df[cols]


class RVEEnsemble:
    """Manages an ensemble of RVE realizations for rigorous statistical analysis."""

    def __init__(
        self,
        sources: Sequence[Union[str, Path, RVESimulation, np.ndarray]],
        weights: Optional[Sequence[float]] = None,
        rve_ids: Optional[Sequence[str]] = None,
    ):
        """Initialize an RVE ensemble.
        
        Args:
            sources: List of file paths, RVESimulation instances, or numpy arrays.
            weights: Optional realization weights w_r. If None, equal weights 1/M are used.
            rve_ids: Optional list of identifier names for realizations.
        """
        if not sources:
            raise ValueError("Ensemble sources list cannot be empty.")
            
        self.sources = list(sources)
        self.num_realizations = len(self.sources)

        # Assign / normalize weights
        if weights is None:
            self.weights = np.full(self.num_realizations, 1.0 / self.num_realizations, dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
            if len(w_arr) != self.num_realizations:
                raise ValueError(f"Length of weights ({len(w_arr)}) must match realizations ({self.num_realizations})")
            if np.any(w_arr < 0):
                raise ValueError("Weights must be non-negative.")
            total_w = np.sum(w_arr)
            if total_w <= 0:
                raise ValueError("Sum of weights must be positive.")
            self.weights = w_arr / total_w

        # Assign identifiers
        if rve_ids is not None:
            if len(rve_ids) != self.num_realizations:
                raise ValueError(f"Length of rve_ids ({len(rve_ids)}) must match realizations ({self.num_realizations})")
            self.rve_ids = [str(x) for x in rve_ids]
        else:
            self.rve_ids = []
            for i, s in enumerate(self.sources):
                if isinstance(s, (str, Path)):
                    self.rve_ids.append(Path(s).stem)
                elif isinstance(s, RVESimulation):
                    self.rve_ids.append(s.id)
                else:
                    self.rve_ids.append(f"rve_{i}")

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        set_indices: Optional[Sequence[int]] = None,
        load_step: int = 1,
        as_grid: bool = False,
        weights: Optional[Sequence[float]] = None,
    ) -> RVEEnsemble:
        """Create an RVEEnsemble from sets within a single HDF5 file (e.g. multi-realization runs)."""
        from mv3d_stress_stats.reader import RVEReader
        reader = RVEReader(file_path)
        if set_indices is None:
            set_indices = list(range(len(reader.set_ids)))
        sims = [
            RVESimulation(
                source=file_path,
                set_index=idx,
                load_step=load_step,
                as_grid=as_grid,
            )
            for idx in set_indices
        ]
        return cls(sources=sims, weights=weights)

    def _get_realizations(
        self,
        set_index: int = 0,
        load_step: int = 1,
    ) -> List[RVESimulation]:
        """Materialize RVESimulation objects for the given set and load step."""
        sims: List[RVESimulation] = []
        for i, s in enumerate(self.sources):
            if isinstance(s, RVESimulation):
                if s.file_path and s.load_step != load_step:
                    sim = RVESimulation(
                        source=s.file_path,
                        set_index=s.set_index,
                        load_step=load_step,
                        realization_id=s.id,
                        as_grid=(s.stress.ndim > 2),
                    )
                    sims.append(sim)
                else:
                    sims.append(s)
            else:
                sim = RVESimulation(
                    source=s,
                    set_index=set_index,
                    load_step=load_step,
                    realization_id=self.rve_ids[i],
                )
                sims.append(sim)
        return sims

    def decompose_variance(
        self,
        rve_values: List[np.ndarray],
    ) -> VarianceDecomposition:
        """Compute within-RVE and between-RVE variance decomposition.
        
        Law of total variance:
            Var(Q) = E_R[Var(Q|R)] + Var_R(E[Q|R])
        """
        rve_means = np.array([float(np.mean(vals)) for vals in rve_values], dtype=np.float64)
        rve_vars = np.array([float(np.var(vals)) for vals in rve_values], dtype=np.float64)

        # Within-RVE variance = E_R[Var(Q|R)]
        v_within = float(np.sum(self.weights * rve_vars))

        # Between-RVE variance = Var_R(E[Q|R])
        mean_of_means = float(np.sum(self.weights * rve_means))
        v_between = float(np.sum(self.weights * (rve_means - mean_of_means) ** 2))

        v_total = v_within + v_between
        eta_within = v_within / v_total if v_total > 0 else 1.0
        eta_between = v_between / v_total if v_total > 0 else 0.0
        std_of_means = float(np.sqrt(v_between))

        return VarianceDecomposition(
            v_within=v_within,
            v_between=v_between,
            v_total=v_total,
            eta_within=eta_within,
            eta_between=eta_between,
            mean_of_means=mean_of_means,
            std_of_means=std_of_means,
            rve_means=rve_means,
            rve_variances=rve_vars,
        )

    def analyze(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]] = "von_mises",
        set_index: int = 0,
        load_step: int = 1,
        bins: Union[int, np.ndarray] = 100,
        bin_range: Optional[Tuple[float, float]] = None,
        kde: bool = False,
        kde_points: int = 200,
        kde_bandwidth: Optional[Union[float, str]] = None,
    ) -> EnsembleAnalysisResult:
        """Run complete statistical analysis for a scalar quantity across the ensemble.
        
        Args:
            quantity: Name in standard registry or custom vectorized callable.
            set_index: 0-based set index.
            load_step: 1-based load step.
            bins: Number of common histogram bins or explicit bin edges.
            bin_range: Optional (min, max) for binning. If None, determined globally.
            kde: If True, computes common-grid KDE for each realization and ensemble mixture.
            kde_points: Number of points on common grid for KDE.
            kde_bandwidth: Optional KDE bandwidth parameter.
            
        Returns:
            EnsembleAnalysisResult containing all statistics, decomposition, and PDFs.
        """
        sims = self._get_realizations(set_index=set_index, load_step=load_step)
        q_fn = resolve_quantity(quantity)
        q_name = quantity if isinstance(quantity, str) else getattr(quantity, "__name__", "custom_quantity")

        # 1. Extract values for each RVE
        rve_values: List[np.ndarray] = []
        rve_stats: List[RVEStatistics] = []
        for sim in sims:
            vals = q_fn(sim.stress).ravel()
            vals = vals[np.isfinite(vals)]
            rve_values.append(vals)
            rve_stats.append(compute_statistics(vals))

        # 2. Variance decomposition
        var_decomp = self.decompose_variance(rve_values)

        # 3. Common bins for histograms
        if bin_range is None:
            global_min = min(s.min for s in rve_stats)
            global_max = max(s.max for s in rve_stats)
            if global_min == global_max:
                # Expand range slightly if degenerate
                global_min -= 1.0
                global_max += 1.0
            actual_range = (global_min, global_max)
        else:
            actual_range = bin_range

        if isinstance(bins, int):
            bin_edges = np.linspace(actual_range[0], actual_range[1], bins + 1)
        else:
            bin_edges = np.asarray(bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 4. Per-RVE histograms and ensemble mixture
        histograms: List[np.ndarray] = []
        ensemble_pdf = np.zeros(len(bin_centers), dtype=np.float64)
        for w, vals in zip(self.weights, rve_values):
            h, _ = np.histogram(vals, bins=bin_edges, density=True)
            histograms.append(h)
            ensemble_pdf += w * h

        # 5. Ensemble CDF on bin edges
        # Integration of ensemble PDF
        dx = np.diff(bin_edges)
        cum_prob = np.cumsum(ensemble_pdf * dx)
        ensemble_cdf = np.clip(cum_prob, 0.0, 1.0)

        # 6. Optional KDE
        kde_eval_pts = None
        rve_kdes = None
        ensemble_kde_vals = None
        if kde:
            kde_eval_pts = np.linspace(actual_range[0], actual_range[1], kde_points)
            rve_kdes = []
            ensemble_kde_vals = np.zeros(kde_points, dtype=np.float64)
            for w, vals in zip(self.weights, rve_values):
                k_val = compute_kde(vals, kde_eval_pts, bandwidth=kde_bandwidth)
                rve_kdes.append(k_val)
                ensemble_kde_vals += w * k_val

        # 7. Reproducibility metadata
        metadata = {
            "quantity": q_name,
            "set_index": set_index,
            "load_step": load_step,
            "num_realizations": self.num_realizations,
            "num_spatial_points_per_rve": [len(v) for v in rve_values],
            "bin_range": actual_range,
            "num_bins": len(bin_centers),
        }

        return EnsembleAnalysisResult(
            quantity_name=q_name,
            load_step=load_step,
            set_index=set_index,
            weights=self.weights,
            rve_stats=rve_stats,
            rve_ids=self.rve_ids,
            variance_decomposition=var_decomp,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            histograms=histograms,
            ensemble_pdf=ensemble_pdf,
            ensemble_cdf=ensemble_cdf,
            kde_eval_points=kde_eval_pts,
            rve_kdes=rve_kdes,
            ensemble_kde=ensemble_kde_vals,
            metadata=metadata,
        )

    def homogenized_response(
        self,
        set_index: int = 0,
        load_step: int = 1,
        quantity: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """Compute macroscopic homogenized response across realizations (Question B).
        
        Returns:
            Dictionary containing:
            - macro_stresses: (M, 6) array of bar(sigma)^(r)
            - mean_macro_stress: (6,) weighted ensemble average
            - cov_macro_stress: (6, 6) covariance matrix of homogenized stress across RVEs
            - scalar_values (if quantity specified): (M,) macroscopic scalar values
        """
        sims = self._get_realizations(set_index=set_index, load_step=load_step)
        macro_stresses = np.array([sim.homogenized_stress() for sim in sims]) # (M, 6)
        
        mean_macro = np.sum(self.weights[:, np.newaxis] * macro_stresses, axis=0)
        # Weighted covariance
        diff = macro_stresses - mean_macro
        cov_macro = np.einsum("m,mi,mj->ij", self.weights, diff, diff)

        res: Dict[str, Any] = {
            "macro_stresses": macro_stresses,
            "mean_macro_stress": mean_macro,
            "cov_macro_stress": cov_macro,
        }

        if quantity is not None:
            q_fn = resolve_quantity(quantity)
            macro_scalars = np.array([q_fn(s[np.newaxis, :])[0] for s in macro_stresses])
            mean_scalar = float(np.sum(self.weights * macro_scalars))
            var_scalar = float(np.sum(self.weights * (macro_scalars - mean_scalar) ** 2))
            res["macro_scalar_values"] = macro_scalars
            res["macro_scalar_mean"] = mean_scalar
            res["macro_scalar_std"] = float(np.sqrt(var_scalar))

        return res
