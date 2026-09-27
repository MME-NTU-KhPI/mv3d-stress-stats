"""
Single-RVE spatial statistical analysis module.

Quantifies spatial heterogeneity inside an individual RVE realization:
- Summary statistics (mean, variance, std, min, max, percentiles)
- Spatial probability density function (histogram & KDE)
- Empirical cumulative distribution function (CDF)
- Quantiles
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats

from mv3d_stress_stats.invariants import resolve_quantity
from mv3d_stress_stats.reader import RVEReader, RVEStressData


@dataclass
class RVEStatistics:
    """Summary statistics for a scalar quantity inside one RVE."""
    mean: float
    std: float
    var: float
    min: float
    max: float
    q01: float
    q05: float
    q10: float
    q25: float
    q50: float  # median
    q75: float
    q90: float
    q95: float
    q99: float
    skewness: float
    kurtosis: float
    num_samples: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_statistics(values: np.ndarray, weights: Optional[np.ndarray] = None) -> RVEStatistics:
    """Compute summary statistics for a 1D array of spatial samples.
    
    Args:
        values: Array of scalar values (flattened automatically).
        weights: Optional spatial weights (e.g. voxel volume fraction).
        
    Returns:
        RVEStatistics dataclass.
    """
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    
    if len(vals) == 0:
        raise ValueError("Values array contains no finite numbers.")
        
    n = len(vals)
    if weights is not None:
        w = np.asarray(weights).ravel()
        w = w / np.sum(w)
        mean_val = float(np.sum(w * vals))
        var_val = float(np.sum(w * (vals - mean_val) ** 2))
        std_val = float(np.sqrt(var_val))
        # Weighted percentiles
        sorter = np.argsort(vals)
        sorted_vals = vals[sorter]
        cum_w = np.cumsum(w[sorter])
        
        def weighted_percentile(p: float) -> float:
            idx = np.searchsorted(cum_w, p / 100.0)
            return float(sorted_vals[min(idx, n - 1)])
            
        q01 = weighted_percentile(1.0)
        q05 = weighted_percentile(5.0)
        q10 = weighted_percentile(10.0)
        q25 = weighted_percentile(25.0)
        q50 = weighted_percentile(50.0)
        q75 = weighted_percentile(75.0)
        q90 = weighted_percentile(90.0)
        q95 = weighted_percentile(95.0)
        q99 = weighted_percentile(99.0)
        skew_val = float(np.sum(w * ((vals - mean_val) / (std_val if std_val > 0 else 1.0)) ** 3))
        kurt_val = float(np.sum(w * ((vals - mean_val) / (std_val if std_val > 0 else 1.0)) ** 4) - 3.0)
    else:
        mean_val = float(np.mean(vals))
        var_val = float(np.var(vals))
        std_val = float(np.std(vals))
        
        qs = np.percentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        q01, q05, q10, q25, q50, q75, q90, q95, q99 = [float(x) for x in qs]
        
        is_nearly_constant = (std_val <= 1e-12) or (abs(mean_val) > 1e-12 and (std_val / abs(mean_val)) < 1e-7)
        if is_nearly_constant:
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = float(stats.skew(vals))
            kurt_val = float(stats.kurtosis(vals))

    return RVEStatistics(
        mean=mean_val,
        std=std_val,
        var=var_val,
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        q01=q01,
        q05=q05,
        q10=q10,
        q25=q25,
        q50=q50,
        q75=q75,
        q90=q90,
        q95=q95,
        q99=q99,
        skewness=skew_val,
        kurtosis=kurt_val,
        num_samples=n,
    )


def compute_histogram(
    values: np.ndarray,
    bins: Union[int, np.ndarray] = 100,
    range: Optional[Tuple[float, float]] = None,
    density: bool = True,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute histogram / empirical PDF of spatial samples.
    
    Args:
        values: Array of scalar values.
        bins: Number of bins or explicit bin edges array.
        range: (min, max) range for bins.
        density: If True, normalizes counts so the integral over the range is 1.0 (PDF).
        weights: Optional spatial sample weights.
        
    Returns:
        (hist, bin_edges, bin_centers)
    """
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    
    hist, bin_edges = np.histogram(
        vals,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist, bin_edges, bin_centers


def compute_cdf(
    values: np.ndarray,
    eval_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical cumulative distribution function (CDF).
    
    Args:
        values: Array of scalar samples.
        eval_points: Optional 1D evaluation points. If None, uses sorted data values.
        
    Returns:
        (x_eval, cdf_values)
    """
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    
    if n == 0:
        raise ValueError("Empty array for CDF computation.")
        
    if eval_points is None:
        x_eval = vals_sorted
        cdf_vals = np.arange(1, n + 1) / float(n)
    else:
        x_eval = np.asarray(eval_points)
        cdf_vals = np.searchsorted(vals_sorted, x_eval, side="right") / float(n)
        
    return x_eval, cdf_vals


def compute_kde(
    values: np.ndarray,
    eval_points: np.ndarray,
    bandwidth: Optional[Union[float, str]] = None,
) -> np.ndarray:
    """Evaluate 1D Kernel Density Estimate on given evaluation points.
    
    Args:
        values: Array of scalar samples.
        eval_points: Points at which to evaluate the PDF.
        bandwidth: Optional bandwidth factor or rule ('scott', 'silverman', or float).
        
    Returns:
        pdf_values: Evaluated density on eval_points.
    """
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    
    if len(vals) < 2 or np.all(vals == vals[0]):
        # Degenerate case (e.g. constant field)
        pdf = np.zeros_like(eval_points, dtype=float)
        # Find closest point
        closest = np.argmin(np.abs(eval_points - vals[0]))
        dx = eval_points[1] - eval_points[0] if len(eval_points) > 1 else 1.0
        pdf[closest] = 1.0 / (dx if dx > 0 else 1.0)
        return pdf
        
    kde_estimator = stats.gaussian_kde(vals, bw_method=bandwidth)
    return kde_estimator(eval_points)


class RVESimulation:
    """Representation of an individual RVE realization."""

    def __init__(
        self,
        source: Union[str, Path, np.ndarray, RVEStressData],
        set_index: int = 0,
        load_step: int = 1,
        realization_id: Optional[str] = None,
        as_grid: bool = False,
    ):
        """Initialize an RVE realization.
        
        Args:
            source: Path to HDF5 file, or existing RVEStressData, or numpy array of shape (..., 6).
            set_index: Set index in HDF5 file (default 0).
            load_step: Load step number (1-based, default 1).
            realization_id: Identifier name for this realization.
            as_grid: If loading from file, whether to load as regular 3D grid.
        """
        self.set_index = set_index
        self.load_step = load_step
        
        if isinstance(source, (str, Path)):
            self.file_path = str(source)
            self.reader = RVEReader(source)
            if realization_id:
                self.id = realization_id
            elif len(self.reader.set_ids) > 1:
                cur_set = self.reader.set_ids[set_index] if set_index < len(self.reader.set_ids) else str(set_index)
                self.id = f"{Path(source).stem}_s{cur_set}"
            else:
                self.id = Path(source).stem

            self._data = self.reader.read_stress(
                set_index=set_index,
                load_step=load_step,
                as_grid=as_grid,
            )
            self.stress = self._data.stress
            self.coordinates = self._data.coordinates
            self.metadata = self._data.metadata or {}
        elif isinstance(source, RVEStressData):
            self.file_path = source.file_path
            self.id = realization_id or f"rve_{source.set_id}_ls{source.load_step}"
            self._data = source
            self.stress = source.stress
            self.coordinates = source.coordinates
            self.metadata = source.metadata or {}
        elif isinstance(source, np.ndarray):
            self.file_path = ""
            self.id = realization_id or "rve_in_memory"
            self._data = None
            self.stress = source
            self.coordinates = None
            self.metadata = {}
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    @property
    def num_points(self) -> int:
        """Total number of spatial integration / voxel points."""
        return int(np.prod(self.stress.shape[:-1]))

    def get_quantity(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]],
    ) -> np.ndarray:
        """Calculate scalar quantity across all spatial points.
        
        Returns:
            np.ndarray of shape matching spatial grid or flat samples.
        """
        fn = resolve_quantity(quantity)
        return fn(self.stress)

    def describe(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]],
    ) -> RVEStatistics:
        """Compute spatial summary statistics for the chosen quantity."""
        vals = self.get_quantity(quantity)
        return compute_statistics(vals)

    def histogram(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]],
        bins: Union[int, np.ndarray] = 100,
        range: Optional[Tuple[float, float]] = None,
        density: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spatial histogram / PDF for the chosen quantity."""
        vals = self.get_quantity(quantity)
        return compute_histogram(vals, bins=bins, range=range, density=density)

    def cdf(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]],
        eval_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial CDF for the chosen quantity."""
        vals = self.get_quantity(quantity)
        return compute_cdf(vals, eval_points=eval_points)

    def kde(
        self,
        quantity: Union[str, Callable[[np.ndarray], np.ndarray]],
        eval_points: np.ndarray,
        bandwidth: Optional[Union[float, str]] = None,
    ) -> np.ndarray:
        """Compute spatial KDE for the chosen quantity."""
        vals = self.get_quantity(quantity)
        return compute_kde(vals, eval_points=eval_points, bandwidth=bandwidth)

    def homogenized_stress(self) -> np.ndarray:
        """Macroscopic mean stress tensor bar(sigma) = E_X[sigma].
        
        Returns:
            Array of shape (6,): [SX, SY, SZ, SXY, SYZ, SXZ]
        """
        flat_stress = self.stress.reshape(-1, 6)
        return np.mean(flat_stress, axis=0)
