"""
Modality diagnostics for RVE stress distributions.

A spatial distribution p(Q | R=r, L) is sometimes bimodal (e.g. elastically
"soft" vs "hard" oriented grain populations under a given loading direction).
Averaging/binning procedures elsewhere in this package are mixture-based
(weighted sums of per-RVE curves, not pooled-point histograms) specifically so
that multimodality is not washed out when combining realizations -- but a
single RVE's own KDE bandwidth can still smooth a real second mode away if it
is not checked for. This module provides two independent diagnostics:

  - peak counting on the KDE curve (cheap, scipy-only, always available)
  - Gaussian-mixture BIC comparison (sharper, optional -- needs scikit-learn)

Neither is run by default; callers opt in explicitly (see
EnsembleAnalysisResult / RVEEnsemble.analyze(detect_modality=True)).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy.signal import find_peaks

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only when sklearn absent
    GaussianMixture = None
    _HAS_SKLEARN = False


@dataclass
class ModalityResult:
    """Result of a modality check on one distribution (one RVE or one ensemble curve)."""
    n_modes_peaks: int
    peak_locations: np.ndarray          # x-positions of detected KDE peaks
    peak_heights: np.ndarray
    gmm_available: bool                 # False if scikit-learn is not installed
    gmm_n_components: Optional[int] = None   # best k by BIC, among those tried
    gmm_means: Optional[np.ndarray] = None
    gmm_stds: Optional[np.ndarray] = None
    gmm_weights: Optional[np.ndarray] = None
    bic_scores: Dict[int, float] = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        """True if either diagnostic found more than one mode."""
        peak_says_multi = self.n_modes_peaks > 1
        gmm_says_multi = (self.gmm_n_components or 1) > 1
        return peak_says_multi or gmm_says_multi

    def summary(self) -> str:
        lines = [f"Peak count: {self.n_modes_peaks} mode(s) at {np.round(self.peak_locations, 4).tolist()}"]
        if self.gmm_available:
            rounded_bic = {k: round(v, 1) for k, v in self.bic_scores.items()}
            lines.append(f"GMM-BIC best: {self.gmm_n_components} component(s) (BIC: {rounded_bic})")
        else:
            lines.append("GMM-BIC: unavailable (install scikit-learn for this diagnostic)")
        return "\n".join(lines)


def detect_modes_from_kde(
    kde_eval_points: np.ndarray,
    kde_values: np.ndarray,
    prominence_frac: float = 0.05,
) -> tuple:
    """Count peaks on an already-evaluated KDE curve.

    Args:
        kde_eval_points: x-grid the KDE was evaluated on.
        kde_values: KDE density values on that grid.
        prominence_frac: minimum peak prominence, as a fraction of the curve's
            own peak-to-trough range. Guards against counting noise ripples as
            separate modes; raise this if the curve is noisy, lower it if a
            real shallow second mode is being missed.

    Returns:
        (n_modes, peak_x_locations, peak_heights)
    """
    if len(kde_values) < 3:
        return 0, np.array([]), np.array([])
    vspan = float(np.max(kde_values) - np.min(kde_values))
    prominence = max(prominence_frac * vspan, 1e-12)
    peak_idx, props = find_peaks(kde_values, prominence=prominence)
    return len(peak_idx), kde_eval_points[peak_idx], kde_values[peak_idx]


def fit_gmm_bic(
    values: np.ndarray,
    max_components: int = 3,
    random_state: int = 0,
) -> Dict:
    """Fit 1..max_components-component 1D Gaussian mixtures and pick the best by BIC.

    Returns a dict with keys: n_components, means, stds, weights, bic_scores.
    Raises ImportError if scikit-learn is not installed -- callers should
    check `_HAS_SKLEARN` (or catch ImportError) before calling this directly;
    detect_modality() below handles that for you.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "fit_gmm_bic requires scikit-learn. Install the 'modality' extra: "
            "pip install 'mv3d-stress-ensemble-statistics[modality]'"
        )
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    bic_scores: Dict[int, float] = {}
    best_k, best_gmm = 1, None
    for k in range(1, max_components + 1):
        if k >= len(x):
            break
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=4)
        gmm.fit(x)
        bic = float(gmm.bic(x))
        bic_scores[k] = bic
        if best_gmm is None or bic < bic_scores[best_k]:
            best_k, best_gmm = k, gmm

    order = np.argsort(best_gmm.means_.ravel())
    return {
        "n_components": best_k,
        "means": best_gmm.means_.ravel()[order],
        "stds": np.sqrt(best_gmm.covariances_.ravel())[order],
        "weights": best_gmm.weights_.ravel()[order],
        "bic_scores": bic_scores,
    }


def detect_modality(
    values: np.ndarray,
    kde_eval_points: Optional[np.ndarray] = None,
    kde_values: Optional[np.ndarray] = None,
    max_gmm_components: int = 3,
    prominence_frac: float = 0.05,
) -> ModalityResult:
    """Run both modality diagnostics and package them into one result.

    Args:
        values: raw (spatial) samples for this distribution -- used for GMM-BIC.
        kde_eval_points, kde_values: already-evaluated KDE curve -- used for
            peak counting. If not supplied, peak counting is skipped
            (n_modes_peaks will be 0) and only GMM-BIC runs.
    """
    if kde_eval_points is not None and kde_values is not None:
        n_peaks, peak_x, peak_h = detect_modes_from_kde(
            kde_eval_points, kde_values, prominence_frac=prominence_frac
        )
    else:
        n_peaks, peak_x, peak_h = 0, np.array([]), np.array([])

    if _HAS_SKLEARN:
        gmm_res = fit_gmm_bic(values, max_components=max_gmm_components)
        return ModalityResult(
            n_modes_peaks=n_peaks,
            peak_locations=peak_x,
            peak_heights=peak_h,
            gmm_available=True,
            gmm_n_components=gmm_res["n_components"],
            gmm_means=gmm_res["means"],
            gmm_stds=gmm_res["stds"],
            gmm_weights=gmm_res["weights"],
            bic_scores=gmm_res["bic_scores"],
        )
    else:
        return ModalityResult(
            n_modes_peaks=n_peaks,
            peak_locations=peak_x,
            peak_heights=peak_h,
            gmm_available=False,
        )
