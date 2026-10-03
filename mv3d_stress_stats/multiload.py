"""
Multiload statistical analysis module.

Analyzes how different macroscopic loading conditions (e.g. eps_x, eps_y, eps_z,
eps_xy, eps_yz, eps_xz, and multiaxial states) influence the probability
density function, dispersion, and variance decomposition of stress quantities.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from mv3d_stress_stats.ensemble import EnsembleAnalysisResult, RVEEnsemble
from mv3d_stress_stats.single_rve import RVESimulation


@dataclass
class LoadingCaseInfo:
    """Metadata describing an applied loading condition."""
    load_step: int
    applied_strain: Optional[np.ndarray]  # (6,) [eps_x, eps_y, eps_z, eps_xy, eps_yz, eps_xz]
    label: str                            # LaTeX string, e.g. "$\\varepsilon_x$"
    text_label: str                       # ASCII string, e.g. "eps_x"
    category: str                         # "normal", "shear", "biaxial", "combined", "zero"
    strain_norm: float                    # ||eps||


def identify_loading_case(
    applied_strain: Optional[np.ndarray],
    load_step: int,
) -> LoadingCaseInfo:
    """Identify and classify loading condition from applied strain vector."""
    if applied_strain is None:
        return LoadingCaseInfo(
            load_step=load_step,
            applied_strain=None,
            label=f"LS {load_step}",
            text_label=f"ls_{load_step}",
            category="unspecified",
            strain_norm=0.0,
        )
    eps = np.asarray(applied_strain, dtype=np.float64)
    strain_norm = float(np.linalg.norm(eps))
    
    threshold = 1e-7 if strain_norm > 0 else 0.0
    active_indices = [i for i in range(len(eps)) if abs(eps[i]) > threshold]
    
    names_latex = [
        r"\varepsilon_x",
        r"\varepsilon_y",
        r"\varepsilon_z",
        r"\varepsilon_{xy}",
        r"\varepsilon_{yz}",
        r"\varepsilon_{xz}",
    ]
    names_text = ["eps_x", "eps_y", "eps_z", "eps_xy", "eps_yz", "eps_xz"]
    
    if len(active_indices) == 1:
        idx = active_indices[0]
        cat = "normal" if idx < 3 else "shear"
        lbl = f"${names_latex[idx]}$"
        txt = names_text[idx]
    elif len(active_indices) == 0:
        cat = "zero"
        lbl = f"LS {load_step}"
        txt = f"ls_{load_step}"
    else:
        all_normal = all(i < 3 for i in active_indices)
        all_shear = all(i >= 3 for i in active_indices)
        if all_normal:
            cat = "biaxial" if len(active_indices) == 2 else "triaxial"
        elif all_shear:
            cat = "multiaxial_shear"
        else:
            cat = "combined"
        lbl = "$" + " + ".join(names_latex[i] for i in active_indices) + "$"
        txt = "+".join(names_text[i] for i in active_indices)
        
    return LoadingCaseInfo(
        load_step=load_step,
        applied_strain=eps,
        label=lbl,
        text_label=txt,
        category=cat,
        strain_norm=strain_norm,
    )


@dataclass
class MultiloadAnalysisResult:
    """Results of statistical analysis across multiple loading conditions."""
    quantity: str
    cases: Dict[int, LoadingCaseInfo]
    results: Dict[int, EnsembleAnalysisResult]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export tabular summary of statistics and variance decomposition per load step."""
        rows = []
        for ls, case in self.cases.items():
            res = self.results[ls]
            vd = res.variance_decomposition
            row = {
                "load_step": ls,
                "label": case.label,
                "text_label": case.text_label,
                "category": case.category,
                "mean": vd.mean_of_means,
                "std": np.sqrt(vd.v_total),
                "q50": float(np.mean([s.q50 for s in res.rve_stats])),
                "q95": float(np.mean([s.q95 for s in res.rve_stats])),
                "v_within": vd.v_within,
                "v_between": vd.v_between,
                "v_total": vd.v_total,
                "eta_within": vd.eta_within,
                "eta_between": vd.eta_between,
                "normalized": res.is_normalized,
            }
            if res.modality is not None:
                row["n_modes_peaks"] = res.modality.n_modes_peaks
                row["is_multimodal"] = res.modality.is_multimodal
                row["gmm_n_components"] = res.modality.gmm_n_components
            rows.append(row)
        return pd.DataFrame(rows)

    def directional_kde_field(self) -> "Tuple[List[str], np.ndarray, np.ndarray]":
        """Stack per-load-step ensemble KDEs into a (direction x Q) density field.

        Requires analyze_multiload(..., kde=True, shared_range=True) so every
        load step's KDE was evaluated on the same Q grid -- otherwise the rows
        of the returned matrix would not be directly comparable/stackable.

        Returns:
            labels: one text label per load step (direction), e.g. "eps_x".
            q_grid: (n_points,) shared quantity grid.
            density_field: (n_directions, n_points) array; row i is p(Q | L=labels[i]).
        """
        labels: List[str] = []
        q_grid: Optional[np.ndarray] = None
        rows: List[np.ndarray] = []
        for ls, case in self.cases.items():
            res = self.results[ls]
            if res.ensemble_kde is None or res.kde_eval_points is None:
                raise ValueError(
                    f"Load step {ls} has no KDE. Re-run analyze_multiload(..., kde=True)."
                )
            if q_grid is None:
                q_grid = res.kde_eval_points
            elif not np.allclose(q_grid, res.kde_eval_points):
                raise ValueError(
                    f"Load step {ls} was evaluated on a different Q grid than the others. "
                    "Re-run analyze_multiload(..., shared_range=True) so all directions "
                    "share one grid before stacking them into a field."
                )
            labels.append(case.text_label)
            rows.append(res.ensemble_kde)
        return labels, q_grid, np.asarray(rows)

    def anisotropy_index(self, category: str = "normal") -> float:
        """Compute directional anisotropy index for the given loading category.
        
        Calculated as max(mean_stress) / min(mean_stress) among load steps in that category.
        A value of 1.0 indicates isotropic stress response; values > 1.0 indicate directional anisotropy.
        """
        means = [
            self.results[ls].variance_decomposition.mean_of_means
            for ls, case in self.cases.items()
            if case.category == category and self.results[ls].variance_decomposition.mean_of_means > 0
        ]
        if len(means) < 2:
            return 1.0
        return float(np.max(means) / np.min(means))


def analyze_multiload(
    source: Union[RVEEnsemble, RVESimulation, str],
    quantity: str = "von_mises",
    load_steps: Optional[Sequence[int]] = None,
    bins: int = 100,
    kde: bool = True,
    shared_range: bool = True,
    normalize: bool = False,
    detect_modality: bool = False,
) -> MultiloadAnalysisResult:
    """Analyze stress distribution across multiple loading conditions.
    
    Args:
        source: RVEEnsemble, RVESimulation, or path to HDF5 file.
        quantity: Scalar quantity name (default "von_mises").
        load_steps: Sequence of 1-based load step indices to evaluate.
                    If None, evaluates the canonical first 6 steps or all available.
        bins: Number of histogram bins.
        kde: Whether to compute kernel density estimation.
        shared_range: If True (default), first probes every load step to find a
                       single global (min, max) for `quantity`, then reuses that
                       range as `bin_range` for every load step's histogram/KDE.
                       This is required for directional_kde_field()/cut plots to
                       be meaningful -- without it, each direction gets its own
                       independently auto-ranged grid and the per-direction
                       densities cannot be stacked or compared point-for-point.
                       Individual overlaid-line comparisons (e.g.
                       plot_loading_pdf_comparison) still work fine either way.
        normalize: If True, each direction's values are normalized to
                       eta = Q_local / Q_macro before histogram/KDE/stats are
                       computed (see RVEEnsemble.analyze(normalize=...) for the
                       exact definition and its elastic-linearity requirement).
                       This makes different loading AMPLITUDES along the same
                       direction comparable; it does not collapse direction
                       itself -- directional_kde_field() still varies by L.
        detect_modality: If True (forces kde=True), attaches a ModalityResult
                       (peak count + GMM-BIC) to each load step's result and
                       to to_dataframe()'s output, without altering the PDF/KDE.
        
    Returns:
        MultiloadAnalysisResult with per-loading statistical distributions.
    """
    if isinstance(source, (str, RVESimulation)):
        if isinstance(source, str):
            sim = RVESimulation(source, load_step=1)
        else:
            sim = source
        ensemble = RVEEnsemble([sim], rve_ids=[sim.id])
    elif isinstance(source, RVEEnsemble):
        ensemble = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
        
    # Determine load steps to process
    if load_steps is None:
        # Check first realization reader if available
        first_sim = ensemble._get_realizations(load_step=1)[0]
        if hasattr(first_sim, "reader") and first_sim.reader:
            all_steps = first_sim.reader.load_steps(first_sim.set_index)
            # Default to up to 6 canonical steps (normal & shear)
            load_steps = all_steps[:6] if len(all_steps) >= 6 else all_steps
        else:
            load_steps = [1]
            
    cases: Dict[int, LoadingCaseInfo] = {}
    results: Dict[int, EnsembleAnalysisResult] = {}

    global_range: Optional[Tuple[float, float]] = None
    if shared_range:
        # Cheap probe pass (no histogram/KDE) just to learn each direction's
        # (min, max) -- in the SAME units the final pass will use (normalized
        # eta if normalize=True) -- so we can pick one grid that covers all of them.
        mins, maxs = [], []
        for ls in load_steps:
            probe = ensemble.analyze(
                quantity=quantity, load_step=ls, bins=2, kde=False, normalize=normalize
            )
            mins.append(min(s.min for s in probe.rve_stats))
            maxs.append(max(s.max for s in probe.rve_stats))
        global_range = (float(min(mins)), float(max(maxs)))

    for ls in load_steps:
        # Materialize realization at load_step ls
        res = ensemble.analyze(
            quantity=quantity,
            load_step=ls,
            bins=bins,
            bin_range=global_range,
            kde=kde,
            normalize=normalize,
            detect_modality=detect_modality,
        )
        results[ls] = res
        
        # Extract applied strain from first realization
        sims = ensemble._get_realizations(load_step=ls)
        applied_strain = sims[0].applied_strain if sims else None
        cases[ls] = identify_loading_case(applied_strain, load_step=ls)
        
    return MultiloadAnalysisResult(
        quantity=quantity,
        cases=cases,
        results=results,
    )
