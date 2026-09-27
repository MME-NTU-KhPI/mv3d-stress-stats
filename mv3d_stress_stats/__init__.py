"""
mv3d-stress_ensemble_statistics
===============================
A general statistical framework for analyzing stress fields in ensembles of RVE simulations.
Combines spatial heterogeneity inside individual RVEs with realization-to-realization variability.
"""

from mv3d_stress_stats.reader import (
    RVEReader,
    RVEStressData,
)
from mv3d_stress_stats.tensors import (
    to_tensor_3x3,
    from_tensor_3x3,
    rotate_stress,
    deviatoric_stress,
    hydrostatic_stress,
    trace_stress,
)
from mv3d_stress_stats.invariants import (
    i1,
    mean_stress,
    pressure,
    j2,
    j3,
    von_mises,
    principal_stresses,
    principal_1,
    principal_2,
    principal_3,
    lode_angle,
    lode_parameter,
    triaxiality,
    resolve_quantity,
    QUANTITY_REGISTRY,
)
from mv3d_stress_stats.single_rve import (
    RVESimulation,
    RVEStatistics,
    compute_statistics,
    compute_histogram,
    compute_cdf,
    compute_kde,
)
from mv3d_stress_stats.ensemble import (
    RVEEnsemble,
    EnsembleAnalysisResult,
    VarianceDecomposition,
)
from mv3d_stress_stats.joint import (
    JointDistribution2D,
    compute_joint_histogram_2d,
    compute_ensemble_joint_2d,
)
from mv3d_stress_stats.copula import (
    fit_gaussian_copula,
    sample_copula,
    sample_independent,
    evaluate_copula_impact,
    CopulaReconstructionResult,
)
from mv3d_stress_stats.correlation import (
    SpatialCorrelationResult,
    compute_spatial_autocorrelation_3d,
    estimate_effective_sample_size,
)
from mv3d_stress_stats.directional import (
    DirectionalHistogram,
    principal_directions,
    principal_direction_angles,
    compute_circular_statistics,
    compute_directional_histogram,
)
from mv3d_stress_stats.multiload import (
    LoadingCaseInfo,
    MultiloadAnalysisResult,
    identify_loading_case,
    analyze_multiload,
)
from mv3d_stress_stats.plotting import (
    plot_ensemble_pdf,
    plot_variance_decomposition,
    plot_joint_density_2d,
    plot_spatial_autocorrelation,
    plot_summary_dashboard,
    plot_loading_pdf_comparison,
    plot_directional_histogram,
    plot_multiload_directional_grid,
    plot_loading_dashboard,
)

__all__ = [
    # Reader
    "RVEReader",
    "RVEStressData",
    # Tensors
    "to_tensor_3x3",
    "from_tensor_3x3",
    "rotate_stress",
    "deviatoric_stress",
    "hydrostatic_stress",
    "trace_stress",
    # Invariants
    "i1",
    "mean_stress",
    "pressure",
    "j2",
    "j3",
    "von_mises",
    "principal_stresses",
    "principal_1",
    "principal_2",
    "principal_3",
    "lode_angle",
    "lode_parameter",
    "triaxiality",
    "resolve_quantity",
    "QUANTITY_REGISTRY",
    # Single RVE
    "RVESimulation",
    "RVEStatistics",
    "compute_statistics",
    "compute_histogram",
    "compute_cdf",
    "compute_kde",
    # Ensemble
    "RVEEnsemble",
    "EnsembleAnalysisResult",
    "VarianceDecomposition",
    # Joint
    "JointDistribution2D",
    "compute_joint_histogram_2d",
    "compute_ensemble_joint_2d",
    # Copula
    "fit_gaussian_copula",
    "sample_copula",
    "sample_independent",
    "evaluate_copula_impact",
    "CopulaReconstructionResult",
    # Spatial correlation
    "SpatialCorrelationResult",
    "compute_spatial_autocorrelation_3d",
    "estimate_effective_sample_size",
    # Directional
    "DirectionalHistogram",
    "principal_directions",
    "principal_direction_angles",
    "compute_circular_statistics",
    "compute_directional_histogram",
    # Multiload
    "LoadingCaseInfo",
    "MultiloadAnalysisResult",
    "identify_loading_case",
    "analyze_multiload",
    # Plotting
    "plot_ensemble_pdf",
    "plot_variance_decomposition",
    "plot_joint_density_2d",
    "plot_spatial_autocorrelation",
    "plot_summary_dashboard",
    "plot_loading_pdf_comparison",
    "plot_directional_histogram",
    "plot_multiload_directional_grid",
    "plot_loading_dashboard",
]

