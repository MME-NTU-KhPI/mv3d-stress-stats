"""
Copula-based and marginal Monte Carlo reconstruction of stress tensors.

Allows reconstruction of joint stress distributions from marginal distributions:
1. Independent sampling (baseline naive assumption)
2. Gaussian copula (preserves correlation/dependence structure between components)
3. Quantitative comparison between raw joint samples, copula samples, and independent samples.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy import stats

from mv3d_stress_stats.invariants import resolve_quantity


@dataclass
class CopulaReconstructionResult:
    """Comparison results between raw joint samples and reconstructed samples."""
    quantity_name: str
    n_samples: int
    raw_mean: float
    raw_std: float
    copula_mean: float
    copula_std: float
    indep_mean: float
    indep_std: float
    correlation_matrix: np.ndarray


def fit_gaussian_copula(stress_samples: np.ndarray) -> Tuple[np.ndarray, list]:
    """Fit a Gaussian copula to 6-component stress samples.
    
    Args:
        stress_samples: Array of shape (N, 6).
        
    Returns:
        (corr_matrix, marginal_values):
            corr_matrix: (6, 6) correlation matrix in normal score space.
            marginal_values: list of 6 sorted arrays for empirical quantile lookup.
    """
    s = np.asarray(stress_samples).reshape(-1, 6)
    n = len(s)
    
    # 1. Transform each marginal to uniform via empirical CDF (ranks)
    # Using rank / (N + 1) to avoid 0 and 1
    ranks = np.empty_like(s, dtype=np.float64)
    marginals = []
    
    for i in range(6):
        col = s[:, i]
        order = np.argsort(col)
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(1, n + 1) / (n + 1.0)
        ranks[:, i] = rank
        marginals.append(np.sort(col))
        
    # 2. Transform uniforms to standard normal scores Z_i = Phi^{-1}(U_i)
    z_scores = stats.norm.ppf(ranks)
    
    # 3. Estimate correlation matrix in Gaussian space
    corr_matrix = np.corrcoef(z_scores, rowvar=False)
    # Ensure positive semi-definite
    corr_matrix = (corr_matrix + corr_matrix.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)
    
    return corr_matrix, marginals


def sample_copula(
    corr_matrix: np.ndarray,
    marginals: list,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic 6-component stress samples using Gaussian copula.
    
    Args:
        corr_matrix: (6, 6) correlation matrix.
        marginals: List of 6 sorted arrays representing empirical marginal distributions.
        n_samples: Number of samples to generate.
        seed: Optional RNG seed.
        
    Returns:
        np.ndarray of shape (n_samples, 6).
    """
    rng = np.random.default_rng(seed)
    # 1. Sample correlated Gaussian variates
    z = rng.multivariate_normal(
        mean=np.zeros(6),
        cov=corr_matrix,
        size=n_samples,
    )
    
    # 2. Transform to uniform [0, 1]
    u = stats.norm.cdf(z)
    
    # 3. Invert marginal empirical quantiles
    out = np.empty((n_samples, 6), dtype=np.float64)
    for i in range(6):
        sorted_col = marginals[i]
        n_m = len(sorted_col)
        indices = np.clip((u[:, i] * n_m).astype(np.intp), 0, n_m - 1)
        out[:, i] = sorted_col[indices]
        
    return out


def sample_independent(
    marginals: list,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic stress samples assuming complete independence between components."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 6), dtype=np.float64)
    for i in range(6):
        sorted_col = marginals[i]
        n_m = len(sorted_col)
        idx = rng.integers(0, n_m, size=n_samples)
        out[:, i] = sorted_col[idx]
    return out


def evaluate_copula_impact(
    stress_samples: np.ndarray,
    quantity: Union[str, Callable[[np.ndarray], np.ndarray]] = "von_mises",
    n_synthetic: int = 50000,
    seed: Optional[int] = 42,
) -> CopulaReconstructionResult:
    """Evaluate difference between raw joint samples, copula reconstruction, and independent marginals.
    
    Demonstrates the principle in AGENTS.md Section 7: why raw joint stress samples
    must be preferred over independent marginals when computing non-linear derived quantities.
    """
    q_fn = resolve_quantity(quantity)
    q_name = quantity if isinstance(quantity, str) else "custom_quantity"
    
    raw = np.asarray(stress_samples).reshape(-1, 6)
    q_raw = q_fn(raw)
    
    corr_mat, marginals = fit_gaussian_copula(raw)
    
    # Reconstructed samples
    copula_stress = sample_copula(corr_mat, marginals, n_samples=n_synthetic, seed=seed)
    q_copula = q_fn(copula_stress)
    
    indep_stress = sample_independent(marginals, n_samples=n_synthetic, seed=seed)
    q_indep = q_fn(indep_stress)
    
    return CopulaReconstructionResult(
        quantity_name=q_name,
        n_samples=len(raw),
        raw_mean=float(np.mean(q_raw)),
        raw_std=float(np.std(q_raw)),
        copula_mean=float(np.mean(q_copula)),
        copula_std=float(np.std(q_copula)),
        indep_mean=float(np.mean(q_indep)),
        indep_std=float(np.std(q_indep)),
        correlation_matrix=corr_mat,
    )
