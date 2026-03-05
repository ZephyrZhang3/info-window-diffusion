"""
Metrics computation for Phase 1 toy experiments.

This module implements Monte Carlo estimation of:
- J^ε(γ) = (1/d) E ||Δε(Y,C)||^2
- J^{x0}(γ) = (1/d) E ||E[X|Y,C] - E[X|Y]||^2

Note: The main experiment uses compute_J_for_gamma in run_phase1.py.
This module provides additional utilities for standalone use.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from iw_dm.toy.bayes import BayesOptimalEstimator
from iw_dm.toy.gmm import ConditionalGMM


@dataclass
class MetricsResult:
    """Container for metrics results at a single gamma value."""
    gamma: float
    J_epsilon: float
    J_epsilon_std: float
    J_x0: Optional[float] = None
    J_x0_std: Optional[float] = None
    n_samples: int = 0


def compute_J_for_gamma(
    gmm: ConditionalGMM,
    gamma: float,
    n_samples: int,
    seed: int,
    batch_size: int = 10000,
    compute_J_x0: bool = True
) -> MetricsResult:
    """
    Compute J^ε(γ) and optionally J^{x0}(γ) for a single gamma value.
    
    Monte Carlo procedure:
    1. Sample (X, C) from the GMM
    2. Sample ε ~ N(0, I)
    3. Construct Y = sqrt(γ) X + ε
    4. Compute Δε(Y,C) using Bayes-optimal estimator
    5. Average ||Δε||^2 over samples
    
    Args:
        gmm: ConditionalGMM instance
        gamma: SNR value
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        batch_size: Batch size for memory management
        compute_J_x0: Whether to also compute J^{x0}
    
    Returns:
        MetricsResult with computed values
    """
    estimator = BayesOptimalEstimator(gmm)
    d = gmm.d
    rng = np.random.default_rng(seed)
    
    sqrt_gamma = np.sqrt(max(gamma, 1e-12))
    
    J_eps_samples = []
    J_x0_samples = [] if compute_J_x0 else None
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    samples_processed = 0
    
    for _ in range(n_batches):
        current_batch_size = min(batch_size, n_samples - samples_processed)
        if current_batch_size <= 0:
            break
        
        x, c = gmm.sample(current_batch_size, seed=rng.integers(0, 2**31))
        epsilon = rng.standard_normal((current_batch_size, d))
        
        y = sqrt_gamma * x + epsilon
        
        for ci in range(gmm.n_conditions):
            mask = c == ci
            if not mask.any():
                continue
            
            y_c = y[mask]
            
            delta_eps = estimator.compute_delta_epsilon(y_c, gamma, ci)
            J_eps_samples.extend((np.sum(delta_eps ** 2, axis=1) / d).tolist())
            
            if compute_J_x0:
                delta_x0 = estimator.compute_delta_x0(y_c, gamma, ci)
                J_x0_samples.extend((np.sum(delta_x0 ** 2, axis=1) / d).tolist())
        
        samples_processed += current_batch_size
    
    J_eps_arr = np.array(J_eps_samples)
    J_x0_arr = np.array(J_x0_samples) if compute_J_x0 else None
    
    return MetricsResult(
        gamma=gamma,
        J_epsilon=float(np.mean(J_eps_arr)),
        J_epsilon_std=float(np.std(J_eps_arr) / np.sqrt(len(J_eps_arr))),
        J_x0=float(np.mean(J_x0_arr)) if J_x0_arr is not None else None,
        J_x0_std=float(np.std(J_x0_arr) / np.sqrt(len(J_x0_arr))) if J_x0_arr is not None else None,
        n_samples=len(J_eps_samples)
    )


def verify_identity(
    results: List[MetricsResult],
    gamma_grid: List[float]
) -> tuple:
    """
    Verify the identity: J^ε(γ) = γ J^{x0}(γ)
    
    Computes the relative error between J^ε(γ) and γ J^{x0}(γ)
    across all gamma values.
    
    Args:
        results: List of MetricsResult from compute_J_for_gamma
        gamma_grid: List of gamma values
    
    Returns:
        Tuple of (mean_relative_error, max_relative_error, errors_array)
    """
    errors = []
    
    for result, gamma in zip(results, gamma_grid):
        if result.J_x0 is None:
            continue
        
        J_eps = result.J_epsilon
        J_x0_scaled = gamma * result.J_x0
        
        if J_eps > 1e-12:
            rel_error = abs(J_eps - J_x0_scaled) / J_eps
            errors.append(rel_error)
    
    errors = np.array(errors)
    
    if len(errors) == 0:
        return float('nan'), float('nan'), np.array([])
    
    return float(np.mean(errors)), float(np.max(errors)), errors
