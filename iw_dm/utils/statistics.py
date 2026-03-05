"""
Statistical analysis utilities for Phase 1 plateau detection and validation.

This module provides functions for:
- Computing plateau stability (CV)
- Linear fitting through origin
- Plateau test metrics
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PlateauMetrics:
    """Container for plateau test metrics."""
    cv: float
    mean_j_over_gamma: float
    std_j_over_gamma: float
    slope: float
    r_squared: float
    n_points: int
    gamma_range: Tuple[float, float]


def compute_plateau_cv(
    j_values: np.ndarray,
    gamma_values: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute plateau stability using coefficient of variation of J/γ.
    
    CV = std(J/γ) / mean(J/γ)
    
    Lower CV indicates a more stable plateau.
    
    Args:
        j_values: Array of J values
        gamma_values: Array of corresponding gamma values
    
    Returns:
        Tuple of (cv, mean_j_over_gamma, std_j_over_gamma)
    """
    gamma_values = np.maximum(gamma_values, 1e-12)
    j_over_gamma = j_values / gamma_values
    
    mean_val = np.mean(j_over_gamma)
    std_val = np.std(j_over_gamma)
    
    if mean_val > 1e-12:
        cv = std_val / mean_val
    else:
        cv = float('inf')
    
    return cv, float(mean_val), float(std_val)


def fit_linear_through_origin(
    y: np.ndarray,
    x: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Fit a linear model y = a*x through the origin.
    
    Uses least squares: a = (x'y) / (x'x)
    
    R² is computed as:
    R² = 1 - SS_res / SS_tot
    where SS_res = Σ(y - a*x)²
          SS_tot = Σ(y - mean(y))²
    
    Note: For through-origin regression, R² can be negative
    if the fit is worse than just using mean(y).
    
    Args:
        y: Dependent variable
        x: Independent variable
    
    Returns:
        Tuple of (slope, r_squared, residuals)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    x_sum_sq = np.sum(x ** 2)
    if x_sum_sq < 1e-12:
        return 0.0, 0.0, y
    
    slope = np.sum(x * y) / x_sum_sq
    
    y_pred = slope * x
    residuals = y - y_pred
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    if ss_tot > 1e-12:
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = 0.0
    
    return float(slope), float(r_squared), residuals


def compute_identity_error(
    j_epsilon: float,
    j_x0: float,
    gamma: float
) -> float:
    """
    Compute relative error for identity verification: J^ε(γ) = γ·J^{x0}(γ)
    
    err_id = |J^ε(γ) - γ·J^{x0}(γ)| / max(J^ε(γ), 1e-12)
    
    Args:
        j_epsilon: J^ε value
        j_x0: J^{x0} value
        gamma: SNR value
    
    Returns:
        Relative error
    """
    j_x0_scaled = gamma * j_x0
    
    if j_epsilon > 1e-12:
        return abs(j_epsilon - j_x0_scaled) / j_epsilon
    else:
        return float('inf')
