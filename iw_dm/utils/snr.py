"""
SNR (Signal-to-Noise Ratio) utilities for Phase 1 experiments.

This module provides utility functions for computing SNR-related quantities.
For Phase 1, we use fixed gamma grids directly, not diffusion schedules.
"""

from typing import List, Tuple, Union

import numpy as np


def get_gamma_grid() -> Tuple[List[float], List[float]]:
    """
    Return the standard gamma grids for Phase 1 experiments.
    
    From phase1.md:
    - Global grid: for observing the full shape of J(γ)
    - Asymptotic grid: for plateau test (small γ region)
    
    Returns:
        Tuple of (global_grid, asymptotic_grid)
    """
    global_grid = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    asymptotic_grid = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    return global_grid, asymptotic_grid


def compute_gamma_from_alpha_bar(
    alpha_bar: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute SNR γ from cumulative product alpha_bar.
    
    γ = ᾱ / (1 - ᾱ)
    
    Args:
        alpha_bar: Cumulative product of alpha values
    
    Returns:
        SNR value(s)
    """
    alpha_bar = np.clip(alpha_bar, 1e-12, 1 - 1e-12)
    return alpha_bar / (1 - alpha_bar)


def compute_alpha_sigma_from_gamma(
    gamma: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute alpha and sigma from SNR γ.
    
    Given γ = α² / σ² and α² + σ² = 1 (normalized):
    α = sqrt(γ / (1 + γ))
    σ = sqrt(1 / (1 + γ))
    
    Args:
        gamma: SNR value(s)
    
    Returns:
        Tuple of (alpha, sigma)
    """
    gamma = np.maximum(gamma, 1e-12)
    alpha = np.sqrt(gamma / (1 + gamma))
    sigma = np.sqrt(1 / (1 + gamma))
    return alpha, sigma


def compute_gamma_from_alpha_sigma(
    alpha: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute SNR γ from alpha and sigma.
    
    γ = α² / σ²
    
    Args:
        alpha: Signal scaling factor(s)
        sigma: Noise scaling factor(s)
    
    Returns:
        SNR value(s)
    """
    return alpha ** 2 / (sigma ** 2 + 1e-12)
