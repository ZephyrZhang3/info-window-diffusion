"""
Utility functions for experiments.

This module provides:
- SNR/γ computation utilities
- Statistical analysis tools
- Visualization functions
"""

from .snr import (
    get_gamma_grid,
    compute_gamma_from_alpha_bar,
    compute_alpha_sigma_from_gamma,
    compute_gamma_from_alpha_sigma,
)
from .statistics import (
    compute_plateau_cv,
    fit_linear_through_origin,
    compute_identity_error,
    PlateauMetrics,
)
from .visualization import (
    plot_J_vs_gamma,
    plot_R_vs_gamma,
    plot_identity_check,
    plot_plateau_zoom,
    generate_all_figures,
)

__all__ = [
    "get_gamma_grid",
    "compute_gamma_from_alpha_bar",
    "compute_alpha_sigma_from_gamma",
    "compute_gamma_from_alpha_sigma",
    "compute_plateau_cv",
    "fit_linear_through_origin",
    "compute_identity_error",
    "PlateauMetrics",
    "plot_J_vs_gamma",
    "plot_R_vs_gamma",
    "plot_identity_check",
    "plot_plateau_zoom",
    "generate_all_figures",
]
