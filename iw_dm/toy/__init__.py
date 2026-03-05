"""
Toy experiments module for Phase 1 (Oracle Toy validation).

This module implements:
- Conditional GMM data generation
- Bayes-optimal estimation
- J^ε and J^{x0} metrics computation
"""

from .gmm import ConditionalGMM
from .bayes import BayesOptimalEstimator
from .metrics import compute_J_for_gamma, verify_identity, MetricsResult

__all__ = [
    "ConditionalGMM",
    "BayesOptimalEstimator",
    "compute_J_for_gamma",
    "verify_identity",
    "MetricsResult",
]
