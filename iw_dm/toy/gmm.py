"""
Conditional Gaussian Mixture Model for Phase 1 toy experiments.

This module implements a 2D conditional GMM distribution with:
- Binary condition C ∈ {0, 1}
- Each condition has 2 Gaussian components
- Designed to produce clear prior shift between conditions
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GMMComponent:
    """A single Gaussian component in the mixture."""
    mean: np.ndarray
    cov: np.ndarray
    weight: float = 0.5


class ConditionalGMM:
    """
    Conditional Gaussian Mixture Model for toy experiments.
    
    Configuration (from exp plan):
    - d = 2 (dimension)
    - C ∈ {0, 1}, P(C=0) = P(C=1) = 0.5
    - Each condition has 2-component GMM with equal weights
    
    Component means:
        μ_{0,1} = (+m, 0),   μ_{0,2} = (+m, +m)
        μ_{1,1} = (-m, 0),   μ_{1,2} = (-m, -m)
    
    Component covariances:
        Σ_{c,k} = diag(s^2, s^2)
    
    Attributes:
        d: Dimension (default 2)
        m: Mean shift parameter (default 4.0)
        s: Standard deviation (default 0.5)
        n_conditions: Number of conditions (2)
        n_components_per_condition: Components per condition (2)
    """
    
    def __init__(self, d: int = 2, m: float = 4.0, s: float = 0.5):
        """
        Initialize the conditional GMM.
        
        Args:
            d: Dimension of the data (default 2)
            m: Mean shift parameter controlling separation between conditions
            s: Standard deviation for each component
        """
        self.d = d
        self.m = m
        self.s = s
        self.n_conditions = 2
        self.n_components_per_condition = 2
        
        self._components = self._build_components()
        self._condition_probs = np.array([0.5, 0.5])
    
    def _build_components(self) -> dict:
        """
        Build all GMM components.
        
        For d > 2, the means are padded with zeros in higher dimensions.
        The separation between conditions is maintained in the first two dimensions.
        
        Returns:
            Dictionary mapping (condition, component) to GMMComponent
        """
        components = {}
        
        cov = np.eye(self.d) * (self.s ** 2)
        
        # Base 2D means (from phase1.md specification)
        base_means = {
            (0, 0): np.array([self.m, 0.0]),
            (0, 1): np.array([self.m, self.m]),
            (1, 0): np.array([-self.m, 0.0]),
            (1, 1): np.array([-self.m, -self.m]),
        }
        
        # Pad means to d dimensions if d > 2
        for (c, k), base_mean in base_means.items():
            if self.d > 2:
                mean = np.zeros(self.d)
                mean[:2] = base_mean
            else:
                mean = base_mean.copy()
            
            components[(c, k)] = GMMComponent(
                mean=mean,
                cov=cov.copy(),
                weight=0.5
            )
        
        return components
    
    def get_component_params(self) -> dict:
        """
        Get all component parameters.
        
        Returns:
            Dictionary mapping (condition, component) to (mean, cov, weight)
        """
        return {
            key: (comp.mean, comp.cov, comp.weight)
            for key, comp in self._components.items()
        }
    
    def sample(self, n_samples: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (x, c) pairs from the conditional GMM.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        
        Returns:
            x: Sampled data points, shape (n_samples, d)
            c: Sampled conditions, shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        
        conditions = rng.choice(self.n_conditions, size=n_samples, p=self._condition_probs)
        
        x = np.zeros((n_samples, self.d))
        
        for c in range(self.n_conditions):
            mask = conditions == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            
            component_indices = rng.choice(
                self.n_components_per_condition,
                size=n_c,
                p=[0.5, 0.5]
            )
            
            for k in range(self.n_components_per_condition):
                k_mask = component_indices == k
                n_k = k_mask.sum()
                if n_k == 0:
                    continue
                
                comp = self._components[(c, k)]
                samples = rng.multivariate_normal(comp.mean, comp.cov, size=n_k)
                x[mask][k_mask] = samples
        
        return x, conditions
    
    def compute_conditional_mean(self, c: int) -> np.ndarray:
        """
        Compute E[X | C=c], the conditional mean given condition.
        
        Args:
            c: Condition index (0 or 1)
        
        Returns:
            Conditional mean, shape (d,)
        """
        mean = np.zeros(self.d)
        for k in range(self.n_components_per_condition):
            comp = self._components[(c, k)]
            mean += comp.weight * comp.mean
        return mean
    
    def compute_prior_mean(self) -> np.ndarray:
        """
        Compute E[X], the unconditional prior mean.
        
        Returns:
            Prior mean, shape (d,)
        """
        mean = np.zeros(self.d)
        for c in range(self.n_conditions):
            mean += self._condition_probs[c] * self.compute_conditional_mean(c)
        return mean
    
    def compute_vc(self) -> float:
        """
        Compute v_C = (1/d) E || E[X|C] - E[X] ||^2
        
        This measures the prior shift energy between conditions.
        
        Returns:
            v_C value (scalar)
        """
        prior_mean = self.compute_prior_mean()
        
        vc = 0.0
        for c in range(self.n_conditions):
            cond_mean = self.compute_conditional_mean(c)
            diff = cond_mean - prior_mean
            vc += self._condition_probs[c] * np.sum(diff ** 2)
        
        return vc / self.d
    
    def compute_conditional_cov(self, c: int) -> np.ndarray:
        """
        Compute Cov[X | C=c], the conditional covariance given condition.
        
        Uses the law of total variance:
        Cov[X|C] = E[Cov[X|C,K]|C] + Cov[E[X|C,K]|C]
        
        Args:
            c: Condition index (0 or 1)
        
        Returns:
            Conditional covariance, shape (d, d)
        """
        cond_mean = self.compute_conditional_mean(c)
        
        cov = np.zeros((self.d, self.d))
        for k in range(self.n_components_per_condition):
            comp = self._components[(c, k)]
            cov += comp.weight * comp.cov
            diff = comp.mean - cond_mean
            cov += comp.weight * np.outer(diff, diff)
        
        return cov
    
    def compute_prior_cov(self) -> np.ndarray:
        """
        Compute Cov[X], the unconditional prior covariance.
        
        Returns:
            Prior covariance, shape (d, d)
        """
        prior_mean = self.compute_prior_mean()
        
        cov = np.zeros((self.d, self.d))
        for c in range(self.n_conditions):
            cond_mean = self.compute_conditional_mean(c)
            cond_cov = self.compute_conditional_cov(c)
            
            cov += self._condition_probs[c] * cond_cov
            diff = cond_mean - prior_mean
            cov += self._condition_probs[c] * np.outer(diff, diff)
        
        return cov
    
    def __repr__(self) -> str:
        return (
            f"ConditionalGMM(d={self.d}, m={self.m}, s={self.s}, "
            f"v_C={self.compute_vc():.4f})"
        )
