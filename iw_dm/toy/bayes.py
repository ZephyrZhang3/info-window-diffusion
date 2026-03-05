"""
Bayes-optimal estimation for conditional GMM under AWGN observation model.

This module implements the core Bayesian inference for Phase 1 experiments:
- Posterior probability computation P(c,k|y)
- Conditional posterior mean E[X|y,c]
- Unconditional posterior mean E[X|y]
- Delta epsilon Δε(y,c) = E[ε|y,c] - E[ε|y]

Observation model:
    Y = sqrt(γ) X + ε, where ε ~ N(0, I_d)
"""

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from .gmm import ConditionalGMM


class BayesOptimalEstimator:
    """
    Bayes-optimal estimator for conditional GMM under AWGN.
    
    This class computes the optimal (MMSE) estimators for X and ε
    given noisy observation Y under the AWGN observation model.
    
    Observation model:
        Y = sqrt(γ) X + ε, where ε ~ N(0, I_d)
    
    For each component (c,k):
        Y | (c,k) ~ N(sqrt(γ) μ_{c,k}, I + γ Σ_{c,k})
    
    Attributes:
        gmm: The ConditionalGMM distribution
        d: Dimension
    """
    
    def __init__(self, gmm: ConditionalGMM):
        """
        Initialize the Bayes-optimal estimator.
        
        Args:
            gmm: A ConditionalGMM instance
        """
        self.gmm = gmm
        self.d = gmm.d
    
    def _compute_log_likelihood_component(
        self,
        y: np.ndarray,
        gamma: float,
        c: int,
        k: int
    ) -> float:
        """
        Compute log P(Y=y | c,k) for a single component.
        
        Y | (c,k) ~ N(sqrt(γ) μ_{c,k}, I + γ Σ_{c,k})
        
        Args:
            y: Observation, shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
            k: Component index
        
        Returns:
            Log likelihood (scalar or shape (n,))
        """
        mean_y, cov_y = self._get_observation_distribution(gamma, c, k)
        
        diff = y - mean_y
        
        try:
            L = np.linalg.cholesky(cov_y)
            solved = np.linalg.solve(L, diff.T if diff.ndim == 2 else diff.reshape(-1, 1))
            mahal = np.sum(solved ** 2, axis=0)
            log_det = 2 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.inv(cov_y)
            mahal = diff @ cov_inv @ diff.T if diff.ndim == 1 else np.sum(diff @ cov_inv * diff, axis=1)
            log_det = np.linalg.slogdet(cov_y)[1]
        
        log_lik = -0.5 * (self.d * np.log(2 * np.pi) + log_det + mahal)
        
        return float(log_lik) if np.isscalar(log_lik) or log_lik.ndim == 0 else log_lik
    
    def _get_observation_distribution(
        self,
        gamma: float,
        c: int,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distribution of Y | (c,k).
        
        Y | (c,k) ~ N(sqrt(γ) μ_{c,k}, I + γ Σ_{c,k})
        
        Args:
            gamma: SNR value
            c: Condition index
            k: Component index
        
        Returns:
            mean: Mean of Y | (c,k), shape (d,)
            cov: Covariance of Y | (c,k), shape (d, d)
        """
        comp = self.gmm._components[(c, k)]
        sqrt_gamma = np.sqrt(max(gamma, 1e-12))
        
        mean_y = sqrt_gamma * comp.mean
        cov_y = np.eye(self.d) + gamma * comp.cov
        
        return mean_y, cov_y
    
    def compute_log_posterior_probs(
        self,
        y: np.ndarray,
        gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log P(c,k | y) for all components.
        
        Uses Bayes rule with softmax for numerical stability:
        log P(c,k|y) = log P(y|c,k) + log P(c,k) - log P(y)
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
        
        Returns:
            log_probs: Log posterior probabilities, shape (2, 2) or (n, 2, 2)
            probs: Posterior probabilities, shape (2, 2) or (n, 2, 2)
        """
        single_sample = y.ndim == 1
        if single_sample:
            y = y.reshape(1, -1)
        
        n = y.shape[0]
        log_probs = np.zeros((n, self.gmm.n_conditions, self.gmm.n_components_per_condition))
        
        for c in range(self.gmm.n_conditions):
            for k in range(self.gmm.n_components_per_condition):
                log_lik = self._compute_log_likelihood_component(y, gamma, c, k)
                log_prior = np.log(self.gmm._condition_probs[c]) + np.log(0.5)
                log_probs[:, c, k] = log_lik + log_prior
        
        log_probs_flat = log_probs.reshape(n, -1)
        log_probs_flat = log_probs_flat - logsumexp(log_probs_flat, axis=1, keepdims=True)
        log_probs = log_probs_flat.reshape(n, self.gmm.n_conditions, self.gmm.n_components_per_condition)
        
        probs = np.exp(log_probs)
        
        if single_sample:
            return log_probs[0], probs[0]
        return log_probs, probs
    
    def compute_posterior_mean_x_component(
        self,
        y: np.ndarray,
        gamma: float,
        c: int,
        k: int
    ) -> np.ndarray:
        """
        Compute E[X | y, c, k] for a single component.
        
        For Gaussian observation model:
        E[X | y, c,k] = μ_{c,k} + Σ_{c,k} sqrt(γ) (I + γ Σ_{c,k})^{-1} (y - sqrt(γ) μ_{c,k})
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
            k: Component index
        
        Returns:
            Posterior mean, shape (d,) or (n, d)
        """
        single_sample = y.ndim == 1
        if single_sample:
            y = y.reshape(1, -1)
        
        comp = self.gmm._components[(c, k)]
        sqrt_gamma = np.sqrt(max(gamma, 1e-12))
        
        mean_y = sqrt_gamma * comp.mean
        cov_y = np.eye(self.d) + gamma * comp.cov
        
        diff = y - mean_y
        
        cov_y_inv = np.linalg.inv(cov_y)
        gain = comp.cov @ (sqrt_gamma * cov_y_inv)
        
        posterior_mean = comp.mean + diff @ gain.T
        
        if single_sample:
            return posterior_mean[0]
        return posterior_mean
    
    def compute_posterior_mean_x_cond(
        self,
        y: np.ndarray,
        gamma: float,
        c: int
    ) -> np.ndarray:
        """
        Compute E[X | y, c], the conditional posterior mean.
        
        E[X | y, c] = Σ_k P(k | y, c) E[X | y, c, k]
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
        
        Returns:
            Conditional posterior mean, shape (d,) or (n, d)
        """
        single_sample = y.ndim == 1
        if single_sample:
            y = y.reshape(1, -1)
        
        _, probs = self.compute_log_posterior_probs(y, gamma)
        
        n = y.shape[0]
        posterior_mean = np.zeros((n, self.d))
        
        for k in range(self.gmm.n_components_per_condition):
            mean_k = self.compute_posterior_mean_x_component(y, gamma, c, k)
            p_k_given_yc = probs[:, c, k] / (probs[:, c, :].sum(axis=1) + 1e-12)
            posterior_mean += p_k_given_yc[:, np.newaxis] * mean_k
        
        if single_sample:
            return posterior_mean[0]
        return posterior_mean
    
    def compute_posterior_mean_x_uncond(
        self,
        y: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Compute E[X | y], the unconditional posterior mean.
        
        E[X | y] = Σ_{c,k} P(c,k | y) E[X | y, c, k]
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
        
        Returns:
            Unconditional posterior mean, shape (d,) or (n, d)
        """
        single_sample = y.ndim == 1
        if single_sample:
            y = y.reshape(1, -1)
        
        _, probs = self.compute_log_posterior_probs(y, gamma)
        
        n = y.shape[0]
        posterior_mean = np.zeros((n, self.d))
        
        for c in range(self.gmm.n_conditions):
            for k in range(self.gmm.n_components_per_condition):
                mean_ck = self.compute_posterior_mean_x_component(y, gamma, c, k)
                posterior_mean += probs[:, c, k, np.newaxis] * mean_ck
        
        if single_sample:
            return posterior_mean[0]
        return posterior_mean
    
    def compute_posterior_mean_epsilon_cond(
        self,
        y: np.ndarray,
        gamma: float,
        c: int
    ) -> np.ndarray:
        """
        Compute E[ε | y, c], the conditional posterior mean of noise.
        
        Since ε = y - sqrt(γ) X:
        E[ε | y, c] = y - sqrt(γ) E[X | y, c]
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
        
        Returns:
            Conditional posterior mean of ε, shape (d,) or (n, d)
        """
        sqrt_gamma = np.sqrt(max(gamma, 1e-12))
        e_x_given_yc = self.compute_posterior_mean_x_cond(y, gamma, c)
        return y - sqrt_gamma * e_x_given_yc
    
    def compute_posterior_mean_epsilon_uncond(
        self,
        y: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Compute E[ε | y], the unconditional posterior mean of noise.
        
        Since ε = y - sqrt(γ) X:
        E[ε | y] = y - sqrt(γ) E[X | y]
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
        
        Returns:
            Unconditional posterior mean of ε, shape (d,) or (n, d)
        """
        sqrt_gamma = np.sqrt(max(gamma, 1e-12))
        e_x_given_y = self.compute_posterior_mean_x_uncond(y, gamma)
        return y - sqrt_gamma * e_x_given_y
    
    def compute_delta_epsilon(
        self,
        y: np.ndarray,
        gamma: float,
        c: int
    ) -> np.ndarray:
        """
        Compute Δε(y,c) = E[ε | y, c] - E[ε | y].
        
        This is the key quantity for computing J^ε(γ).
        
        Simplified form:
        Δε(y,c) = -sqrt(γ) (E[X | y, c] - E[X | y])
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
        
        Returns:
            Delta epsilon, shape (d,) or (n, d)
        """
        sqrt_gamma = np.sqrt(max(gamma, 1e-12))
        e_x_cond = self.compute_posterior_mean_x_cond(y, gamma, c)
        e_x_uncond = self.compute_posterior_mean_x_uncond(y, gamma)
        return -sqrt_gamma * (e_x_cond - e_x_uncond)
    
    def compute_delta_x0(
        self,
        y: np.ndarray,
        gamma: float,
        c: int
    ) -> np.ndarray:
        """
        Compute E[X | y, c] - E[X | y].
        
        This is the key quantity for computing J^{x0}(γ).
        
        Args:
            y: Observation(s), shape (d,) or (n, d)
            gamma: SNR value
            c: Condition index
        
        Returns:
            Delta x0, shape (d,) or (n, d)
        """
        e_x_cond = self.compute_posterior_mean_x_cond(y, gamma, c)
        e_x_uncond = self.compute_posterior_mean_x_uncond(y, gamma)
        return e_x_cond - e_x_uncond
