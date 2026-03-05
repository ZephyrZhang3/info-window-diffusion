"""
Visualization utilities for Phase 1 experiments.

This module provides functions for generating the required figures:
- fig_J_vs_gamma.png: J^ε(γ) vs γ (log-log)
- fig_R_vs_gamma.png: R(γ) = J^ε(γ)/γ vs γ
- fig_identity_check.png: J^ε(γ) vs γ·J^{x0}(γ)
- fig_plateau_zoom.png: R(γ) for small γ region
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def _ensure_matplotlib():
    """Lazy import of matplotlib to avoid dependency issues."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def _setup_publication_style():
    """Setup matplotlib style for publication quality figures."""
    plt = _ensure_matplotlib()
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })


def plot_J_vs_gamma(
    gamma_values: np.ndarray,
    j_values: np.ndarray,
    j_std: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    show_plot: bool = False
):
    """
    Plot J^ε(γ) vs γ on log-log scale.
    
    This figure shows the decay of information window as SNR decreases.
    
    Args:
        gamma_values: Array of gamma values (SNR)
        j_values: Array of J^ε values
        j_std: Optional array of standard deviations
        save_path: Path to save the figure
        title: Plot title (default: auto-generated)
        figsize: Figure size in inches
        show_plot: Whether to display the plot
    """
    plt = _ensure_matplotlib()
    _setup_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gamma_values = np.asarray(gamma_values)
    j_values = np.asarray(j_values)
    
    sort_idx = np.argsort(gamma_values)
    gamma_sorted = gamma_values[sort_idx]
    j_sorted = j_values[sort_idx]
    
    ax.loglog(gamma_sorted, j_sorted, 'o-', color='#1f77b4', alpha=0.9,
              markersize=6, linewidth=1.5, markerfacecolor='white',
              markeredgewidth=1.5, markeredgecolor='#1f77b4',
              label=r'$J^\epsilon(\gamma)$')
    
    if j_std is not None:
        j_std_sorted = j_std[sort_idx]
        ax.fill_between(
            gamma_sorted,
            j_sorted - j_std_sorted,
            j_sorted + j_std_sorted,
            color='#1f77b4',
            alpha=0.15,
            linewidth=0
        )
    
    ax.set_xlabel(r'$\gamma$ (SNR)', fontsize=12)
    ax.set_ylabel(r'$J^\epsilon(\gamma)$', fontsize=12)
    
    if title is None:
        title = r'$J^\epsilon(\gamma)$ vs $\gamma$ (log-log)'
    ax.set_title(title, fontsize=13)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)


def plot_R_vs_gamma(
    gamma_values: np.ndarray,
    j_values: np.ndarray,
    v_c: float,
    j_std: Optional[np.ndarray] = None,
    asymptotic_grid: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    show_plot: bool = False
):
    """
    Plot R(γ) = J^ε(γ)/γ vs γ, showing convergence to plateau v_C.
    
    This is the key diagnostic figure for the plateau phenomenon.
    
    Args:
        gamma_values: Array of gamma values
        j_values: Array of J^ε values
        v_c: Theoretical plateau height
        j_std: Optional array of standard deviations
        asymptotic_grid: List of gamma values in asymptotic grid (highlighted)
        save_path: Path to save the figure
        title: Plot title
        figsize: Figure size in inches
        show_plot: Whether to display the plot
    """
    plt = _ensure_matplotlib()
    _setup_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gamma_values = np.asarray(gamma_values)
    j_values = np.asarray(j_values)
    
    gamma_safe = np.maximum(gamma_values, 1e-16)
    j_over_gamma = j_values / gamma_safe
    log_gamma = np.log10(gamma_safe)
    
    sort_idx = np.argsort(log_gamma)
    log_gamma_sorted = log_gamma[sort_idx]
    jog_sorted = j_over_gamma[sort_idx]
    gamma_sorted = gamma_safe[sort_idx]
    
    ax.plot(log_gamma_sorted, jog_sorted, 'o-', color='#2ca02c', alpha=0.9,
            markersize=6, linewidth=1.5, markerfacecolor='white',
            markeredgewidth=1.5, markeredgecolor='#2ca02c',
            label=r'$J^\epsilon(\gamma)/\gamma$')
    
    if j_std is not None:
        jog_std = j_std[sort_idx] / gamma_sorted
        ax.fill_between(
            log_gamma_sorted,
            jog_sorted - jog_std,
            jog_sorted + jog_std,
            color='#2ca02c',
            alpha=0.15,
            linewidth=0
        )
    
    ax.axhline(y=v_c, color='#d62728', linestyle='--', linewidth=2,
               alpha=0.9, label=f'$v_C = {v_c:.1f}$')
    
    if asymptotic_grid is not None:
        asymptotic_set = set(asymptotic_grid)
        asymptotic_mask = np.array([g in asymptotic_set for g in gamma_values])
        if np.any(asymptotic_mask):
            ax.scatter(
                log_gamma[asymptotic_mask],
                j_over_gamma[asymptotic_mask],
                color='red',
                s=80,
                zorder=5,
                marker='*',
                label='Asymptotic grid'
            )
    
    ax.set_xlabel(r'$\log_{10}\gamma$', fontsize=12)
    ax.set_ylabel(r'$J^\epsilon(\gamma)/\gamma$', fontsize=12)
    
    if title is None:
        title = r'Plateau Test: $J^\epsilon(\gamma)/\gamma$ vs $\gamma$'
    ax.set_title(title, fontsize=13)
    
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)


def plot_identity_check(
    gamma_values: np.ndarray,
    j_epsilon: np.ndarray,
    j_x0: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    show_plot: bool = False
):
    """
    Plot identity verification: J^ε(γ) vs γ·J^{x0}(γ).
    
    This figure verifies the theoretical identity:
    J^ε(γ) = γ·J^{x0}(γ)
    
    Args:
        gamma_values: Array of gamma values
        j_epsilon: Array of J^ε values
        j_x0: Array of J^{x0} values
        save_path: Path to save the figure
        title: Plot title
        figsize: Figure size in inches
        show_plot: Whether to display the plot
    """
    plt = _ensure_matplotlib()
    _setup_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gamma_values = np.asarray(gamma_values)
    j_epsilon = np.asarray(j_epsilon)
    j_x0 = np.asarray(j_x0)
    
    j_x0_scaled = gamma_values * j_x0
    
    sort_idx = np.argsort(gamma_values)
    gamma_sorted = gamma_values[sort_idx]
    j_eps_sorted = j_epsilon[sort_idx]
    j_x0_scaled_sorted = j_x0_scaled[sort_idx]
    
    ax.loglog(gamma_sorted, j_eps_sorted, 'o-', color='#1f77b4', alpha=0.9,
              markersize=6, linewidth=1.5, markerfacecolor='white',
              markeredgewidth=1.5, markeredgecolor='#1f77b4',
              label=r'$J^\epsilon(\gamma)$')
    
    ax.loglog(gamma_sorted, j_x0_scaled_sorted, 's--', color='#d62728', alpha=0.9,
              markersize=6, linewidth=1.5, markerfacecolor='white',
              markeredgewidth=1.5, markeredgecolor='#d62728',
              label=r'$\gamma \cdot J^{x_0}(\gamma)$')
    
    ax.set_xlabel(r'$\gamma$ (SNR)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    if title is None:
        title = r'Identity Check: $J^\epsilon(\gamma)$ vs $\gamma \cdot J^{x_0}(\gamma)$'
    ax.set_title(title, fontsize=13)
    
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)


def plot_plateau_zoom(
    gamma_values: np.ndarray,
    j_values: np.ndarray,
    v_c: float,
    j_std: Optional[np.ndarray] = None,
    asymptotic_grid: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    show_plot: bool = False
):
    """
    Plot plateau detail: R(γ) for small γ region only.
    
    This figure zooms into the asymptotic region to show the plateau clearly.
    
    Args:
        gamma_values: Array of gamma values
        j_values: Array of J^ε values
        v_c: Theoretical plateau height
        j_std: Optional array of standard deviations
        asymptotic_grid: List of gamma values in asymptotic grid
        save_path: Path to save the figure
        title: Plot title
        figsize: Figure size in inches
        show_plot: Whether to display the plot
    """
    plt = _ensure_matplotlib()
    _setup_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gamma_values = np.asarray(gamma_values)
    j_values = np.asarray(j_values)
    
    if asymptotic_grid is not None:
        asymptotic_set = set(asymptotic_grid)
        mask = np.array([g in asymptotic_set for g in gamma_values])
        gamma_plot = gamma_values[mask]
        j_plot = j_values[mask]
        j_std_plot = j_std[mask] if j_std is not None else None
    else:
        gamma_max = 1e-3
        mask = gamma_values <= gamma_max
        gamma_plot = gamma_values[mask]
        j_plot = j_values[mask]
        j_std_plot = j_std[mask] if j_std is not None else None
    
    gamma_safe = np.maximum(gamma_plot, 1e-16)
    j_over_gamma = j_plot / gamma_safe
    log_gamma = np.log10(gamma_safe)
    
    sort_idx = np.argsort(log_gamma)
    log_gamma_sorted = log_gamma[sort_idx]
    jog_sorted = j_over_gamma[sort_idx]
    gamma_sorted = gamma_safe[sort_idx]
    
    ax.plot(log_gamma_sorted, jog_sorted, 'o-', color='#2ca02c', alpha=0.9,
            markersize=8, linewidth=2, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#2ca02c',
            label=r'$J^\epsilon(\gamma)/\gamma$')
    
    if j_std_plot is not None:
        jog_std = j_std_plot[sort_idx] / gamma_sorted
        ax.fill_between(
            log_gamma_sorted,
            jog_sorted - jog_std,
            jog_sorted + jog_std,
            color='#2ca02c',
            alpha=0.2,
            linewidth=0
        )
    
    ax.axhline(y=v_c, color='#d62728', linestyle='--', linewidth=2.5,
               alpha=0.9, label=f'$v_C = {v_c:.1f}$')
    
    ax.axhspan(v_c * 0.85, v_c * 1.15, alpha=0.1, color='#d62728',
               label=r'$\pm 15\%$ tolerance')
    
    ax.set_xlabel(r'$\log_{10}\gamma$', fontsize=12)
    ax.set_ylabel(r'$J^\epsilon(\gamma)/\gamma$', fontsize=12)
    
    if title is None:
        title = 'Plateau Detail (Small γ Region)'
    ax.set_title(title, fontsize=13)
    
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)


def generate_all_figures(
    gamma_values: np.ndarray,
    j_values: np.ndarray,
    j_x0_values: np.ndarray,
    v_c: float,
    output_dir: Path,
    j_std: Optional[np.ndarray] = None,
    asymptotic_grid: Optional[List[float]] = None
):
    """
    Generate all required figures for Phase 1.
    
    Args:
        gamma_values: Array of gamma values
        j_values: Array of J^ε values
        j_x0_values: Array of J^{x0} values
        v_c: Theoretical plateau height
        output_dir: Output directory
        j_std: Optional standard deviations
        asymptotic_grid: List of gamma values in asymptotic grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_J_vs_gamma(
        gamma_values, j_values, j_std,
        save_path=output_dir / "fig_J_vs_gamma.png"
    )
    
    plot_R_vs_gamma(
        gamma_values, j_values, v_c, j_std, asymptotic_grid,
        save_path=output_dir / "fig_R_vs_gamma.png"
    )
    
    plot_identity_check(
        gamma_values, j_values, j_x0_values,
        save_path=output_dir / "fig_identity_check.png"
    )
    
    plot_plateau_zoom(
        gamma_values, j_values, v_c, j_std, asymptotic_grid,
        save_path=output_dir / "fig_plateau_zoom.png"
    )
    
    return output_dir
