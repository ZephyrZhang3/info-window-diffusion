"""
Phase 1 Experiment: Oracle Toy Validation

This script runs the complete Phase 1 experiment according to phase1.md:
1. Creates conditional GMM distribution
2. Computes J^ε(γ) and J^{x0}(γ) for fixed gamma grids
3. Performs plateau test on asymptotic grid
4. Verifies identity relationship
5. Generates required output files

Usage:
    python -m iw_dm.experiments.run_phase1
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress
from tqdm import tqdm

from iw_dm.toy.gmm import ConditionalGMM
from iw_dm.toy.bayes import BayesOptimalEstimator
from iw_dm.utils.statistics import (
    PlateauMetrics,
    compute_plateau_cv,
    fit_linear_through_origin,
)
from iw_dm.utils.visualization import (
    plot_J_vs_gamma,
    plot_R_vs_gamma,
    plot_identity_check,
    plot_plateau_zoom,
)


@dataclass
class MetricsResult:
    """Container for metrics results at a single gamma value."""
    gamma: float
    J_epsilon: float
    J_epsilon_std: float
    J_x0: Optional[float] = None
    J_x0_std: Optional[float] = None
    n_samples: int = 0


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load experiment configuration.
    
    Args:
        config_path: Path to config file (default: configs/phase1.yaml)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "phase1.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return {
            'gmm': {'d': 2, 'm': 4.0, 's': 0.5},
            'gamma_grid': {
                'global': [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
                'asymptotic': [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
            },
            'monte_carlo': {
                'n_samples_global': 200000,
                'n_samples_asymptotic': 1000000,
                'seeds': [0, 1, 2],
                'batch_size': 10000,
            },
            'plateau_test': {
                'cv_threshold': 0.15,
                'r2_threshold': 0.98,
                'vc_relative_tol': 0.20,
            },
            'identity_check': {
                'median_err_threshold': 1e-3,
            },
            'output': {
                'base_dir': 'phase1_oracle',
            }
        }


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


def run_phase1_experiment(
    config: dict,
    console: Optional[Console] = None
) -> Dict:
    """
    Run the complete Phase 1 experiment.
    
    Args:
        config: Experiment configuration
        console: Rich console for output
    
    Returns:
        Dictionary with all results
    """
    if console is None:
        console = Console()
    
    gmm_config = config['gmm']
    gmm = ConditionalGMM(
        d=gmm_config['d'],
        m=gmm_config['m'],
        s=gmm_config['s']
    )
    
    v_c = gmm.compute_vc()
    
    console.print(f"\n[bold blue]GMM Distribution:[/bold blue]")
    console.print(f"  Dimension: {gmm.d}")
    console.print(f"  Mean shift (m): {gmm.m}")
    console.print(f"  Std dev (s): {gmm.s}")
    console.print(f"  v_C (theoretical plateau height): {v_c:.6f}")
    
    gamma_grid_config = config['gamma_grid']
    global_grid = [float(g) for g in gamma_grid_config['global']]
    asymptotic_grid = [float(g) for g in gamma_grid_config['asymptotic']]
    
    all_gamma = global_grid + asymptotic_grid
    all_gamma_sorted = sorted(all_gamma, reverse=True)
    
    console.print(f"\n[bold blue]Gamma Grid:[/bold blue]")
    console.print(f"  Global grid ({len(global_grid)} points): {global_grid}")
    console.print(f"  Asymptotic grid ({len(asymptotic_grid)} points): {asymptotic_grid}")
    console.print(f"  Total: {len(all_gamma)} unique gamma values")
    
    mc_config = config['monte_carlo']
    n_samples_global = mc_config['n_samples_global']
    n_samples_asymptotic = mc_config['n_samples_asymptotic']
    seeds = mc_config['seeds']
    batch_size = mc_config.get('batch_size', 10000)
    
    console.print(f"\n[bold blue]Monte Carlo Parameters:[/bold blue]")
    console.print(f"  Samples per γ (global): {n_samples_global:,}")
    console.print(f"  Samples per γ (asymptotic): {n_samples_asymptotic:,}")
    console.print(f"  Seeds: {seeds}")
    console.print(f"  Batch size: {batch_size}")
    
    all_results: Dict[int, List[MetricsResult]] = {seed: [] for seed in seeds}
    
    with Progress() as progress:
        task = progress.add_task(
            "[green]Running Monte Carlo...",
            total=len(seeds) * len(all_gamma_sorted)
        )
        
        for seed in seeds:
            console.print(f"\n[yellow]Running with seed {seed}...[/yellow]")
            
            for gamma in all_gamma_sorted:
                if gamma in asymptotic_grid:
                    n_samples = n_samples_asymptotic
                else:
                    n_samples = n_samples_global
                
                result = compute_J_for_gamma(
                    gmm=gmm,
                    gamma=gamma,
                    n_samples=n_samples,
                    seed=seed,
                    batch_size=batch_size,
                    compute_J_x0=True
                )
                all_results[seed].append(result)
                progress.update(task, advance=1)
    
    aggregated = aggregate_results(all_results, all_gamma_sorted)
    
    identity_errors = compute_identity_errors(all_results, all_gamma_sorted)
    median_err = float(np.median(identity_errors))
    max_err = float(np.max(identity_errors))
    
    console.print(f"\n[bold blue]Identity Verification:[/bold blue]")
    console.print(f"  J^ε(γ) = γ·J^{{x0}}(γ)")
    console.print(f"  Median relative error: {median_err:.6e}")
    console.print(f"  Max relative error: {max_err:.6e}")
    
    identity_config = config.get('identity_check', {})
    err_threshold = float(identity_config.get('median_err_threshold', 1e-3))
    identity_pass = median_err <= err_threshold
    console.print(f"  Threshold: {err_threshold:.0e}")
    console.print(f"  Status: {'[green]PASS[/green]' if identity_pass else '[red]FAIL[/red]'}")
    
    plateau_metrics = plateau_test(
        aggregated['J_epsilon_mean'],
        np.array(all_gamma_sorted),
        asymptotic_grid,
        v_c
    )
    
    console.print(f"\n[bold blue]Plateau Test (asymptotic grid):[/bold blue]")
    console.print(f"  CV(J/γ): {plateau_metrics.cv:.4f}")
    console.print(f"  Mean(J/γ): {plateau_metrics.mean_j_over_gamma:.6f}")
    console.print(f"  Std(J/γ): {plateau_metrics.std_j_over_gamma:.6f}")
    console.print(f"  Slope (J ≈ aγ): {plateau_metrics.slope:.6f}")
    console.print(f"  R²: {plateau_metrics.r_squared:.4f}")
    console.print(f"  v_C: {v_c:.6f}")
    console.print(f"  |slope - v_C| / v_C: {abs(plateau_metrics.slope - v_c) / v_c:.4f}")
    
    plateau_config = config['plateau_test']
    cv_threshold = float(plateau_config['cv_threshold'])
    r2_threshold = float(plateau_config['r2_threshold'])
    vc_tol = float(plateau_config['vc_relative_tol'])
    
    cv_pass = plateau_metrics.cv <= cv_threshold
    r2_pass = plateau_metrics.r_squared >= r2_threshold
    vc_pass = abs(plateau_metrics.slope - v_c) / v_c <= vc_tol
    
    console.print(f"\n[bold blue]Acceptance Criteria:[/bold blue]")
    console.print(f"  CV ≤ {cv_threshold}: {'[green]PASS[/green]' if cv_pass else '[red]FAIL[/red]'}")
    console.print(f"  R² ≥ {r2_threshold}: {'[green]PASS[/green]' if r2_pass else '[red]FAIL[/red]'}")
    console.print(f"  |a - v_C|/v_C ≤ {vc_tol}: {'[green]PASS[/green]' if vc_pass else '[red]FAIL[/red]'}")
    
    overall_pass = identity_pass and cv_pass and r2_pass and vc_pass
    console.print(f"\n  Overall: {'[green bold]PASSED[/green bold]' if overall_pass else '[red bold]FAILED[/red bold]'}")
    
    results_dict = {
        'config': config,
        'gmm': {
            'd': gmm.d,
            'm': gmm.m,
            's': gmm.s,
            'v_c': v_c,
        },
        'gamma_grid': {
            'global': global_grid,
            'asymptotic': asymptotic_grid,
            'all': all_gamma_sorted,
        },
        'seeds': seeds,
        'results_by_seed': {
            seed: [
                {
                    'gamma': r.gamma,
                    'J_epsilon': r.J_epsilon,
                    'J_epsilon_std': r.J_epsilon_std,
                    'J_x0': r.J_x0,
                    'J_x0_std': r.J_x0_std,
                    'n_samples': r.n_samples,
                }
                for r in results
            ]
            for seed, results in all_results.items()
        },
        'aggregated': aggregated,
        'identity_verification': {
            'median_error': median_err,
            'max_error': max_err,
            'errors': identity_errors.tolist(),
            'threshold': err_threshold,
            'passed': identity_pass,
        },
        'plateau_metrics': {
            'cv': plateau_metrics.cv,
            'mean_j_over_gamma': plateau_metrics.mean_j_over_gamma,
            'std_j_over_gamma': plateau_metrics.std_j_over_gamma,
            'slope': plateau_metrics.slope,
            'r_squared': plateau_metrics.r_squared,
            'n_points': plateau_metrics.n_points,
            'gamma_range': plateau_metrics.gamma_range,
        },
        'acceptance': {
            'identity_pass': identity_pass,
            'cv_pass': cv_pass,
            'r2_pass': r2_pass,
            'vc_pass': vc_pass,
            'overall_pass': overall_pass,
        }
    }
    
    return results_dict


def aggregate_results(
    all_results: Dict[int, List[MetricsResult]],
    gamma_grid: List[float]
) -> Dict:
    """
    Aggregate results across multiple seeds.
    
    Args:
        all_results: Dictionary mapping seed to list of MetricsResult
        gamma_grid: List of gamma values
    
    Returns:
        Dictionary with aggregated statistics
    """
    n_gamma = len(gamma_grid)
    seeds = list(all_results.keys())
    n_seeds = len(seeds)
    
    J_eps_all = np.zeros((n_seeds, n_gamma))
    J_x0_all = np.zeros((n_seeds, n_gamma))
    
    for seed_idx, (seed, results) in enumerate(all_results.items()):
        for gamma_idx, result in enumerate(results):
            J_eps_all[seed_idx, gamma_idx] = result.J_epsilon
            J_x0_all[seed_idx, gamma_idx] = result.J_x0 if result.J_x0 is not None else np.nan
    
    J_eps_mean = np.mean(J_eps_all, axis=0)
    J_eps_std = np.std(J_eps_all, axis=0)
    J_x0_mean = np.nanmean(J_x0_all, axis=0)
    J_x0_std = np.nanstd(J_x0_all, axis=0)
    
    return {
        'gamma_grid': gamma_grid,
        'J_epsilon_mean': J_eps_mean.tolist(),
        'J_epsilon_std': J_eps_std.tolist(),
        'J_x0_mean': J_x0_mean.tolist(),
        'J_x0_std': J_x0_std.tolist(),
        'J_epsilon_by_seed': J_eps_all.tolist(),
        'J_x0_by_seed': J_x0_all.tolist(),
    }


def compute_identity_errors(
    all_results: Dict[int, List[MetricsResult]],
    gamma_grid: List[float]
) -> np.ndarray:
    """
    Compute relative errors for identity verification: J^ε(γ) = γ·J^{x0}(γ)
    
    Args:
        all_results: Dictionary mapping seed to list of MetricsResult
        gamma_grid: List of gamma values
    
    Returns:
        Array of relative errors
    """
    errors = []
    
    for seed, results in all_results.items():
        for result, gamma in zip(results, gamma_grid):
            if result.J_x0 is None:
                continue
            
            J_eps = result.J_epsilon
            J_x0_scaled = gamma * result.J_x0
            
            if J_eps > 1e-12:
                rel_error = abs(J_eps - J_x0_scaled) / J_eps
                errors.append(rel_error)
    
    return np.array(errors)


def plateau_test(
    j_values: np.ndarray,
    gamma_values: np.ndarray,
    asymptotic_grid: List[float],
    v_c: float
) -> PlateauMetrics:
    """
    Perform plateau test on asymptotic grid.
    
    Args:
        j_values: Array of J values (aggregated mean)
        gamma_values: Array of gamma values
        asymptotic_grid: List of gamma values in asymptotic grid
        v_c: Theoretical plateau height
    
    Returns:
        PlateauMetrics object
    """
    gamma_values = np.asarray(gamma_values)
    j_values = np.asarray(j_values)
    
    asymptotic_mask = np.isin(gamma_values, asymptotic_grid)
    
    j_asymptotic = j_values[asymptotic_mask]
    gamma_asymptotic = gamma_values[asymptotic_mask]
    
    cv, mean_jog, std_jog = compute_plateau_cv(j_asymptotic, gamma_asymptotic)
    slope, r_squared, _ = fit_linear_through_origin(j_asymptotic, gamma_asymptotic)
    
    return PlateauMetrics(
        cv=cv,
        mean_j_over_gamma=mean_jog,
        std_j_over_gamma=std_jog,
        slope=slope,
        r_squared=r_squared,
        n_points=len(j_asymptotic),
        gamma_range=(float(gamma_asymptotic.min()), float(gamma_asymptotic.max()))
    )


def save_results(
    results: Dict,
    config: dict,
    console: Optional[Console] = None
):
    """
    Save experiment results to files.
    
    Args:
        results: Results dictionary from run_phase1_experiment
        config: Configuration dictionary
        console: Rich console for output
    """
    if console is None:
        console = Console()
    
    output_config = config['output']
    base_dir = Path(output_config['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold blue]Saving Results to {base_dir}:[/bold blue]")
    
    gamma_arr = np.array(results['gamma_grid']['all'])
    J_eps_mean = np.array(results['aggregated']['J_epsilon_mean'])
    J_eps_std = np.array(results['aggregated']['J_epsilon_std'])
    J_x0_mean = np.array(results['aggregated']['J_x0_mean'])
    J_x0_std = np.array(results['aggregated']['J_x0_std'])
    v_c = results['gmm']['v_c']
    
    readme_path = base_dir / "README_phase1.md"
    write_readme(readme_path, results, config)
    console.print(f"  Saved: {readme_path}")
    
    results_path = base_dir / "results.csv"
    write_results_csv(results_path, results)
    console.print(f"  Saved: {results_path}")
    
    fig1_path = base_dir / "fig_J_vs_gamma.png"
    plot_J_vs_gamma(
        gamma_values=gamma_arr,
        j_values=J_eps_mean,
        j_std=J_eps_std,
        save_path=fig1_path
    )
    console.print(f"  Saved: {fig1_path}")
    
    fig2_path = base_dir / "fig_R_vs_gamma.png"
    plot_R_vs_gamma(
        gamma_values=gamma_arr,
        j_values=J_eps_mean,
        j_std=J_eps_std,
        v_c=v_c,
        asymptotic_grid=results['gamma_grid']['asymptotic'],
        save_path=fig2_path
    )
    console.print(f"  Saved: {fig2_path}")
    
    fig3_path = base_dir / "fig_identity_check.png"
    plot_identity_check(
        gamma_values=gamma_arr,
        j_epsilon=J_eps_mean,
        j_x0=J_x0_mean,
        save_path=fig3_path
    )
    console.print(f"  Saved: {fig3_path}")
    
    fig4_path = base_dir / "fig_plateau_zoom.png"
    plot_plateau_zoom(
        gamma_values=gamma_arr,
        j_values=J_eps_mean,
        j_std=J_eps_std,
        v_c=v_c,
        asymptotic_grid=results['gamma_grid']['asymptotic'],
        save_path=fig4_path
    )
    console.print(f"  Saved: {fig4_path}")
    
    fit_report_path = base_dir / "fit_report.md"
    write_fit_report(fit_report_path, results)
    console.print(f"  Saved: {fit_report_path}")


def write_readme(path: Path, results: Dict, config: dict):
    """Write README_phase1.md file."""
    v_c = results['gmm']['v_c']
    pm = results['plateau_metrics']
    acc = results['acceptance']
    iv = results['identity_verification']
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Oracle Toy Validation Results\n\n")
        f.write("## Overview\n\n")
        f.write("This experiment validates the plateau law in a Bayes-optimal toy setting:\n")
        f.write("- **Goal**: Verify that J^ε(γ)/γ → v_C as γ → 0\n")
        f.write("- **Method**: Monte Carlo estimation with conditional GMM\n\n")
        
        f.write("## Configuration\n\n")
        f.write("### GMM Parameters\n")
        f.write(f"- Dimension (d): {results['gmm']['d']}\n")
        f.write(f"- Mean shift (m): {results['gmm']['m']}\n")
        f.write(f"- Std dev (s): {results['gmm']['s']}\n")
        f.write(f"- **v_C (theoretical plateau)**: {v_c:.6f}\n\n")
        
        f.write("### Gamma Grid\n")
        f.write(f"- Global grid: {results['gamma_grid']['global']}\n")
        f.write(f"- Asymptotic grid: {results['gamma_grid']['asymptotic']}\n\n")
        
        f.write("### Monte Carlo\n")
        mc = config['monte_carlo']
        f.write(f"- Seeds: {mc['seeds']}\n")
        f.write(f"- Samples (global): {mc['n_samples_global']:,}\n")
        f.write(f"- Samples (asymptotic): {mc['n_samples_asymptotic']:,}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("### Identity Verification\n")
        f.write("  J^ε(γ) = γ·J^{x0}(γ)\n")
        f.write(f"- Median relative error: {iv['median_error']:.6e}\n")
        f.write(f"- Status: {'✓ PASS' if iv['passed'] else '✗ FAIL'}\n\n")
        
        f.write("### Plateau Test (Asymptotic Grid)\n")
        f.write(f"- CV(J/γ): {pm['cv']:.4f}\n")
        f.write(f"- Mean(J/γ): {pm['mean_j_over_gamma']:.6f}\n")
        f.write(f"- Slope (J ≈ aγ): {pm['slope']:.6f}\n")
        f.write(f"- R²: {pm['r_squared']:.4f}\n")
        f.write(f"- |slope - v_C|/v_C: {abs(pm['slope'] - v_c) / v_c:.4f}\n\n")
        
        f.write("## Acceptance\n\n")
        f.write(f"| Criterion | Threshold | Value | Status |\n")
        f.write(f"|-----------|-----------|-------|--------|\n")
        f.write(f"| Identity error | ≤ 1e-3 | {iv['median_error']:.2e} | {'PASS' if acc['identity_pass'] else 'FAIL'} |\n")
        f.write(f"| CV | ≤ 0.15 | {pm['cv']:.4f} | {'PASS' if acc['cv_pass'] else 'FAIL'} |\n")
        f.write(f"| R² | ≥ 0.98 | {pm['r_squared']:.4f} | {'PASS' if acc['r2_pass'] else 'FAIL'} |\n")
        f.write(f"| v_C proximity | ≤ 0.20 | {abs(pm['slope'] - v_c) / v_c:.4f} | {'PASS' if acc['vc_pass'] else 'FAIL'} |\n\n")
        
        f.write(f"**Overall: {'PASSED ✓' if acc['overall_pass'] else 'FAILED ✗'}**\n\n")
        
        f.write("## Output Files\n\n")
        f.write("- `results.csv`: Detailed results for each gamma value\n")
        f.write("- `fig_J_vs_gamma.png`: J^ε(γ) vs γ (log-log)\n")
        f.write("- `fig_R_vs_gamma.png`: R(γ) = J^ε(γ)/γ vs γ\n")
        f.write("- `fig_identity_check.png`: Identity verification\n")
        f.write("- `fig_plateau_zoom.png`: Plateau detail view\n")
        f.write("- `fit_report.md`: Statistical analysis report\n")


def write_results_csv(path: Path, results: Dict):
    """Write results.csv file."""
    gamma_grid = results['gamma_grid']['all']
    asymptotic_set = set(results['gamma_grid']['asymptotic'])
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        header = ['gamma', 'grid_type', 'J_epsilon_mean', 'J_epsilon_std']
        for seed in results['seeds']:
            header.extend([f'J_epsilon_seed{seed}', f'J_x0_seed{seed}'])
        header.extend(['J_x0_mean', 'J_x0_std', 'J_over_gamma'])
        writer.writerow(header)
        
        for i, gamma in enumerate(gamma_grid):
            grid_type = 'asymptotic' if gamma in asymptotic_set else 'global'
            row = [
                f"{gamma:.6e}",
                grid_type,
                f"{results['aggregated']['J_epsilon_mean'][i]:.6e}",
                f"{results['aggregated']['J_epsilon_std'][i]:.6e}",
            ]
            
            for seed in results['seeds']:
                seed_result = results['results_by_seed'][seed][i]
                row.extend([
                    f"{seed_result['J_epsilon']:.6e}",
                    f"{seed_result['J_x0']:.6e}" if seed_result['J_x0'] is not None else '',
                ])
            
            row.extend([
                f"{results['aggregated']['J_x0_mean'][i]:.6e}",
                f"{results['aggregated']['J_x0_std'][i]:.6e}",
                f"{results['aggregated']['J_epsilon_mean'][i] / gamma:.6e}" if gamma > 0 else 'inf',
            ])
            writer.writerow(row)


def write_fit_report(path: Path, results: Dict):
    """Write fit_report.md file."""
    v_c = results['gmm']['v_c']
    pm = results['plateau_metrics']
    iv = results['identity_verification']
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Plateau Statistical Report\n\n")
        
        f.write("## Theoretical Plateau Height\n\n")
        f.write(f"**v_C = {v_c:.6f}**\n\n")
        f.write("This is the between-condition prior shift energy:\n")
        f.write("```\n")
        f.write("v_C = (1/d) E || E[X|C] - E[X] ||^2\n")
        f.write("```\n\n")
        
        f.write("## Asymptotic Grid Statistics\n\n")
        f.write(f"- **Mean(R)**: {pm['mean_j_over_gamma']:.6f}\n")
        f.write(f"- **Std(R)**: {pm['std_j_over_gamma']:.6f}\n")
        f.write(f"- **CV(R)**: {pm['cv']:.4f}\n\n")
        
        f.write("## Linear Fit (J ≈ aγ)\n\n")
        f.write(f"- **Slope (a)**: {pm['slope']:.6f}\n")
        f.write(f"- **R²**: {pm['r_squared']:.4f}\n\n")
        
        f.write("## Proximity to v_C\n\n")
        rel_diff = abs(pm['slope'] - v_c) / v_c
        f.write(f"- |a - v_C| / v_C = {rel_diff:.4f}\n")
        f.write(f"- |Mean(R) - v_C| / v_C = {abs(pm['mean_j_over_gamma'] - v_c) / v_c:.4f}\n\n")
        
        f.write("## Identity Verification\n\n")
        f.write("J^ε(γ) = γ·J^{x0}(γ)\n\n")
        f.write(f"- **Median error**: {iv['median_error']:.6e}\n")
        f.write(f"- **Max error**: {iv['max_error']:.6e}\n\n")
        
        f.write("## Verdict\n\n")
        if results['acceptance']['overall_pass']:
            f.write("**PASS** - All criteria met. Ready for Phase 2.\n")
        else:
            f.write("**FAIL** - Some criteria not met. Review implementation.\n")


def main():
    """Main entry point for Phase 1 experiment."""
    parser = argparse.ArgumentParser(description="Phase 1: Oracle Toy Validation")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/phase1.yaml)'
    )
    
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold cyan]Phase 1: Oracle Toy Validation[/bold cyan]")
    console.print("=" * 50)
    
    config = load_config(args.config)
    
    results = run_phase1_experiment(config, console)
    
    save_results(results, config, console)
    
    console.print("\n[bold green]Phase 1 Experiment Complete![/bold green]")
    
    if results['acceptance']['overall_pass']:
        console.print("[green]✓ All acceptance criteria met. Ready for Phase 2.[/green]")
    else:
        console.print("[yellow]⚠ Some acceptance criteria not met. Review results.[/yellow]")
    
    return results


if __name__ == "__main__":
    main()
