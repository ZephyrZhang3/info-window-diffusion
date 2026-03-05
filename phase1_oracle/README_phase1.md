# Phase 1: Oracle Toy Validation Results

## Overview

This experiment validates the plateau law in a Bayes-optimal toy setting:
- **Goal**: Verify that J^ε(γ)/γ → v_C as γ → 0
- **Method**: Monte Carlo estimation with conditional GMM

## Configuration

### GMM Parameters
- Dimension (d): 32
- Mean shift (m): 8.0
- Std dev (s): 1.0
- **v_C (theoretical plateau)**: 2.500000

### Gamma Grid
- Global grid: [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
- Asymptotic grid: [0.0003, 0.0001, 3e-05, 1e-05, 3e-06, 1e-06]

### Monte Carlo
- Seeds: [0, 1, 2]
- Samples (global): 20,000
- Samples (asymptotic): 100,000

## Results Summary

### Identity Verification
  J^ε(γ) = γ·J^{x0}(γ)
- Median relative error: 1.416118e-16
- Status: ✓ PASS

### Plateau Test (Asymptotic Grid)
- CV(J/γ): 0.0071
- Mean(J/γ): 2.512568
- Slope (J ≈ aγ): 2.546645
- R²: 1.0000
- |slope - v_C|/v_C: 0.0187

## Acceptance

| Criterion | Threshold | Value | Status |
|-----------|-----------|-------|--------|
| Identity error | ≤ 1e-3 | 1.42e-16 | PASS |
| CV | ≤ 0.15 | 0.0071 | PASS |
| R² | ≥ 0.98 | 1.0000 | PASS |
| v_C proximity | ≤ 0.20 | 0.0187 | PASS |

**Overall: PASSED ✓**

## Output Files

- `results.csv`: Detailed results for each gamma value
- `fig_J_vs_gamma.png`: J^ε(γ) vs γ (log-log)
- `fig_R_vs_gamma.png`: R(γ) = J^ε(γ)/γ vs γ
- `fig_identity_check.png`: Identity verification
- `fig_plateau_zoom.png`: Plateau detail view
- `fit_report.md`: Statistical analysis report
