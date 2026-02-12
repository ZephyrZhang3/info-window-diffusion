# Information Windows in Diffusion Models

**Paper title (draft):** *Information Windows in Diffusion Models: Why High-Noise Control Must Fail*  
**Keywords:** diffusion models, high-noise regime, classifier-free guidance, information theory, I–MMSE, rate–distortion, stabilization

This repository contains experimental code and reproducible scripts for the paper above.  
Our goal is to **unify** a set of widely observed high-noise phenomena in diffusion sampling—**prior reversion / posterior collapse**, **conditioning & guidance failure**, and the need for **stabilization under large guidance**—under a single information-theoretic framework.

---

## 1. Core Idea

Diffusion forward marginals can be viewed as a family of Gaussian channels.  
We characterize the **high-noise region** as an **information window**, where the per-dimension mutual information between data and noisy state is small:
- unconditional:  $r_t := \frac{1}{d} I(X; X_t)$
- conditional:    $r_{c,t} := \frac{1}{d} I(X; X_t \mid C)$

**Main thesis:** In this window, the **Bayes-optimal denoising gain** is information-limited.  
As a consequence, *any* control mechanism that relies on conditional information (e.g., CFG) must be ineffective (or even harmful) in the high-noise regime.

---

## 2. What We Validate Empirically (Predictions)

We convert the theory into falsifiable predictions for real diffusion models.

### P1 — Window-predicted optimal interval for guidance
There exists a time interval where guidance is useful, and it should **avoid the low-information window**.  
We estimate a measurable proxy of conditional signal strength:

$$
J_t := \frac{1}{d}\mathbb{E}\big\|s_\theta(x_t,t,c) - s_\theta(x_t,t,\varnothing)\big\|_2^2.
$$

We predict the best guidance onset time by a threshold crossing of $J_t$, and verify it by grid-searching
guidance activation intervals $[t_\mathrm{on}, t_\mathrm{off}]$.

### P2 — Why monotone schedules work (budget alignment)
Monotone guidance schedules succeed because they implicitly **track information availability**.  
We propose a reproducible, non-learned schedule driven by $J_t$, and compare it against common baselines.

### P3 — Stabilizers improve stability, not information
Stabilization (e.g., dynamic thresholding, solver controls) expands the **stability boundary** (max usable guidance scale),
but does **not** move the **usefulness boundary** (guidance still cannot help inside the information window).

---

## 3. Repository Structure (planned)

