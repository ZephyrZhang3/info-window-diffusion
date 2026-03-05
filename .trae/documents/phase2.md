# Phase 2 Plan (Agent-ready): Real-model plateau test with Stable Diffusion v1.5 + Diffusers
> Goal: Measure the **CFG-aligned ε-gap energy proxy** on *real pretrained models* using **forward-corrupt** latents, and test whether the high-noise region exhibits a **linear regime / stable ratio**:
>
> - \(J_k^{fc} := \frac{1}{d}\mathbb E\|\hat\epsilon(z_k,t_k,c)-\hat\epsilon(z_k,t_k,\varnothing)\|_2^2\)
> - \(R_k^{fc} := J_k^{fc}/\gamma_k\), where \(\gamma_k=\alpha_k^2/\sigma_k^2\)
>
> This plan includes **exact links** (provided as URLs in code blocks) and concrete defaults.

---

## 0) Links (open these first)

### Model + training detail (CFG-ready uncond branch)
```text
Stable Diffusion v1.5 model card (mentions "10% dropping of the text-conditioning"):
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

Plain README for SD v1.5 (same "10% dropping" line, easier to grep):
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/README.md?download=true

CompVis Stable Diffusion page (also mentions 10% conditioning drop in SD training lineage):
https://huggingface.co/CompVis/stable-diffusion
````

Evidence: SD model cards explicitly mention “10% dropping of the text-conditioning to improve classifier-free guidance sampling”. ([Hugging Face][1])

### CFG implementation reference (Diffusers)

```text
Hugging Face blog "Stable Diffusion with Diffusers" (shows concatenating uncond/cond embeddings into one batch):
https://huggingface.co/blog/stable_diffusion
```

This post states CFG requires two forward passes (cond/uncond) and recommends concatenating into one batch. ([Hugging Face][2])

### Scheduler docs

```text
DDIMScheduler docs:
https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddim

DPMSolverMultistepScheduler docs (recommends solver_order=2 for guided sampling; dpmsolver++):
https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver
```

DPMSolverMultistepScheduler docs explicitly recommend `solver_order=2` for guided sampling. ([Hugging Face][3])

---

## 1) Phase 2 scope + success criteria (keep this tight)

**We are NOT optimizing image quality yet.** We are only validating whether the *observable proxy* (R_k^{fc}=J_k^{fc}/\gamma_k) shows a stable/linear regime in the **highest-noise** part of the schedule.

### PASS (weak)

For at least one inference grid (K=50 or K=100):

* In the **smallest-γ** 20% of timesteps:

  * slope (\hat\beta) from fit (\log J_k^{fc}) vs (\log \gamma_k) satisfies (|\hat\beta-1|\le 0.25)
  * CV of (R_k^{fc}) in that region (\le 0.25)
  * repeatable across seeds (mean(R) differs by ≤20%)

### PASS (strong)

* (|\hat\beta-1|\le 0.15), CV ≤0.15, seed difference ≤10%

---

## 2) Fixed experimental choices (do not “tune” these in Phase 2)

### 2.1 Pretrained model

* Use **Stable Diffusion v1.5** in Diffusers.
* Use **512×512** resolution (native setting).

Model link above. ([Hugging Face][4])

### 2.2 Data

To avoid licensing headaches and keep everything public:

* Use **MS-COCO 2017 val** (5k images) as (x_0) source.
* Use one caption per image as the condition prompt.

(Any public COCO loader is fine; Phase 2 only needs images + captions.)

### 2.3 Condition / uncondition definition

* **cond prompt**: a COCO caption string
* **uncond prompt**: empty string `""` (Diffusers standard CFG uncond)

This matches the Diffusers CFG setup described in the blog. ([Hugging Face][2])

### 2.4 Where diffusion happens

SD is **latent diffusion**. We measure everything in **VAE latent space**:

* encode image (x_0) → latent (z_0)
* forward-corrupt: (z_k=\alpha_k z_0 + \sigma_k \epsilon)

---

## 3) Inference grids (the only “k” set we will measure)

We do **not** measure all 1000 training steps. We measure the **inference grid**:

* Grid A: `K = 50` timesteps
* Grid B: `K = 100` timesteps

For each grid:

1. Create scheduler timesteps via `scheduler.set_timesteps(K)`
2. The resulting `timesteps[]` define the measured indices (t_k)

Store per-k:

* `t_k`
* `alpha_bar_k = scheduler.alphas_cumprod[t_k]`
* `alpha_k = sqrt(alpha_bar_k)`
* `sigma_k = sqrt(1-alpha_bar_k)`
* `gamma_k = alpha_k^2/sigma_k^2 = alpha_bar_k/(1-alpha_bar_k)`

Output files:

* `grid_K50.csv`, `grid_K100.csv`

---

## 4) forward-corrupt measurement protocol (main result)

### 4.1 Key variance control (non-negotiable)

For each (image, noise, timestep):

* Construct **one** (z_k^{fc})
* Evaluate **cond** and **uncond** on the **same** (z_k^{fc})

Do NOT use separate noises for cond/uncond.

### 4.2 How to run UNet efficiently (recommended)

Use the “concat embeddings into one batch” method:

* concatenate `[uncond_embedding, cond_embedding]` along batch dimension
* run UNet once to get `[eps_u, eps_c]`

Reference: HF blog code pattern. ([Hugging Face][2])

### 4.3 Exact definitions

For each k:

* ( \Delta_k = \hat\epsilon(z_k,t_k,\text{cond}) - \hat\epsilon(z_k,t_k,\text{uncond}))
* ( J_k^{fc} = \mathrm{mean}(\Delta_k^2))  (mean over batch + channels + H + W)
* ( R_k^{fc} = J_k^{fc}/\gamma_k)

**Numerical detail (recommended):**

* UNet forward in fp16/bf16 is fine
* Accumulate `Delta^2` in fp32

### 4.4 Default compute scale

* seeds: `{0,1,2}`
* images: start with COCO val 5k; if too noisy, repeat with 10k/20k (sample with replacement)
* batch size: 8–16 depending on VRAM

Outputs (per grid, per seed):

* `Jk_fc_seed{seed}_K{K}.csv` with columns:

  * `k, t_k, gamma_k, Jk_fc, Rk_fc, n_samples`

Aggregate:

* `Jk_fc_K{K}.csv` with mean/std across seeds.

---

## 5) Visualization (paper-ready for Phase 2)

For each grid K:

1. `fig_fc_Jk_vs_loggamma_K{K}.png`

   * x = log10(gamma_k)
   * y = log10(Jk_fc) (or log scale)
2. `fig_fc_Rk_vs_loggamma_K{K}.png`

   * x = log10(gamma_k)
   * y = Rk_fc
3. `fig_fc_plateau_zoom_K{K}.png`

   * only plot the smallest-γ 20% points
   * annotate CV + slope β_hat

**Suggestion:** color-code the smallest-γ 20% subset.

---

## 6) Plateau/linear-regime statistics (fit_report_phase2.md)

For each grid K:

### 6.1 Define high-noise subset S

* sort points by `gamma_k` ascending
* take first `ceil(0.2*K)` points as S (smallest-γ 20%)

### 6.2 Fit slope β_hat

* Fit `log(Jk_fc)` vs `log(gamma_k)` on S
* Report β_hat and R²

### 6.3 Stability of R_k

* Compute mean(R), std(R), CV(R) on S
* Report seed-to-seed variation of mean(R)

### 6.4 PASS/FAIL

Use criteria in Section 1.

---

## 7) Optional but recommended: chain-estimate contrast (to justify forward-corrupt)

Purpose: show that measuring on *inference-chain states* is confounded by distribution shift.

### 7.1 Choose two schedulers (same K)

* **DDIMScheduler** baseline
* **DPMSolverMultistepScheduler** (dpmsolver++, solver_order=2)

Docs:

* DDIMScheduler ([Hugging Face][5])
* DPMSolverMultistepScheduler ([Hugging Face][3])

### 7.2 Chain configs

* K = 50
* guidance scale s ∈ {0, 6, 10}
* generate 1k samples (enough)

At each chain step, compute:

* `Jk_chain`, `Rk_chain` using same Δ = eps_c - eps_u but evaluated at `z_k^chain`

Output:

* `Jk_chain_K50_s{scale}_{scheduler}.csv`
* `fig_chain_vs_fc_Rk.png` overlaying chain vs fc.

Expected: chain R_k shows stronger dependence on scheduler/scale (more confounded) than fc R_k.

---

## 8) Troubleshooting checklist (if FAIL)

Run in this order:

1. **Uncond branch sanity**: Does empty prompt `""` actually change the embedding path? (It should; SD trained with conditioning drop.) ([Hugging Face][6])
2. **Same z_k for cond/uncond**: verify identical inputs except embeddings.
3. **Gamma computation**: verify `gamma_k = alpha_bar/(1-alpha_bar)` using scheduler’s `alphas_cumprod[t]`.
4. **Precision**: try fp32 accumulation for `Delta^2` and/or run UNet in bf16.
5. **Sample size**: increase N_images in the smallest-γ region (importance sampling over k is allowed).

---

## 9) Directory layout (agent should follow)

```
phase2_real/
  README_phase2.md
  env_check.md
  grids/
    grid_K50.csv
    grid_K100.csv
  fc/
    Jk_fc_seed0_K50.csv
    Jk_fc_seed1_K50.csv
    Jk_fc_seed2_K50.csv
    Jk_fc_K50.csv
    (same for K100)
  chain/   # optional
    Jk_chain_*.csv
  figs/
    fig_fc_Jk_vs_loggamma_K50.png
    fig_fc_Rk_vs_loggamma_K50.png
    fig_fc_plateau_zoom_K50.png
    (same for K100)
    fig_chain_vs_fc_Rk.png
  fit_report_phase2.md
```

---

## 10) Minimal “definition of done”

Phase 2 is complete when:

* You have `Jk_fc_K50.csv` and/or `Jk_fc_K100.csv` with seed-aggregated stats
* You have the three forward-corrupt figures per grid
* `fit_report_phase2.md` includes slope/CV/seed variation + PASS/FAIL

Only after PASS do we proceed to Phase 3 (interval/schedule performance).

```text
[1]: https://huggingface.co/CompVis/stable-diffusion?utm_source=chatgpt.com "CompVis/stable-diffusion"
[2]: https://huggingface.co/blog/stable_diffusion?utm_source=chatgpt.com "Stable Diffusion with 🧨 Diffusers"
[3]: https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver?utm_source=chatgpt.com "DPMSolverMultistepScheduler"
[4]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5?utm_source=chatgpt.com "stable-diffusion-v1-5/stable-diffusion-v1-5"
[5]: https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddim?utm_source=chatgpt.com "DDIMScheduler"
[6]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/README.md?download=true&utm_source=chatgpt.com "14.5 kB"
```