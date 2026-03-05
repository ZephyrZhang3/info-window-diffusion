# Phase 1 实验方案（V10）：Oracle Toy 上验证 high-noise plateau law
> 目的：在完全可解（Bayes-optimal）的 toy setting 中，验证  
> **(i)** 低 SNR（γ→0）时 `J^ε(γ) ≈ v_C·γ`（little-o 口径），  
> **(ii)** `J^ε(γ)/γ → v_C` 的 **plateau law**，并给出可复现的统计验收。  
> 本方案不涉及 Phase 2/3，不涉及真实模型，不涉及任何工程推理链。

---

## 0. 产出物（交付标准）
你需要交付一个目录 `phase1_oracle/`，至少包含：

1) `README_phase1.md`：运行说明与结论摘要（含验收是否通过）
2) `results.csv`：每个 γ 的估计值（含 seed 统计）
3) 图（PNG）：
   - `fig_J_vs_gamma.png`：`J^ε(γ)` vs `γ`（log-log）
   - `fig_R_vs_gamma.png`：`R(γ)=J^ε(γ)/γ` vs `γ`（横轴 log γ）
   - `fig_identity_check.png`（可选但强烈推荐）：`J^ε(γ)` vs `γ·J^{x0}(γ)` 的一致性检验
   - `fig_plateau_zoom.png`：只画小 γ 区间的 `R(γ)`（更清晰）
4) `fit_report.md`：plateau 统计（CV、线性拟合 R²、与 v_C 的接近度）

---

## 1. 理论对象与定义（必须一致）
### 1.1 AWGN family（实验通道）
对每个 γ≥0，生成观测：
- `Y = sqrt(γ)·X + ε`, 其中 `ε ~ N(0, I_d)`，独立于 X。

### 1.2 条件变量
- 条件 `C ∈ {0,1}`（二类），`P(C=0)=P(C=1)=0.5`。

### 1.3 要计算的 Bayes-optimal 量
令 `d` 为 X 的维度。

1) posterior mean gap（x0-space）：
- `Δμ(y,c) := E[X | Y=y, C=c] - E[X | Y=y]`
- `J^{x0}(γ) := (1/d) E ||Δμ(Y,C)||^2`

2) ε-gap（CFG-aligned 注入量）：
- `Δε(y,c) := E[ε | Y=y, C=c] - E[ε | Y=y]`
- `J^ε(γ) := (1/d) E ||Δε(Y,C)||^2`

3) plateau ratio：
- `R(γ) := J^ε(γ) / γ`（只对 γ>0 计算；γ=0 单独处理）

### 1.4 必须做的恒等核对（避免实现错误）
由 `ε = Y - sqrt(γ)X`：
- `E[ε | Y, C] = Y - sqrt(γ) E[X | Y, C]`
- `E[ε | Y] = Y - sqrt(γ) E[X | Y]`
因此：
- `Δε = -sqrt(γ)·Δμ`
- **应当满足** `J^ε(γ) = γ·J^{x0}(γ)`（数值误差允许很小偏差）

### 1.5 理论平台高度（可计算的 ground-truth）
定义：
- `μ := E[X]`
- `μ_C := E[X | C]`
- `v_C := (1/d) E || μ_C - μ ||^2`   （between-condition prior shift energy）

Phase 1 的核心主张是：
- `R(γ) = J^ε(γ)/γ → v_C` 当 `γ→0`

---

## 2. Toy 分布设计（必须满足“可精确推 posterior”）
### 2.1 分布：conditional Gaussian mixture model（GMM）
选择 `d=2`（默认），每个条件下 2 个 component（默认）。

- `C ∈ {0,1}`，均匀
- 在给定 C=c 下：
  - component `K ∈ {1,2}`，均匀
  - `X | (C=c, K=k) ~ N( μ_{c,k}, Σ_{c,k} )`

### 2.2 默认参数（推荐直接用）
- `d = 2`
- `m = 4.0`
- `s = 0.5`
- means：
  - `μ_{0,1} = (+m, 0)`
  - `μ_{0,2} = (+m, +m)`
  - `μ_{1,1} = (-m, 0)`
  - `μ_{1,2} = (-m, -m)`
- covariances：
  - `Σ_{c,k} = diag(s^2, s^2)` 对所有 (c,k)

### 2.3 v_C 的解析计算（必须输出）
对该 GMM，先算：
- `μ_C = 0.5(μ_{c,1}+μ_{c,2})`
- `μ = 0.5(μ_0 + μ_1)`
然后：
- `v_C = (1/d) * 0.5||μ_0-μ||^2 + (1/d) * 0.5||μ_1-μ||^2`
你必须把 v_C 写入 `fit_report.md` 并画在 `fig_R_vs_gamma.png` 上作为虚线。

> 备注：该默认设置下 v_C 是一个非零常数，平台应明显可见。

---

## 3. Bayes-optimal 计算方法（可复制实现规格）
对每个 γ 和每个 y：

### 3.1 component likelihood
对每个 (c,k)：
- `Y | (c,k)` 是高斯：
  - mean: `sqrt(γ) μ_{c,k}`
  - cov:  `I_d + γ Σ_{c,k}`

记：
- `L_{c,k}(y) := N(y; sqrt(γ) μ_{c,k}, I + γ Σ_{c,k})`

### 3.2 posterior weights
- joint prior：`P(c,k)=P(c)P(k|c)=0.25`
- posterior：
  - `w_{c,k}(y) = P(c,k|y) = P(c,k)L_{c,k}(y) / Z(y)`
  - `Z(y)=Σ_{c',k'} P(c',k')L_{c',k'}(y)`

- conditional-on-C posterior：
  - `w_{k|c}(y) = P(k|y,c) ∝ P(k|c) L_{c,k}(y)`，归一化 over k

### 3.3 component posterior mean
对线性高斯：
- `E[X | y, c,k] = μ_{c,k} + Σ_{c,k} sqrt(γ) (I + γ Σ_{c,k})^{-1} (y - sqrt(γ) μ_{c,k})`

### 3.4 aggregate posterior mean
- `E[X | y] = Σ_{c,k} w_{c,k}(y) E[X | y, c,k]`
- `E[X | y, c] = Σ_{k} w_{k|c}(y) E[X | y, c,k]`

然后计算：
- `Δμ(y,c)=E[X|y,c]-E[X|y]`
- `Δε(y,c)=-sqrt(γ)Δμ(y,c)`（建议直接用这个等式，数值更稳）

---

## 4. Monte Carlo 估计（必须可复现）
### 4.1 γ 网格（两阶段，保证既能看全局又能看渐近）
使用固定列表（不要自适应）：

**全局网格（用于形状）**
- `γ ∈ {1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3}`

**渐近网格（用于 plateau）**
- `γ ∈ {3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6}`

> 备注：若你担心数值 underflow，可把最小 γ 提到 `1e-5` 起步；但最终需至少做到 `1e-6` 才算“强验证”。

### 4.2 样本量（分配策略）
为了省算力，把样本集中在小 γ：

- 全局网格每个 γ：`N = 200,000`
- 渐近网格每个 γ：`N = 1,000,000`

### 4.3 随机种子
- seeds：`{0,1,2}` 三次独立重复
- 每个 seed 对每个 γ 都要跑，输出 mean/std

### 4.4 采样流程（每个 γ、每个 seed）
重复 N 次：
1) sample `C ~ Bernoulli(0.5)`
2) sample `K ~ Uniform{1,2}` conditional on C
3) sample `X ~ N(μ_{c,k}, Σ_{c,k})`
4) sample `ε ~ N(0,I)`
5) compute `Y = sqrt(γ)X + ε`
6) compute `Δμ(Y,C)` via Bayes rule above
7) compute `Δε = -sqrt(γ)Δμ`
8) accumulate:
   - `j_x0 = (1/d)||Δμ||^2`
   - `j_eps = (1/d)||Δε||^2`
并最终得到：
- `J^{x0}(γ)`（mean over samples）
- `J^ε(γ)`（mean over samples）
- `R(γ)=J^ε(γ)/γ`

---

## 5. 统计检验与验收（必须写入 fit_report.md）
### 5.1 恒等核对（必要）
对每个 γ，计算相对误差：
- `err_id(γ) = |J^ε(γ) - γ J^{x0}(γ)| / max(J^ε(γ), 1e-12)`

验收：
- 渐近网格中 `median(err_id) <= 1e-3`（允许更松：1e-2）
- 若不满足：优先排查 posterior 计算实现错误或数值稳定性问题

### 5.2 Plateau test（核心）
只在渐近网格上做 plateau 统计（不要混入全局点）。

定义：
- `R_i = R(γ_i)` for γ_i in asymptotic grid
- `CV = std(R_i) / mean(R_i)`
- 线性拟合（过原点）：
  - 拟合 `J^ε(γ) ≈ a γ`（仅用渐近网格）
  - 报告 `a` 与 `R²`

验收（强版本）：
- `CV <= 0.15`
- `R² >= 0.98`
- 与 ground-truth v_C 的接近性：
  - `|a - v_C| / v_C <= 0.20`
  - 等价地 `|mean(R_i) - v_C| / v_C <= 0.20`

> 若 CV 很小但 mean(R) 明显偏离 v_C：说明你还在 pre-asymptotic（需要更小 γ 或更大 N）。

### 5.3 报告模板（必须包含）
在 `fit_report.md` 输出以下字段：
- v_C（解析值）
- 渐近网格的 mean(R)、std(R)、CV
- 线性拟合 slope a、R²
- 相对误差：|a-v_C|/v_C
- 恒等核对 err_id 的 median/max
- “PASS/FAIL” 判定（按上面阈值）

---

## 6. 可选鲁棒性实验（加分，但不要求）
若时间允许，做一个“轻鲁棒性”：
- 改 d=4（把 means padding 到 4D，cov diag）
- 或改 component covariance 为非等方差（例如 s_x=0.5, s_y=1.0）
重复一遍渐近网格，观察 plateau 是否仍成立（不要求完全同 v_C，但应有清晰收敛与平台）。

---

## 7. 实现建议（不规定语言，但必须满足数值稳定）
- 计算高斯 log-likelihood 用 `logsumexp` 做归一化，避免 underflow（特别是 γ 很小/很大时）
- `Δε` 建议用 `-sqrt(γ)Δμ` 而不是 `E[ε|·]` 直接差分（更稳）
- 所有矩阵操作利用 `I + γΣ` 的解析形式（Σ 是对角更快、更稳定）
- 运行时间控制：先只跑一个 seed + 渐近网格检查是否接近 v_C，再开全量 seeds 与全局网格

---

## 8. 最终“可以进入 Phase 2 的条件”
Phase 1 只有在满足以下全部条件时，才算完成：
- 恒等核对通过（err_id 小）
- 渐近网格 plateau test 通过（CV、R²、接近 v_C）
- 图形与统计一致（R(γ) 在小 γ 段明显趋于常数并靠近 v_C）

否则：不要进入 Phase 2；先排查实现/样本量/γ 网格。