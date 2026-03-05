# Phase 1 代码重构计划

## 背景

根据 `phase1.md` 文档的要求，需要重新实现 Phase 1 实验。当前实现基于扩散模型的时间步调度，但 `phase1.md` 要求使用**固定的γ网格**进行 Monte Carlo 估计，而不是通过扩散调度获取γ值。

## 主要差异分析

### 1. γ 网格设计差异

| 项目 | 当前实现 | phase1.md 要求 |
|------|----------|----------------|
| γ 来源 | 扩散调度（cosine/linear）计算 | 固定网格 |
| 全局网格 | 无 | γ ∈ {1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3} |
| 渐近网格 | 无 | γ ∈ {3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6} |

### 2. 样本量分配差异

| 项目 | 当前实现 | phase1.md 要求 |
|------|----------|----------------|
| 标准样本量 | 200,000 | 全局网格: 200,000 |
| 高精度样本量 | 1,000,000 (尾端区域) | 渐近网格: 1,000,000 |

### 3. 输出结构差异

| 项目 | 当前实现 | phase1.md 要求 |
|------|----------|----------------|
| 输出目录 | outputs/phase1/ | phase1_oracle/ |
| README | 无 | README_phase1.md |
| 数据文件 | timestep_stats.csv | results.csv |
| 图表 | 多个子目录 | 4个PNG文件 |
| 报告 | summary.txt | fit_report.md |

### 4. 验收标准差异

| 指标 | 当前实现 | phase1.md 要求 |
|------|----------|----------------|
| CV 阈值 | 0.10 (严格) / 0.15 (宽松) | ≤ 0.15 |
| R² 阈值 | 0.95 | ≥ 0.98 |
| v_C 接近度 | 无明确要求 | \|a - v_C\| / v_C ≤ 0.20 |
| 恒等核对 | 无明确阈值 | median(err_id) ≤ 1e-3 |

## 实施计划

### 步骤 1: 删除 Phase 2 相关代码

**删除文件：**
- `iw_dm/experiments/run_phase2.py`
- `configs/phase2.yaml`
- `iw_dm/real/` 目录下所有文件：
  - `iw_dm/real/__init__.py`
  - `iw_dm/real/chain_estimate.py`
  - `iw_dm/real/data_loader.py`
  - `iw_dm/real/forward_corrupt.py`
  - `iw_dm/real/metrics.py`
  - `iw_dm/real/model_loader.py`
- `outputs/phase2/` 目录

### 步骤 2: 重构配置文件

**修改 `configs/phase1.yaml`：**
- 移除 `diffusion` 配置块
- 移除 `sampling` 配置块
- 移除 `high_noise_region` 配置块
- 添加 `gamma_grid` 配置块：
  ```yaml
  gamma_grid:
    global: [1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    asymptotic: [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
  ```
- 更新 `monte_carlo` 配置：
  ```yaml
  monte_carlo:
    n_samples_global: 200000
    n_samples_asymptotic: 1000000
    seeds: [0, 1, 2]
  ```
- 更新 `output` 配置：
  ```yaml
  output:
    base_dir: phase1_oracle
  ```
- 更新 `plateau_test` 验收标准

### 步骤 3: 重构 `iw_dm/experiments/run_phase1.py`

**主要修改：**
1. 移除扩散调度相关导入和代码
2. 使用固定γ网格替代时间步采样
3. 实现两阶段样本量分配（全局网格 vs 渐近网格）
4. 重构 `run_phase1_experiment()` 函数
5. 更新输出文件生成逻辑：
   - 生成 `README_phase1.md`
   - 生成 `results.csv`（包含 seed 统计）
   - 生成 4 个 PNG 图表
   - 生成 `fit_report.md`

### 步骤 4: 重构 `iw_dm/utils/visualization.py`

**修改内容：**
1. 添加新函数：
   - `plot_J_vs_gamma()` - J^ε(γ) vs γ (log-log)
   - `plot_R_vs_gamma()` - R(γ) = J^ε(γ)/γ vs γ
   - `plot_identity_check()` - J^ε(γ) vs γ·J^{x0}(γ)
   - `plot_plateau_zoom()` - 小γ区间的 R(γ)
2. 移除时间步相关的可视化函数（或保留但不使用）

### 步骤 5: 重构 `iw_dm/utils/snr.py`

**修改内容：**
1. 移除扩散调度相关函数（`create_diffusion_schedule`, `get_cosine_schedule` 等）
2. 保留 `gamma_grid_toy()` 函数并更新为 phase1.md 要求的网格
3. 保留 `compute_gamma_from_alpha_bar()` 等工具函数（可能被其他模块使用）

### 步骤 6: 更新 `iw_dm/utils/statistics.py`

**修改内容：**
1. 更新 `plateau_test()` 函数以支持渐近网格的统计
2. 添加 `compute_identity_error()` 函数用于恒等核对
3. 更新 `check_plateau_criteria()` 函数以匹配新的验收标准

### 步骤 7: 保持不变的模块

以下模块无需修改：
- `iw_dm/toy/gmm.py` - GMM 分布实现符合要求
- `iw_dm/toy/bayes.py` - Bayes-optimal 计算实现符合要求
- `iw_dm/toy/metrics.py` - Monte Carlo 估计实现基本符合要求，可能需要小幅调整

### 步骤 8: 更新输出目录结构

**新的输出结构：**
```
phase1_oracle/
├── README_phase1.md      # 运行说明与结论摘要
├── results.csv           # 每个 γ 的估计值（含 seed 统计）
├── fig_J_vs_gamma.png    # J^ε(γ) vs γ (log-log)
├── fig_R_vs_gamma.png    # R(γ) vs γ
├── fig_identity_check.png # J^ε(γ) vs γ·J^{x0}(γ)
├── fig_plateau_zoom.png  # 小γ区间的 R(γ)
└── fit_report.md         # plateau 统计报告
```

## 验收检查清单

完成重构后，需验证：

1. **恒等核对**：median(err_id) ≤ 1e-3
2. **Plateau test**：
   - CV ≤ 0.15
   - R² ≥ 0.98
   - |a - v_C| / v_C ≤ 0.20
3. **输出完整性**：所有要求的文件都已生成
4. **图形正确性**：图表正确显示 plateau 现象

## 风险评估

1. **数值稳定性**：γ 很小时（如 1e-6）可能出现 underflow，需使用 logsumexp 技术
2. **计算时间**：渐近网格使用 1,000,000 样本，计算时间较长
3. **内存使用**：大批量采样可能需要分批处理

## 预计工作量

| 步骤 | 预计时间 |
|------|----------|
| 删除 Phase 2 代码 | 5 分钟 |
| 重构配置文件 | 10 分钟 |
| 重构 run_phase1.py | 30 分钟 |
| 重构 visualization.py | 20 分钟 |
| 重构 snr.py | 10 分钟 |
| 更新 statistics.py | 15 分钟 |
| 测试验证 | 20 分钟 |
| **总计** | **约 1.5 小时** |
