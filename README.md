# 基于 SemBind 的扩散模型语义水印防御与鲁棒性研究

**当前研究方向（2026-11-06 更新）：** 面向 **CCF-A 类会议**，研究重心已由早期的黑盒伪造攻击转向 **基于 SemBind（ICML 2026）的语义水印防御端与鲁棒性提升**。核心目标是系统剖析 SemBind 及 2025–2026 最新防御方案（ISTS、PGID、Geometry、SEAL、PAI、Sign-Clamping 等）的安全漏洞与鲁棒性缺陷，提出兼具**高抗伪造能力（含自适应攻击）**、**强失真/编辑鲁棒性**与**单遍生成高效性**的新一代语义绑定水印防御框架（**SemBind+**）。

---

## 1. 核心发现：SemBind 与现有防御的漏洞诊断（已验证实测证据）

在前期复现与基线评估（2026-11-02 / 11-03）中，我们对 **SemBind（ICML 2026）**、**PGID（NeurIPS 2026）**、**Geometry（ICML 2026）** 和 **Gaussian Shading（GS）** 进行了代码级审计与实测，发现了支撑新防御研究的四个关键切入点（详见 **[漏洞剖析与 CCF-A 改进方案](docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md)**）：

1. **SemBind 跨图语义哈希自然碰撞与多数投票放大漏洞（已实测证实）**
   - **位级复现**：我们实现的 `SemanticMasker.code()` 与官方 `AttnSupConHashingModel.forward(stage='binarize', s=12.0)` 在 20 张测试图及失真变体上 **Hamming 距离恒为 0**（见 [SemBind Masker 复现报告](reports/repro_sembind_masker_20261103/report.md)）。
   - **自然碰撞证据**：在仅 20 张不同内容载体（190 个跨图配对）中，两个语义完全不同的图像出现了 **Hamming 距离仅 29/1024（位一致率高达 97.2%）** 的自然碰撞！
   - **伪造绕过后果**：由于底层 GS 采用 $64 \times 256$ 重复多数投票（数学边界见 [GS 解码与投票契约](reports/22_decoder_contract/report.md)），部分掩码重合会被多数投票指数级放大。在 30 对、仅 30 步的普通黑盒 Imprint 攻击下，**28 对已测样本中有 2 对直接攻破 SemBind-GS，最高伪造解绑 BA 达 0.9844**（见 [防御基线评估报告](reports/defense_baselines_20261103/report.md)）。
2. **SemBind 公开骨干特征瓶颈与“无密钥特征对齐”绕过风险**
   - SemBind 虽将下游哈希网络与置换密钥 $\pi$ 设为私有，但其输入完全依赖**公开且冻结的 `DINOv2-giant` `[CLS]` 向量**。攻击者无需知道私有 masker 权重或置换密钥，只需在 Imprint 优化中增加公开 `DINOv2 [CLS]` 余弦对齐项，即可迫使未知私有 masker 输出完全相同的掩码。
3. **SemBind 硬二值化悬崖、空间盲区与双倍生成开销**
   - **硬二值化脆弱性**：轻度模糊与 $0.8\times$ 缩放即可导致最多 **78/1024 位翻转（7.6%）**；而通过降低掩码比例 $\sigma < 1$ 换取鲁棒性会使 $(1-\sigma)L$ 个潜坐标完全失去语义保护。
   - **空间盲区与双倍耗时**：仅使用全局 `[CLS]` token 无法防御局部区域篡改/拼接；且嵌入需先生成辅助干净图 $x_{\text{aux}}$（$2\times$ 扩散生成开销），并在 $x_{\text{aux}}$ 与最终水印图 $x_{\text{wm}}$ 之间引入跨轨迹语义漂移。
4. **同期其他前沿防御的实测局限**
   - **PGID（NeurIPS 2026）**：防御后位准确率随攻击步数增加呈单调上升（50/100/150 步下 BA 为 `0.5781 → 0.6211 → 0.6641`，距阈值 `0.7070` 仅差 4.3 个百分点，见 [PGID 复现报告](reports/repro_pgid_20261103/report.md)）。
   - **Geometry（ICML 2026）**：需有状态保存/检索原生成潜变量 $z_T$；实测全局 `G-Cos` AUC 为 `1.000`，但局部协方差指标 `L-SPD` 在论文 $16 \times 16$ 网格下 **AUC 仅为 0.480（随机水平）**，仅在 $32 \times 32$ 网格因秩亏退化产生假高分（见 [Geometry 复现报告](reports/repro_geometry_paper_task_20261103/report.md)）。
   - **ISTS（CVPR 2026 Findings）**：离散 $(t, l)$ 频域/时间步空间较小且聚类固定，双边检测针对符号反转移除，不防全潜空间复制伪造（见 [防御基线清单](docs/DEFENSE_BASELINE_INVENTORY.md)）。

---

## 2. SemBind+ 改进方案与研究路线（面向 CCF-A）

围绕上述漏洞，本项目提出三项递进的防御增强机制（详见 **[完整技术方案](docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md#4-面向-ccf-a-的核心方法设计sembind-防御与鲁棒性提升方案)** 与 **[研究计划](docs/RESEARCH_PLAN.md)**）：

1. **置信度感知的软解绑与加权 LLR 软判决译码（Margin-Aware Soft Unbinding & LLR Decoding）**
   - 保持 $\sigma = 1.0$（100% 坐标绑定，零未掩码泄漏），利用 masker 连续 logit 幅值 $|h_i|$ 与反演潜变量幅值 $|z_{\text{inv}}|$ 构造联合可靠度权重，在 $64 \times 256$ 解码器中执行软判决投票，消除边界位硬翻转导致的鲁棒性悬崖。
2. **密钥化多层空间-语义正交绑定（Keyed Multi-Granular Spatial-Semantic Binding）**
   - 引入验证端密钥化随机正交子空间投影（阻断基于公开 DINOv2 特征的无密钥对齐攻击）、平衡码本/最小距离约束（消除跨图长尾碰撞），并将 DINOv2 空间 Patch Tokens 与潜空间分块绑定，实现**局部篡改检测与定位（Tamper Localization）**。
3. **单遍轨迹自绑定与检测端流形协同（Single-Pass Trajectory Binding & Defense-in-Depth）**
   - 在单次扩散生成早期步利用 Tweedie 估计 $\hat{x}_{0|t}$ 提取语义掩码完成自洽绑定，将生成开销从 $2\times$ 降至 $\sim 1.15\times$ 并消除辅助图构图漂移；验证端可进一步与轻量去噪投影（PGID 思想）协同抵御高步数自适应攻击。

---

## 3. 仓库结构与核心文档导航

本仓库已清理早期纯攻击探索阶段的冗余过程产物，保留并整编所有服务于 **SemBind 防御改进与对比评估** 的核心代码、契约与实测报告：

### 3.1 研究设计与文献契约（`docs/`）
- **[`docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md`](docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md)**：SemBind 四大漏洞剖析、2025–2026 前沿防御对比及 SemBind+ 三大创新模块设计。
- **[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)**：面向 CCF-A 会议的阶段性实验路线图、验证指标与论文结构规划。
- **[`docs/DEFENSE_BASELINE_INVENTORY.md`](docs/DEFENSE_BASELINE_INVENTORY.md)**：7 项顶会防御（SemBind、ISTS、PGID、Geometry、SEAL、PAI、SLICE）与最新预印本（Sign-Clamping、CoverLock）盘点。
- **[`docs/LITERATURE.md`](docs/LITERATURE.md)**：扩散水印伪造与防御相关文献、官方源码版本与实现覆盖矩阵。
- **[`docs/EVALUATION_CONTRACT.md`](docs/EVALUATION_CONTRACT.md)**：GS、SemBind、Geometry 等评估对象与源码边界契约。
- **[`docs/SEAL_EVALUATION_CONTRACT.md`](docs/SEAL_EVALUATION_CONTRACT.md)** / **[`docs/PAI_EVALUATION_CONTRACT.md`](docs/PAI_EVALUATION_CONTRACT.md)** / **[`docs/CSI_INFORMATION_CONTRACT.md`](docs/CSI_INFORMATION_CONTRACT.md)**：SEAL（ICCV 2025）、PAI（ICLR 2026）与 CSI 自适应攻击（WWW 2026）的源码契约审计。
- **[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)** / **[`GPT_CONTEXT.md`](GPT_CONTEXT.md)** / **[`project_state.json`](project_state.json)**：当前项目状态与实验接续入口。

### 3.2 已验证实测报告与数据证据（`reports/`）
- **[`reports/repro_sembind_masker_20261103/report.md`](reports/repro_sembind_masker_20261103/report.md)**：SemBind 官方 masker 位级等价性核验、失真稳定性 R2 与跨图碰撞发现 R3。
- **[`reports/defense_baselines_20261103/report.md`](reports/defense_baselines_20261103/report.md)**：30 对基线在 GS、SemBind、PGID、Geometry 下的评估结果汇总。
- **[`reports/repro_pgid_20261103/report.md`](reports/repro_pgid_20261103/report.md)**：PGID（NeurIPS 2026）复现与随攻击步数单调退化分析。
- **[`reports/repro_geometry_paper_task_20261103/report.md`](reports/repro_geometry_paper_task_20261103/report.md)**：Geometry（ICML 2026）`G-Cos` 精确复现与 `L-SPD` 秩亏退化证明。
- **[`reports/diag_two_pair_identity_20261103/report.md`](reports/diag_two_pair_identity_20261103/report.md)**：逐对独立加密身份（Per-Pair Identity）与共享身份的对比诊断。
- **[`reports/22_decoder_contract/report.md`](reports/22_decoder_contract/report.md)**：GS 量化取位、ChaCha20 解密与 $64 \times 256$ 重复多数投票的数学边界（支撑软判决译码设计）。
- **[`reports/confirmation100_assets_v11_completed/report.md`](reports/confirmation100_assets_v11_completed/report.md)**：100 对冻结评估图像与唯一文本资产（`confirmation100 v1.1`）核验报告。

### 3.3 防御实现与评估脚本（`scripts/` & `protocols/`）
- **`scripts/defense/`**：
  - [`sembind.py`](scripts/defense/sembind.py)：SemBind 语义掩码提取、平铺置换与潜变量绑定/解绑实现。
  - [`sembind_smoke.py`](scripts/defense/sembind_smoke.py) / [`sembind_pairs.py`](scripts/defense/sembind_pairs.py) / [`sembind_pairs_aggregate.py`](scripts/defense/sembind_pairs_aggregate.py)：SemBind 单对自检与 30 对批量抗伪造评估。
  - [`sembind_algebra_test.py`](scripts/defense/sembind_algebra_test.py) / [`sembind_cpu_test.py`](scripts/defense/sembind_cpu_test.py)：SemBind 绑定/解绑代数性质与掩码截断单元测试。
  - [`repro/sembind_masker_equiv.py`](scripts/defense/repro/sembind_masker_equiv.py) / [`repro/sembind_exact_inversion.py`](scripts/defense/repro/sembind_exact_inversion.py)：SemBind 官方 masker 位级等价核验与精确反演测试。
  - [`geometry.py`](scripts/defense/geometry.py) / [`evaluate_geometry.py`](scripts/defense/evaluate_geometry.py) / [`repro/geometry_paper_task.py`](scripts/defense/repro/geometry_paper_task.py) / [`repro/geometry_lspd_sweep.py`](scripts/defense/repro/geometry_lspd_sweep.py)：Geometry 防御指标与网格正则化扫描。
  - [`seal_smoke.py`](scripts/defense/seal_smoke.py)：SEAL（ICCV 2025）语义水印生成与验证测试。
- **`scripts/confirmation100/` & `scripts/salvage30/`**：100 对/30 对标准评估集协议驱动、逐对独立密钥诊断与投票边际分析工具。
- **`protocols/`**：`confirmation100_design_v7.json` 与 `confirmation100_assets_v11.json` 冻结测试集协议。

---

## 4. 快速自检命令（纯 CPU）

```bash
# 验证 SemBind 绑定/解绑代数不变量（平铺、掩码率截断、置换对合性）
CUDA_VISIBLE_DEVICES="" python scripts/defense/sembind_algebra_test.py

# 验证 Geometry 防御指标（G-Cos / SAD / L-SPD / AUC）单元测试
CUDA_VISIBLE_DEVICES="" python -m unittest scripts/defense/test_geometry.py -v

# 验证 GS 解码与 64x256 多数投票数学契约
CUDA_VISIBLE_DEVICES="" python scripts/check_gs_decoder_contract.py
```
