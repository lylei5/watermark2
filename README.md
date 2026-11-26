# 扩散图像水印的来源验证

研究问题是，图像经过正常编辑后，验证器还能否认出原发行方，同时拒绝从其他图像迁移来的水印。

最新主线进展见 **[Commit-then-Steer Stage 4 跨骨干与同协议基线正面对比完整报告](reports/commit_steer_stage4_20261125/report.md)**（全量独立审计 `total_mismatches = 0`）与 **[Stage 0–3 完整实验报告](reports/commit_steer_20261123/report.md)**：
- **Stage 4A（SDXL Base 1.0 `1024×1024` 跨骨干）**：零超参调优、零阈值重调直接迁移至 24 条不重叠统一集原生 SDXL GS-256b（`4×128×128`），实现 Clean **`24/24 (100%)`**（BER `0.00%`）、`LPIPS` 均值 **`0.0705`**、`CLIP = 0.2819`（无引导 `0.2827`）、9 项标准 GS 失真 **`184/216 (85.2%)`**（7 项常规 **`93.5%`**）、200 张真实 COCO 自然图经验误报 **`0/200`**、Live 36 候选 SDXL Reprompt+ **`0/36`**（原生 GS-SDXL `36/36` 全破）。
- **Stage 4B（100 图多维融合 `hyb_sscd32_clip32`）**：同 100 条统一集 Clean `96/100`、9 项失真 `613/900 (68.1%)`、`LPIPS` 降至 **`0.096`**（降 25%）、`CLIP = 0.2739`（无引导 `0.2724`）、离线 COCO-50 FID `227.18→230.64`（配对漂移由 `55.75` 降至 **`43.08`**）、2,000 张真实 COCO 自然图默认与特征项均 **`0/2,000`** 误报（1% auth 切点压制 `22/1,000 -> 0/1,000`）、Live 151 步 Imprint **`0/4`**（全 12 检查点 0 通过 vs 原生 `4/4`）、Live 108 候选 Reprompt+ SD2.1 **`1/54`** & SD1.5 **`1/54`**（原生 `108/108`）。
- **Stage 4C & 4D（同协议基线正面对比：Native GS / 官方 PGID-F / 论文级 SemBind GS-S）**：
  - 对比 **官方 PGID-F（`PGID_dc4ad96`，Matched-30）**：Commit-then-Steer `sscd_64b` 验证快 **5.1×**（`4.22s` vs `21.59s`）、9 项 GS 失真高 **+16.3 pp**（`219/270 = 81.1%` vs `175/270 = 64.8%`）、Live 151 步 Imprint **`0/4` vs `2/4`**、Live 108 候选 Reprompt+ **`2/108 (1.9%)` vs `104/108 (96.3%, 12/12 源代理全破)`**（PGID-F 200 自然图 `0/200`）。
  - 对比 **论文级 SemBind GS-S（Matched-20）**：两者呈清晰 Pareto 权衡——Commit-then-Steer 具备 **Training-Free**、生成快 **3.0×**（`7.35s` vs `22.09s`）、验证快 **11.7×**（`4.22s` vs `49.51s`）、保实例低漂移（`LPIPS 0.093–0.127` vs `0.723`）与跨 SDXL/PRC 零调参迁移优势；SemBind GS-S 凭借训练 DINOv2-Giant 哈希与 100 步 VAE 优化在重度遮挡失真（9-GS `169/180 = 93.9%` vs `147/180 = 81.7%`）及默认阈值下 72 候选 Reprompt+（`0/72`，50 自然图 `0/50`，Imprint `0/4`）上占优。
- **Stage 0–3 基石（SD 2.1 `512×512`）**：在证伪纯 RGB 均值/方差/通道比例（Imprint `24/30–29/30`、Reprompt+ `10/10`）后，100 条官方未见统一集原生 GS-256b 上实现 Clean 盲验 `99/100`（BER `0.00%`）、9 项 GS 失真 `694/900 (77.1%)`、COCO-50 FID `-4.54`，2,000 张真实 COCO 自然图经验误报 `0/2,000`，实时 151 步 Imprint `0/4`（原生 `4/4`）、Live Reprompt+ SD2.1 `0/54` / SD1.5 `2/54`（原生 `108/108`），并无缝兼容原生 512-bit PRC（Clean `15/16`、Live Reprompt `1/8` vs `8/8`）。

此前图像身份验证的[等存储预算实验](reports/color_budget_20261119/report.md)比较了全局色度、亮度分区与明暗布局，[独立确认](reports/color_joint_20261119/report.md)中颜色与布局分别计分在强对比度下通过 118/120，误收同提示词其他种子 4/120。

| 要了解的内容 | 入口 |
|---|---|
| 方法假设、已有先例与证伪条件 | [研究方案](docs/RESEARCH_PLAN.md) |
| 防御基线的实现范围 | [SemBind、PGID、SEAL](docs/THREE_BASELINE_REPRODUCTION.md) · [实现差异](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md) |
| 可采用的公开评测 | [论文、作者版本与适用条件](docs/UNIFIED_BENCHMARK_APPLICABILITY.md) |
| 已有结果与逐样本证据 | [报告索引](reports/README.md) |
| 当前任务、算力与运行状态 | [机器状态](project_state.json) · [项目规则](AGENTS.md) |

已有证据中，SEAL 完成了独立自然图校准与验证；Stage 4C/4D 已完成 Native GS、官方 PGID-F（Matched-30）及论文级 SemBind GS-S（Matched-20）同协议正面对比（官方组件与按论文补齐的绑定部分分别记录）。Commit-then-Steer 在同协议下全面优于 Native GS 与官方 PGID-F，并与 SemBind GS-S 呈速度/免训练/保实例 vs 重度遮挡鲁棒性的 Pareto 权衡，项目不主张在所有维度单边优于 SemBind。[Stage 4 同协议对比](reports/commit_steer_stage4_20261125/report.md) · [SEAL 校准结果](reports/unified_seal_results_20261116/report.md) · [GS-S 历史链路验证](reports/phase1_trajectory_native_gs_20261113/report.md)

实验代码在 `scripts/`，冻结协议在 `protocols/`。模型、密钥、原始图像及完整运行目录保存在 249 的 `/data0/liyanlei/watermark_research`。
