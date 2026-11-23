# 扩散图像水印的来源验证

研究问题是，图像经过正常编辑后，验证器还能否认出原发行方，同时拒绝从其他图像迁移来的水印。

最新主线进展见 **[Commit-then-Steer 无库内容绑定完整实验报告（Stage 0–3）](reports/commit_steer_20261123/report.md)**：在证伪纯 RGB 均值/方差/通道比例（Imprint `24/30–29/30`、Reprompt+ `10/10`）后，通过“12 步预览前向承诺 64-bit 密钥化语义/多维指纹 + 原生水印载荷无改动内嵌 + 第 5–35 步中步流形引导（末尾 14 步自然去噪）”，在 100 条官方未见统一集原生 GS-256b 上实现 Clean 盲验 `99/100`（BER `0.00%`）、9 项 GS 失真 `694/900 (77.1%)`、COCO-50 FID `-4.54`，2,000 张真实 COCO 自然图经验误报 `0/2,000`，实时 151 步 Imprint `0/4`（原生 `4/4`）、Live Reprompt+ SD2.1 `0/54` / SD1.5 `2/54`（原生 `108/108`），并无缝兼容原生 512-bit PRC（16 条统一 prompt Clean `15/16`、Live Reprompt `1/8` vs `8/8`）及多维融合 `32b SSCD + 32b CLIP`（Live Reprompt `0/8`）。

此前图像身份验证的[等存储预算实验](reports/color_budget_20261119/report.md)比较了全局色度、亮度分区与明暗布局，[独立确认](reports/color_joint_20261119/report.md)中颜色与布局分别计分在强对比度下通过 118/120，误收同提示词其他种子 4/120。

| 要了解的内容 | 入口 |
|---|---|
| 方法假设、已有先例与证伪条件 | [研究方案](docs/RESEARCH_PLAN.md) |
| 防御基线的实现范围 | [SemBind、PGID、SEAL](docs/THREE_BASELINE_REPRODUCTION.md) · [实现差异](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md) |
| 可采用的公开评测 | [论文、作者版本与适用条件](docs/UNIFIED_BENCHMARK_APPLICABILITY.md) |
| 已有结果与逐样本证据 | [报告索引](reports/README.md) |
| 当前任务、算力与运行状态 | [机器状态](project_state.json) · [项目规则](AGENTS.md) |

已有证据中，SEAL 完成了独立自然图校准与验证。SemBind 在项目的 GS-S 链路上完成了正常图验证，官方组件和按论文补齐的绑定部分已有分别记录。项目尚未建立在共同合法检出、误报和攻击预算下优于 SemBind 的结论。[SEAL 校准结果](reports/unified_seal_results_20261116/report.md) · [GS-S 验证](reports/phase1_trajectory_native_gs_20261113/report.md)

实验代码在 `scripts/`，冻结协议在 `protocols/`。模型、密钥、原始图像及完整运行目录保存在 249 的 `/data0/liyanlei/watermark_research`。
