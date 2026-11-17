# 扩散图像水印的来源验证

研究问题是，图像经过正常编辑后，验证器还能否认出原发行方，同时拒绝从其他图像迁移来的水印。当前阶段为文献调研与防御方法设计，实验队列为空。

| 要了解的内容 | 入口 |
|---|---|
| 方法假设、已有先例与证伪条件 | [研究方案](docs/RESEARCH_PLAN.md) |
| 防御基线的实现范围 | [SemBind、PGID、SEAL](docs/THREE_BASELINE_REPRODUCTION.md) · [实现差异](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md) |
| 可采用的公开评测 | [论文、作者版本与适用条件](docs/UNIFIED_BENCHMARK_APPLICABILITY.md) |
| 已有结果与逐样本证据 | [报告索引](reports/README.md) |
| 当前任务、算力与运行状态 | [机器状态](project_state.json) · [项目规则](AGENTS.md) |

已有证据中，SEAL 完成了独立自然图校准与验证。SemBind 在项目的 GS-S 链路上完成了正常图验证，官方组件和按论文补齐的绑定部分已有分别记录。项目尚未建立在共同合法检出、误报和攻击预算下优于 SemBind 的结论。[SEAL 校准结果](reports/unified_seal_results_20261116/report.md) · [GS-S 验证](reports/phase1_trajectory_native_gs_20261113/report.md)

实验代码在 `scripts/`，冻结协议在 `protocols/`。模型、密钥、原始图像及完整运行目录保存在 249 的 `/data0/liyanlei/watermark_research`。
