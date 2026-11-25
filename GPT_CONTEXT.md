# 研究交接

从[研究方案](docs/RESEARCH_PLAN.md)读取来源验证假设，从[机器状态](project_state.json)读取执行状态。最新主线进展见[Commit-then-Steer Stage 4 报告（SDXL 跨骨干 + 100 图多维融合 + 同协议基线对比）](reports/commit_steer_stage4_20261125/report.md)与[Stage 0–3 完整实验报告](reports/commit_steer_20261123/report.md)。Stage 4A（SDXL `1024×1024` 24 统一集 Clean `24/24`、9-GS `184/216 (85.2%)`、`LPIPS=0.0705`、Live 36 候选 Reprompt+ `0/36` vs 原生 `36/36`、200 COCO Null `0/200`）与 Stage 4B 100 图多维融合 + 2,000 自然图（`0/2,000`）均已独立审计 `0 mismatch`；Stage 4B 实时攻击及 Stage 4C/4D（官方 PGID-F、论文级 SemBind GS-S）正由 `249` 上内核 `pidfd` 协调器在 GPU 0/1/2 并行执行。

此前历史阶段：[等预算图像身份验证](reports/color_budget_20261119/report.md)给出单项特征结果，[颜色与布局分别计分的独立确认](reports/color_joint_20261119/report.md)与[逐例诊断](reports/color_joint_diagnosis_20261119/report.md)记录冻结门限的颜色/布局表现。

基线规格见[实现差异表](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)。引用数字时回到[证据索引](reports/README.md)里的对应报告，核对载体、分母、工作点以及来源是否独立。
