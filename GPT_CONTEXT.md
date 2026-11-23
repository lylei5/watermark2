# 研究交接

从[研究方案](docs/RESEARCH_PLAN.md)读取来源验证假设，从[机器状态](project_state.json)读取执行状态。最新主线进展见[Commit-then-Steer Stage 0–3 完整实验报告](reports/commit_steer_20261123/report.md)（含纯 RGB/通道比例/空间统计证伪、100 条官方统一集原生 GS-256b + COCO FID、2,000 张真实 COCO 自然图经验 Null FPR、实时 151 步 Imprint 与 108 候选 Reprompt+、跨载体 16 条统一集原生 512-bit PRC、多维 64-bit 融合、窗口消融及白盒自适应 PGD 压力测试，全部独立审计 0 mismatch）。

此前历史阶段：[等预算图像身份验证](reports/color_budget_20261119/report.md)给出单项特征结果，[颜色与布局分别计分的独立确认](reports/color_joint_20261119/report.md)与[逐例诊断](reports/color_joint_diagnosis_20261119/report.md)记录冻结门限的颜色/布局表现。

基线规格见[实现差异表](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)。引用数字时回到[证据索引](reports/README.md)里的对应报告，核对载体、分母、工作点以及来源是否独立。
