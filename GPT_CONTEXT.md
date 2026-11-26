# 研究交接

从[研究方案](docs/RESEARCH_PLAN.md)读取来源验证假设，从[机器状态](project_state.json)读取执行状态。最新主线进展见[Commit-then-Steer Stage 4 完整实验报告（SDXL 跨骨干 + 100 图多维融合 + 同协议基线正面对比）](reports/commit_steer_stage4_20261125/report.md)与[Stage 0–3 完整实验报告](reports/commit_steer_20261123/report.md)。Stage 4 全部四组实验及离线 COCO-50 FID 已收官并通过独立 0-mismatch 审计（`total_mismatches = 0`）：
- **Stage 4A（SDXL `1024×1024`）**：24 统一集 Clean `24/24`、9-GS `184/216 (85.2%)`、`LPIPS=0.0705`、Live 36 候选 Reprompt+ `0/36`（原生 `36/36`）、200 COCO Null `0/200`。
- **Stage 4B（100 图多维融合 `hyb_sscd32_clip32`）**：Clean `96/100`、9-GS `613/900 (68.1%)`、`LPIPS=0.096`、离线 COCO-50 FID `227.18→230.64`（配对漂移 `43.08`）、2,000 COCO Null `0/2,000`（特征项亦 `0/2,000`）、Live 151 步 Imprint `0/4`（vs `4/4`）、Live 108 候选 Reprompt+ SD2.1 `1/54` & SD1.5 `1/54`（vs `108/108`）。
- **Stage 4C（Native GS & 官方 PGID-F）**：Matched-30 下 PGID-F 验证 `21.59s`（5.1× 慢）、9-GS `175/270 (64.8%)`（低于 `sscd_64b` `219/270 = 81.1%` 与 `hyb` `187/270 = 69.3%`）、Null200 `0/200`、Live Imprint step150 `2/4` 失守、Live Reprompt+ `104/108 (96.3%)` 全线失守；Commit-then-Steer 同队列全面胜出。
- **Stage 4D（论文级 SemBind GS-S，Matched-20）**：生成 `22.09s`（3.0× 慢）、验证 `49.51s`（11.7× 慢）、`LPIPS=0.723`、`CLIP=0.2679`、Clean `20/20`、9-GS `169/180 (93.9%)`、Null50 `0/50`、Imprint `0/4`、Reprompt+ `0/72`（`>181/256`；`≥167/256` 下 Imprint50 `1/4`、Reprompt+ `1/36` & `2/36`）；与 Commit-then-Steer 呈“免训练/11.7× 速度/保实例/跨 SDXL-PRC vs 训练哈希重度遮挡存活率”的 Pareto 权衡，不主张单边全面优于 SemBind。

此前历史阶段：[等预算图像身份验证](reports/color_budget_20261119/report.md)给出单项特征结果，[颜色与布局分别计分的独立确认](reports/color_joint_20261119/report.md)与[逐例诊断](reports/color_joint_diagnosis_20261119/report.md)记录冻结门限的颜色/布局表现。

基线规格见[实现差异表](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)。引用数字时回到[证据索引](reports/README.md)里的对应报告，核对载体、分母、工作点以及来源是否独立。
