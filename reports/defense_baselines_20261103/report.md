# 防御基线评估（2026-11-02 / 11-03）

本报告汇总 CVPR 2025 Imprint 基线在 2026 年三项公开防御下的评估结果，以及一项关于“投票边际”路线的负结果诊断。所有模型实验在 `liyanlei@172.18.185.249:/data0/liyanlei/watermark_research` 上完成；本文件随附脚本、协议与关键 JSON，原始 run 保留在同一路径。

## 0. 一句话

30 对、30 步的 Imprint 基线在普通 GS 上 30/30 通过（BA 0.8228，PSNR 24.04 dB）；在 PGID（151 步、1 图）与 SemBind（30 步、28/30 已完成）下大部分候选被压低；Geometry 只报 AUC（G-Cos 0.9958）。同时，无密钥“投票边际”代理被 30 对诊断证伪（参考逐坐标符号错误率 24.3%）。

## 1. GS 官方验证（30 对，30 步）

- 通过：**30/30**；平均 BA **0.8228**，中位 0.8203，IQR 0.7969–0.8477；平均 PSNR **24.04 dB**。
- 20 步：30/30，平均 BA 0.7783，PSNR 24.69 dB。
- 对照：未处理原图 **0/30**（BA 0.5078）；参考图 **30/30**（BA 0.9999）。
- 区间：Clopper–Pearson 95% [0.8843, 1.0]；Wilson [0.8865, 1.0]（n=30，二项）。
- **身份限制：30 对共享同一 GS 身份（message/key/nonce fingerprint 完全相同）**；`zT` 仅因 seed 不同而幅值不同。跨密钥矩阵退化（非对角 870/870 超阈值），因此本组只能标为“共享身份 GS 校准”，不能外推独立身份。独立性补跑需重生成参考并重跑攻击。

证据：`runs/confirmation100_baseline_20261102T204500Z/analysis_s30.json`、`candidates_sealed_30.json`、`evaluate/*`（120 张评估输出）。

## 2. Geometry（ICML 2026）

实现：`scripts/defense/geometry.py`（G-Cos/SAD，Def. 4.5；L-SPD，Def. 4.8，16×16 网格、1/M、εI、top-5）；评估：`scripts/defense/evaluate_geometry.py`；CPU 测试：`scripts/defense/test_geometry.py`。作者未公开阈值与融合规则，因此只报分数与 AUC，不自选阈值。

30 对结果（`reports/geometry_30.json`）：

| label | G-Cos 均值 | G-Cos AUC | L-SPD AUC（拒绝向） |
|---|---:|---:|---:|
| 30 | 0.179 | **0.9958** | 0.638 |
| 20 | 0.151 | 0.9925 | 0.540 |
| reference | 0.688 | 1.000 | 0.802 |
| cover | 0.002 | 0.515 | 0.486 |

## 3. PGID（NeurIPS 2026）

官方代码 `third_party/pgid`（本地快照加载修复：`run.py` 的 model choices、`pipe/pg_utils.py` 的模型表 fallback）。按官方默认：151 步攻击、单图、目标/攻击者均为 `Manojb/stable-diffusion-2-1-base` 快照；GS 阈值 0.70703125。

| 阶段 | 检测率 | BA |
|---|---:|---:|
| 水印图 | 1.0 | 1.000 |
| 攻击后（50/100/150 步） | 1.0 | 1.000 |
| **PGID 防御后（50/100/150）** | **0.0** | **0.578 / 0.621 / 0.664** |

**注意**：n=1；PGID 的模型/身份设定与我们的 SDXL 30 对不同；BA 随攻击预算单调上升，距阈值仅 4.3 个百分点——不能据此写成“PGID 完全防住”。

证据（服务器）：`runs/pgid_smoke_20261102T231752Z/attack=forgery/summary.csv`、`logs/pgid_smoke.log`。

## 4. SemBind（ICML 2026）

masker 权重公开（886,343,333 bytes），DINOv2-giant 公开；绑定流水线未发布，按论文复现（[执行版本源码](https://github.com/lylei5/watermark2/blob/ba6283f0e93f86a03f1636f5515fb91d10562bbf/scripts/defense/sembind.py)）：DINOv2 CLS → masker → 1024 bit → tile → σ=1 → 固定置换 → `S=1−2m` → `S⊙zT`；验证端用候选自身语义重建 mask 后解绑。

- 自检（单对）：SemBind-GS 绑定态 BA 0.5117、解绑后 1.000；vanilla GS 1.000 / 0.5039。
- 伪造（30 步 Imprint，单对）：对 SemBind-GS 解绑后 **0.4844**（被防住），对 vanilla GS 1.000（通过）。
- 30 对批量（30 步）：快照时 **n=28，通过 2，平均 BA 0.5379，中位 0.5059，min 0.4023，max 0.9844**。151 步对照尚未完成（0/2）。
- 偏差：DDIM 反演替代论文的 exact DPM-Solver；30 步预算为论文 150 步的 20%。

证据（服务器）：`runs/sembind_smoke_20261102T225342Z/result.json`、`runs/sembind_pairs30_*/[0-9]*_result.json`。

## 5. 负结果：无密钥“投票边际”代理不可行

GS 解码是先 ChaCha20 去白化、再按 64×256 多数投票；不知道密钥时只能用参考反演符号做模板。30 对离线诊断（仅评估端使用密钥，[执行版本源码](https://github.com/lylei5/watermark2/blob/ba6283f0e93f86a03f1636f5515fb91d10562bbf/scripts/salvage30/template_diag.py)，结果 `reports/template_diag_30.json`）：

- 参考图 BA **0.99987**，但逐坐标符号错误率 **24.27%**（最高 29.78%）。
- 无密钥模板与真实票差的符号一致率 **42.57%**；Spearman **−0.0254**。

结论：单参考、无密钥时估不出 GS 判决量；“投票边际对齐（VMA）”在当前形式下不可行。该诊断作为负结果保留，不再作为方法。

## 6. 文件与协议

- `scripts/defense/{geometry,evaluate_geometry,test_geometry,sembind,sembind_smoke,sembind_algebra_test,sembind_pairs,sembind_pairs_aggregate,seal_smoke}.py`
- `scripts/salvage30/{seal,eval,analyze,template_diag}.py`
- `protocols/confirmation100_{design,attacker,evaluator,controller}_v7.json` 及 `.sha256`
- `reports/geometry_30.json`、`reports/template_diag_30.json`
- `runs/confirmation100_baseline_20261102T204500Z/`（1351 文件，约 158.9 MB）

## 7. 局限

- 共享身份（见 §1），不能外推独立密钥。
- PGID 为 n=1 且模型设定不同（Manojb SD2.1 vs 我们的 SDXL 目标）。
- SemBind 的绑定是我们按论文复现，非作者完整流水线；30 步预算，151 步对照未完成。
- Geometry 阈值/融合未公开，只报 AUC，不构成接受/拒绝。
- 无独立人工内容评审；所有模型辅助视觉检查不等同人工标签。
