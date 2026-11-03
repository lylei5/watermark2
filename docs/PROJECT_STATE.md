# 阶段 01–05 实验谱系

本页保留截至 2026-10-13 UTC 的阶段 01–05 实验事实。当前研究状态与权限见 [研究计划](RESEARCH_PLAN.md) 和 [project_state.json](../project_state.json)。

研究主线是扩散模型语义水印的黑盒伪造攻击与错误来源归属。几何审计用于明确评估对象；目前保留官方 Reprompting 基线和独立全局余弦诊断。

## 已实际完成

| 阶段 | 工作 | 模型正式运行 | 核心证据 |
|---|---|---:|---|
| 01 | 环境、官方源码、PSNR 和参数审计 | 0 | [环境](../reports/01_environment.md)、[基线审计](../reports/01_baseline_audit.md) |
| 02 | 独立环境、固定 snapshot、GS Reprompting seed=1 | 1 | [报告](../reports/02_smoke_report.md)、[run](../runs/02_reprompt_gs_sdxl_seed1_20261012T172624Z/) |
| 03 | 几何论文 v1、公开代码与信息边界静态核查 | 0 | [审计](../reports/03_geometry_audit_20261013T020025Z/03_audit.md)、[边界](../reports/03_geometry_audit_20261013T020025Z/information_boundary.md) |
| 04 | 新运行 seed=1，真实目标潜变量留存，CPU 余弦 | 1 | [报告](../reports/04_latent_diag_20261013T025322Z/04_report.md)、[run](../runs/04_reprompt_latent_diag_seed1_20261013T025322Z/) |
| 05 | 固定提示词、seeds 2–9，每种子一次，串行 | 8 | [报告](../reports/05_reprompt_seed_pilot_20261013T041832Z/05_report.md)、[run](../runs/05_reprompt_seed_pilot_20261013T041832Z/) |

因此有 10 次正式模型进程运行（阶段 02 的 1 次、阶段 04 的 1 次、阶段 05 的 8 次）；阶段 02 和 04 的 seed=1 输出图像哈希相同，不能作为两个独立样本计入阶段 05。阶段 05 统计仅使用其八个预定种子。

## 工程状态的时间顺序

- 阶段 01：服务器 Git TLS 握手失败，官方代码经本机校验后中转；初始依赖和缓存不足。它们是当时状态，不应读成阶段 05 当前仍缺依赖。
- 阶段 02：初次准备使用系统 Python 3.6，因 shlex.join 不可用而停止；另有小张量退出码文件换行处理问题。均发生在官方模型入口启动前。失败目录和日志保留。最终使用独立 Python 3.10、GPU 3 完成一次正式样例。服务器缓存核验后删除本机约 17.7283 GiB 权重。
- 阶段 03：第一次浅 bundle 缺父提交，之后完整 Git 对象中转完成。几何仓库静态缺项保留为阻塞，没有运行该入口。
- 阶段 04：[首次 GPU 7 预检报告](../reports/04_latent_diag_20261013T024515Z/04_report.md) 记录卡忙，formal_model_runs_started=0；用户指定 GPU 3 后有一个新的完成 run。两份报告都保留。
- 阶段 05：8/8 完成，工程失败 0/8，原始图像和原始噪声哈希未发现跨种子重复，墙钟 619.7655127081089 秒。

## 固定条件

目标 `stabilityai/stable-diffusion-xl-base-1.0`，revision `462165984030d82259a11f4367a4eed129e94a7b`。
代理 `Manojb/stable-diffusion-2-1-base`，revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`。

GS，512×512，float32；目标/代理生成各 50 步、guidance=7.5，DDIM；反演空提示词、guidance=1.0；resample=false。GS: l=1、message_width_in_bytes=32、num_replications=64、offset=0；严格 bit_accuracy > 0.70703125。

参考提示词：`A studio photograph of a yellow ceramic teapot on a wooden table.`

攻击提示词：`A red sailboat on a calm blue lake, watercolor illustration.`

实际环境完整版本见 [pip_freeze.txt](../runs/02_reprompt_gs_sdxl_seed1_20261012T172624Z/pip_freeze.txt)。原始上游 commit 为 `ca68950cd7b53b8438a49580f134b2a0db28d778`。完整冻结条件见 [pilot_protocol.json](../runs/05_reprompt_seed_pilot_20261013T041832Z/pilot_protocol.json)。

## 观察与结论边界

阶段 02/04：参考 BA=1.0、攻击 BA=0.953125。阶段 04 cosine 为 0.8201082797320761 / 0.3506198907446562，gap=0.4694883889874199。

阶段 05：参考 GS 通过 8/8；参考有效配对中的攻击 GS 通过 8/8；gap>0 为 8/8。余弦均值分别 0.8138452617511447 / 0.3479337535299411，gap 均值 0.46591150822120353。

独立余弦使用真实目标原始噪声与目标反演噪声，batch=1、展平全部 C/H/W、CPU float64、无中心化/epsilon，原始 float32 NPY 保留。它尚未被证明与缺失的作者函数逐项数值等价。完整几何规则、阈值、融合、L-SPD 尚未复现；geometry_threshold=null、geometry_accepted=null、full_official_geometry_evaluation_completed=false。

这批数据是 development_pilot，固定提示词、模型和基础消息/密钥设置。不同种子和不同哈希不证明统计独立。bit_accuracy 是单图位匹配比例；上述 8/8 不是总体可靠攻击成功率。没有 ROC/AUC、误报率、显著性或跨提示词/跨密钥泛化结论。PSNR/SSIM/MS-SSIM/LPIPS 在这些 Reprompting 运行中未计算，原 CSV 的 -1 是占位。

## 仍需明确的材料

阶段 03 审计的几何 commit `72880c9bb1cee0b04bcb2433deb2ab2cc95ff622` 缺 pipe_utils、finger_utils、spd_utils；几何代码块被注释，活动判定只有原水印。候选如何找到目标原始参考、实际数值规则、固定部署阈值与校准协议仍未知。完整结论应以该固定版本为界，不能扩大成对未来上游的判断。

阶段 04/05 的实际潜变量已保存；这没有恢复阶段 02 当时未保存的张量。阶段 02 与阶段 04 图像相同也不能把阶段 04 张量重标为阶段 02 原始记录。

论文 C.3 讨论 Reforge；讨论、公开算法和进入比较的实验应区分。不能说作者完全没有考虑自适应攻击，也不能从公开代码缺项推出已证实漏洞。

完整官方几何评估依赖对应版本的完整实现、参考关联和固定判定协议。后续进展见 [当前研究审查](../reports/research_review.md)。


## 防御基线（2026-11-03 UTC）

已完成 30 对 Imprint 基线在普通 GS、Geometry、PGID、SemBind 下的评估：普通 GS 30/30（BA 0.8228，PSNR 24.04 dB），Geometry G-Cos AUC 0.9958，PGID 151 步防御后 BA 0.578/0.621/0.664（1 图），SemBind 单对 0.484、30 对批量 28 例中 2 例通过（平均 0.5379）。同时，无密钥“投票边际”代理被 30 对诊断证伪（参考逐坐标符号错误 24.3%，模板符号一致率 42.6%，Spearman −0.025），不再作为方法。

30 对共享同一 GS 身份；PGID 为 n=1 且模型设定不同；SemBind 绑定为论文复现；151 步对照与独立身份补跑未完成。详见 [防御基线报告](../reports/defense_baselines_20261103/report.md)。
