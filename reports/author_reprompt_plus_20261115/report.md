# 作者 Reprompt+ 固定候选预算

249 已生成并核验 240 张 Reprompt+ 候选。GS 每个来源、每个代理使用原论文的 3 个提示词 × 3 次原始重采样，共 9 个候选；TR 每个来源、每个代理使用 3 个提示词，共 3 个候选。GS/TR 各 10 个来源，SD2.1/SD1.5 两个代理。全部候选保留，未依据检测反馈增加尝试。

实现调用作者固定提交 `ca68950cd7b53b8438a49580f134b2a0db28d778` 的 `SDPipeProvider` 和 GS `wiggle_latents`。反演复用前次对同一公开 PNG 执行的 50 步 DDIM 结果，输入、像素及 latent 均核验哈希；每次生成使用 float32、50 步 DDIM、CFG7.5。GS 重采样仅传入公开方法参数、dtype 和 device，未读私有验证密钥。种子与候选循环顺序冻结在 [protocol.json](protocol.json)。

## 完成核验

| 项目 | 结果 |
|---|---:|
| SD2.1 候选 | 120 |
| SD1.5 候选 | 120 |
| 图像文件、RGB 像素及 latent 哈希检查 | 240 |
| 作者 Python 文件哈希检查 | 26 |
| 作者工作区 | 干净 |
| TR 首提示词与前次 basic Reprompt 像素一致 | 20/20 |

来源见 [artifact_audit.json](artifact_audit.json)、两代理的 [sd21_complete.json](sd21_complete.json)、[sd15_complete.json](sd15_complete.json) 以及逐候选 rows。原论文来源为 Müller 等的 [CVPR2025 论文](https://arxiv.org/html/2412.03283v3)及作者代码。

本批覆盖每个来源的完整候选预算。原论文还使用更大来源数及多个目标模型；本批目标为既有 SD2.1 水印图，提示词来自已审阅 I2P 子集，SD1.5 是补充代理。没有给 PRC 自行扩展 Reprompt+。尚未标注目标提示词实现程度，后续统计名称为“候选/来源水印接受”。

原生/PGID 和内容强对照在独立验证进程中执行。每张候选须同时通过内容与载体检测，再在同一来源的固定候选集合内取“任一成功”；不把不同候选的两个成功相拼接。当前执行状态见 [project_state.json](../../project_state.json)，权威检测运行目录为 `reprompt_plus_evaluation_20261115_v2`。
