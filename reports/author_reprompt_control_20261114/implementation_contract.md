# Reprompt 实现与评测范围

攻击流程采用 Müller 等人的 CVPR2025 原作者实现。固定版本为 `ca68950cd7b53b8438a49580f134b2a0db28d778`，执行前后检查 Git HEAD、工作区和 Python 源文件哈希。实验仅在249执行，代理模型分别使用GPU7和GPU5。

| 项目 | 本轮设置 |
|---|---|
| 输入 | 已冻结的GS、TR、PRC水印图，各10张；与完整Imprint及合法图对照来源一致 |
| 代理 | SD2.1、SD1.5，使用已验证的本地模型快照 |
| 反演 | 作者 `SDPipeProvider.invert_images`，作者DDIM逆调度器，float32，50步，空提示词，CFG1 |
| 生成 | 作者 `SDPipeProvider.generate`，作者DDIM正向调度器，float32，50步，CFG7.5 |
| 候选预算 | 每来源、每代理1张；共60张；无重采样、无检测反馈、无候选筛选 |
| 目标提示词 | 既有审核后的I2P子集；每个来源编号使用同一目标提示词，跨载体和代理保持一致 |
| 数据适配 | 以公开PNG替代作者脚本第一阶段的现场生成结果；直接实例化作者SD类以支持SD1.5本地路径 |
| 私有材料 | 攻击进程的输入没有载体密钥、恢复helper或目标原生反演；共享Unix用户，属于逻辑隔离 |
| 后续验证 | 原生载体与PGID各自执行其原有反演；攻击代理的DDIM与PRC原生DPM分开 |

本轮完整执行每个候选的攻击步骤，样本范围是配对开发队列。原论文的数值矩阵仍是另一个完成条件：

| 原文设置 | 来源与预算 | 本轮覆盖 |
|---|---|---|
| CVPR2025 basic Reprompt | 主实验每个目标模型1000个来源；目标模型包含SDXL、PixArtΣ、FLUX和SD2.1-Anime | 作者攻击函数与完整步数；目标载体采用本项目SD2.1队列 |
| CVPR2025 Reprompt+ | 每目标100个来源；3个提示词；GS再做3次重采样，按多候选成功统计 | 尚未执行；不得把单候选结果标为Reprompt+ |
| SemBind主表Reprompt | COCO、SDP各100来源；4种载体及各自SemBind版本；SD2.1和SD1.5代理 | 两种代理、GS/TR/PRC各10个配对来源；完整表格尚未完成 |

过滤I2P子集与原论文目标分布不同。本文报告“生成候选的水印接受率”，尚未把提示词遵循情况或有害内容成功率作为计数条件。所有候选都保留；合法图失败也保留在绝对检出率的分母中。

参考：[CVPR2025论文记录](https://www.openaccess.thecvf.com/content/CVPR2025/html/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.html)、[作者全文与附录](https://arxiv.org/html/2412.03283v3)、[SemBind第4.3节](https://arxiv.org/html/2601.20310#S4.SS3)、[SemBind会议记录](https://proceedings.mlr.press/v306/zhang26hj.html)。
