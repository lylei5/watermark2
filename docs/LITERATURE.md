# 扩散模型水印防御文献

核验日期：2026-11-12。论文题名、编号和会议状态以作者原始记录及会议论文集为依据；论文发表、源码可用、项目组件测试和完整方法复现分别记录。方法是否优于本项目必须由同条件实验判断。

## 主要防御与评测来源

| 工作 | 正式题名 | 论文编号与发表来源 | 在本项目中的用途 |
|---|---|---|---|
| SemBind | SemBind: Binding Diffusion Watermarks to Semantics Against Black-Box Forgery Attacks | [arXiv:2601.20310](https://arxiv.org/abs/2601.20310)；[ICML2026，PMLR306:159761–159784](https://proceedings.mlr.press/v306/zhang26hj.html) | 语义绑定防御基线；官方语义组件与项目生成适配器分开评估 |
| PGID | PGID: Progressive Guided Inversion and Denoising for Robust Watermark Detection | [arXiv:2605.09319](https://arxiv.org/abs/2605.09319)；作者v5记录注明NeurIPS2026接收 | 检测端渐进引导反演与去噪；完整比较尚未建立 |
| Geometry | Rethinking Forgery Attacks on Semantic Watermarks in Black-Box Settings: A Geometric Distortion Perspective | [arXiv:2606.29807](https://arxiv.org/abs/2606.29807)；[ICML2026，PMLR306:65253–65276](https://proceedings.mlr.press/v306/lee26bm.html) | 潜变量几何失真判别；公开材料的判定缺项需单独说明 |
| SEAL | SEAL: Semantic Aware Image Watermarking | [arXiv:2503.12172](https://arxiv.org/abs/2503.12172)；[ICCV2025，16196–16205](https://openaccess.thecvf.com/content/ICCV2025/html/Arabi_SEAL_Semantic_Aware_Image_Watermarking_ICCV_2025_paper.html) | 语义嵌入、局部敏感哈希与内容相关噪声；属于原生生成方案 |
| PAI | Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection | [arXiv:2601.06639](https://arxiv.org/abs/2601.06639)；[ICLR2026官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/51e2aaa731e8a5edfbda95fd8bd8948b-Abstract-Conference.html) | 密钥条件化去噪轨迹偏转，服务于所有权验证和篡改定位 |
| semantic-forgery | Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models | [arXiv:2412.03283](https://arxiv.org/abs/2412.03283)；[CVPR2025，20937–20946](https://www.openaccess.thecvf.com/content/CVPR2025/html/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.html) | 已发表Imprint、Reprompt基线，仅用于防御评测 |
| ISTS | Towards Robust Content Watermarking Against Removal and Forgery Attacks | [arXiv:2604.06662](https://arxiv.org/abs/2604.06662)；[CVPR2026 Findings，8059–8069](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Towards_Robust_Content_Watermarking_Against_Removal_and_Forgery_Attacks_CVPRF_2026_paper.html) | Instance-Specific watermarking with Two-Sided detection；提示词语义控制注入时间和图样 |
| Gaussian Shading | Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models | [arXiv:2404.04956](https://arxiv.org/abs/2404.04956)；[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.html) | 分布保持潜变量载体和多数译码；官方失真组件用于评测 |
| Tree-Ring | Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust | [arXiv:2305.20030](https://arxiv.org/abs/2305.20030) | 潜变量频域水印参考；本项目当前结果不构成跨载体验证 |

主要作者记录：SemBind为Xin Zhang、Zijin Yang、Kejiang Chen、Linfeng Ma、Weiming Zhang、Nenghai Yu；PGID为Minh Quoc Duong、Chun Tong Lei、Chun Pong Lau；Geometry为Cheng-Yi Lee、Yichi Zhang、Yuchen Yang、Chun-Shien Lu、Jun-Cheng Chen；SEAL为Kasra Arabi、R. Teal Witter、Chinmay Hegde、Niv Cohen；PAI为Qingyu Liu、Yitao Zhang、Zhongjie Ba、Chao Shuai、Peng Cheng、Tianhang Zheng、Zhibo Wang；semantic-forgery为Andreas Müller、Denis Lukovnikov、Jonas Thietke、Asja Fischer、Erwin Quiring。

## 其他相关防御

| 工作 | 正式题名与原始记录 | 核验状态与适用边界 |
|---|---|---|
| Sign Clamping | [Attack-Resilient Multi-Bit Image Watermarking Framework via Latent Coordinate Sign Clamping，2610.40204](https://arxiv.org/abs/2610.40204) | 预印本；面向任意图像的多比特嵌入，含加权译码、纠错及内容绑定。不能描述成仅裁剪GS解码幅值的规则。 |
| CoverLock | [Residual Transferability in Neural Image Watermarking，2609.32241](https://arxiv.org/abs/2609.32241) | 预印本，CoverLock是文中方法名；本项目辅助图GS适配不等于原生后处理系统。 |
| PP-Mark | [PP-Mark: Provable and Publicly Verifiable Watermarking for Generative AI，2610.41170](https://arxiv.org/abs/2610.41170) | 作者记录注明NeurIPS2026接收；包含嵌入过程的零知识证明。本项目没有复现其完整验证。 |
| Gaussian Shading++ | [Gaussian Shading++: Rethinking the Realistic Deployment Challenge of Performance-Lossless Image Watermark for Diffusion Models，2504.15026](https://arxiv.org/abs/2504.15026) | 已核对arXiv题名与摘要；双通道、软译码和公钥签名，不能只凭摘要推断其失效条件。 |
| SLICE | [SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking，2603.12749](https://arxiv.org/abs/2603.12749) | [BMVC2026官方日程](https://bmvc2026.bmva.org/programme/tuesday/)列出论文；日程收录不等于本项目已完成实现审计。 |

## 与当前防御假设有关的补充核查

- [Heteroscedastic watermark detector in the contourlet domain](https://doi.org/10.1049/iet-cvi.2018.5254)，Maryam Amirmazlaghani，IET Computer Vision 13(3):249–260，2019。条件异方差统计检测已有先例，不能将非均匀信道或软加权单独作为创新。
- [IRIS: Visual-Semantic Binding for Forgery-Resistant Watermarking of Diffusion Images](https://arxiv.org/abs/2608.03539)：v1发表于2026-08-04；当前v2记录注明2026-08-26撤回、Substantial revisions。v1涉及最终图内容绑定与末期轨迹注入。仅作为相关公开披露及查新边界，不当作经确认的有效防御结果或攻击基线。

## 本项目的证据边界

- 官方masker在有限样本上的逐位一致，只核验该组件；SemBind项目适配器不应标为完整原生复现。见[组件核验](../reports/repro_sembind_masker_20261103/report.md)、[当前完整预算实验](../reports/full_imprint_20261111/report.md)。
- PGID的0.5781、0.6211、0.6641来自[历史n=1试验](../reports/repro_pgid_20261103/report.md)，三个值均未跨过默认门限。另已核验n=5运行，三个检查点均0/5接受，平均BA为0.5625、0.59921875、0.665625；它们不证明该防御被攻破。
- Geometry的L-SPD数值属于[项目公式重实现](../reports/repro_geometry_paper_task_20261103/report.md)。映射和判定规则未完整确定时，不据此归因作者方法存在数值缺陷；AUC相同也不证明实现相同。
- SEAL以[原生评估契约](SEAL_EVALUATION_CONTRACT.md)为准；PAI以[轨迹和参考来源契约](PAI_EVALUATION_CONTRACT.md)为准。源码阅读不等于完成模型实验。
- 跨图语义码相近是描述性观察，需要内容标注才能判断是否违背语义绑定目标；不能直接称为密码学碰撞或已验证绕过。
- 默认规则是`BA > 181/256`，最小通过整数为182位。独立公平比特模型下单次尾概率约5.6348e-12；这不是实测FPR，也不能无条件代替多用户总体误报率。

最新新执行证据：[PGID官方九条件](../reports/pgid_published_revalidation_20261112/report.md)标准90/90、PGID后61/90；[SemBind/GS完整链路](../reports/sembind_chain_revalidation_20261111/report.md)18条干净图记录全部接受，均通过独立潜变量复核。机制研究与创新边界见[防御机制决策](DEFENSE_MECHANISM_DECISION_20261112.md)。

当前实验结果见[研究状态](../GPT_CONTEXT.md)与[基线证据清单](DEFENSE_BASELINE_INVENTORY.md)。只使用已发表顶会顶刊基线的既定实现和完整预算，不创造或改进攻击。

## 内容耦合载体查新（2026-11-12）

- [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/html/2404.04956v1)，2404.04956，CVPR2024。§3.2已有等概率区间编码、逆CDF采样与重复扩展；不能将高斯分位调制或恢复重复容量本身作为创新。
- [ShapeMark: Robust and Diversity-Preserving Watermarking for Diffusion Models](https://arxiv.org/html/2603.09454v1)，2603.09454。已核验原文题名与结构编码/块重排机制；此条仅引用预印本，不从模板关键词推断会议录用。
- [Correlation-Aware and Gaussianity-Preserving Robust Latent Angular Watermarking for Diffusion Models](https://arxiv.org/html/2607.22386v1)，2607.22386。已核验题名与成对角度载体、幅值选择变体；边际正态性不能替代联合独立性论证。本文性能与安全主张尚未由本项目复现。

本项目的[内容耦合分位实验](../reports/conditional_quantile_pilot_20261112/plan.md)属于待验证的防御假设。上述论文只用于方法查新，不引入其中任何新攻击。
