# 解码区域嵌入的查新边界

检索日期：2026-11-12。以下是与本轮防御机制直接相关的原始论文与作者来源；未由此引入任何新攻击基线。

| 先行工作 | 已有核心思想 | 对本轮贡献定位的约束 |
|---|---|---|
| [Gaussian Shading，CVPR2024](https://arxiv.org/abs/2404.04956) | 加密载荷、等概率高斯区域采样、重复译码 | 高斯载体和多数投票均为已有设计 |
| [SemBind，ICML2026](https://proceedings.mlr.press/v306/zhang26hj.html) | 从语义生成掩码，调制潜在水印 | 内容绑定本身不是新贡献；本轮保留其发布的语义组件 |
| [Filler、Judas、Fridrich，TIFS2011](https://doi.org/10.1109/TIFS.2011.2134094) | 最小加性嵌入失真与 syndrome-trellis codes | 最小失真编码及约束优化不是新概念 |
| [Mathon、Cayre、Bas、Macq，TIP2014](https://doi.org/10.1109/TIP.2014.2305873) | 用最优传输降低自然水印的嵌入失真 | “保分布且降低失真”的一般目标已有先例；作者的[机构出版记录](https://pro.univ-lille.fr/patrick-bas/publications)核对卷23、期4、页1694–1705，作者稿标识hal-00940754v2 |
| [TraceMark-LDM，作者稿v2](https://arxiv.org/html/2503.23332v2)及[期刊记录](https://www.sciencedirect.com/science/article/pii/S095741742600744X) | 重排高斯样本、分组处理小幅值元素，并微调编码器 | 不能把“保留幅值”或“小幅值可靠性”当作新的观察 |
| [LAW，2026预印本](https://arxiv.org/abs/2607.22386) | 角度载体与按幅值选择稳定坐标 | 幅值影响几何稳定性已有明确讨论 |
| [RAIN，2026预印本](https://arxiv.org/html/2609.14856v1) | 提取器以恢复正确GS决策区域为目标 | “无需精确恢复初始噪声”及决策区域视角已公开；本轮研究的是生成端的编码选择 |
| [Latent Coordinate Sign Clamping，2026预印本](https://arxiv.org/abs/2610.40204) | 潜变量符号写入及内容认证 | 符号操作、内容认证和高幅值选择不构成单独新贡献 |

本轮需要验证的具体命题是：在相同语义组件、身份载荷、载体坐标、生成预算和硬译码器下，利用重复码解码区域中的自由度，是否能减少水印对绑定对象的改变，并转化为实际合法接受及防伪收益。固定幅值反射的排序解和理想一次性密文下的联合高斯论证只是支撑这一命题的分析，不能单独证明论文创新性。

当前查新覆盖直接相关路线，并不构成穷尽检索或首创性保证。期刊论文与预印本的证据地位分别标注；预印本仅用于避免重复贡献，不作为攻击来源。

## 理想信道图

[机制图](figures/ideal_mechanism.png)及[矢量PDF](figures/ideal_mechanism.pdf)由249服务器的CPU仿真生成。50,000个独立64维高斯组，固定种子202611120；原始样本位于 `runs/decision_region_ideal_20261112/ideal_samples.npz`，摘要和哈希见[数据表](figures/ideal_summary.csv)与[清单](figures/figure_manifest.json)。

56/64约束下，最小反射与随机反射的平均每坐标平方改动分别为0.589307和1.500342。独立高斯测量噪声标准差为1、且没有语义错配时，两者的多数译码正确率分别为0.99228和0.99934。这支持“失真与物理可靠性存在取舍”的理想模型解释，**不构成扩散模型或防伪结果**。误差带分别为均值的1.96倍蒙特卡洛标准误与二项比例的95% Wilson区间；同一组在不同设置间配对，区间不是同时区间。

图形遵循 scientific-visualization 技能的数据溯源、原始样本及可见不确定性规范。软件方法参考：Kassis et al., 2026, [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)。此参考不用于支持水印防御结论。当前图为研究用格式，未声明符合某一投稿期刊要求。
