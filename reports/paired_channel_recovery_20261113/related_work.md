# 图像认证、编辑估计与可恢复绑定的相关工作

核对日期：2026-11-13。技术描述依据作者论文和出版方页面；论文中的实验数字不与本项目数据直接排名。

| 工作与可核验来源 | 已有机制 | 对本项目的要求 |
|---|---|---|
| Lin, Varodayan, Girod. Image Authentication Using Distributed Source Coding. IEEE TIP21(1):273–283,2012. [作者PDF](https://web.stanford.edu/~bgirod/pdfs/LinTransIP2012.pdf)、[PubMed元数据](https://pubmed.ncbi.nlm.nih.gov/21609882/)、[大学出版记录](https://experts.illinois.edu/en/publications/image-authentication-using-distributed-source-coding/)。DOI10.1109/TIP.2011.2157515 | 将量化图像投影做Slepian–Wolf编码，以查询图像作侧信息；解码结合EM估计亮度、对比度和仿射变化 | 纠错辅助认证与解码中估计编辑已有明确先例。新增价值需体现为信息来源、可区分性或协议成本 |
| Ur-Rehman, Živić. Two-Phase Method for Image Authentication and Enhanced Decoding. IEEE Access5:12158–12167,2017. [IEEE出版记录](https://ieeexplore.ieee.org/document/7959560/)、[作者大学主页](https://fdit.htwk-leipzig.de/en/fakultaet-dit/personen/professoren/profin-drin-natasa-zivic)。DOI10.1109/ACCESS.2017.2719632 | 利用信道侧信息认证局部图像，再将已认证数据反馈给信道解码器 | 侧信息、认证与纠错的组合本身不构成新机制 |
| Lai等. CSGuard: Toward Forgery-Resistant Watermarking in Diffusion Models via Compressed Sensing Constraint. [作者arXiv2605.01479v1](https://arxiv.org/html/2605.01479v1)，2026-05-02；本次未核实会议或期刊录用 | 在生成和反演轨迹施加秘密矩阵约束，观测取自原始生成轨迹 | 它修改生成/反演轨迹。本项目保持原生载体，采用最终图登记；秘密投影或轨迹约束的名称不能单独作为创新依据 |
| Yang等. ForenMark: Semantic keyed watermarking with evidence recovery for latent diffusion images. [Knowledge-Based Systems出版页面](https://www.sciencedirect.com/science/article/pii/S0950705126018101)。DOI10.1016/j.knosys.2026.117084 | 出版摘要描述DINO引导掩码、提示词条件语义密钥和证据恢复分支 | 内容绑定及恢复已有同期研究。当前仅核对出版摘要与书目信息；完整算法、代码和实验协议尚待取得后比较 |

当前配对方法的具体假设是：查询内容特征之外的原生反演统计能够估计共同编辑，再校正实例码。它已接入真实密钥恢复；配对打乱和内容自拟合是必要的机制对照。本轮配对方法降低平均残差，但未知编辑的恢复和强对照成本仍更好，故该实现保留为机制实验。

元数据脚本通过PubMed成功取得TIP论文记录，保存于[related_metadata.json](related_metadata.json)。IEEE Access的Crossref请求超时，记录于[日志](related_metadata.log)；该条作者、刊物、页码和DOI改由IEEE页面及作者大学主页交叉核对。ForenMark出版社正文访问返回403，以上仅引用可检索的出版社摘要，不据此宣称已完整阅读。

这些工作只用于防御方法查新。本轮攻击数据仍全部来自既有完整预算CVPR2025 Imprint输出。

软件记录：文献元数据提取采用citation-management的脚本与核对流程，来源为Kassis, Agarwal, He, Patel, Brueckner（2026），[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。本次核对页面为2026-09-02修订的v2；该工具用于文献记录，不作为本项目防御方法的理论或实验依据。
