# 共同编辑、实例区分与密钥恢复的相关工作

查新日期：2026-11-14。本文记录与当前机制直接相关的先例和比较要求；没有新增攻击实现。

| 工作 | 已核实机制与阅读范围 | 对本研究的要求 |
|---|---|---|
| Ojansivu、Heikkilä，Image Registration Using Blur-Invariant Phase Correlation，IEEE SPL 2007，[DOI](https://doi.org/10.1109/LSP.2006.891338) | 对中心对称模糊的相位配准；作者出版记录与摘要核对 | 相位归一化及模糊不变量已有基础，不能将频域恒等式作为新贡献 |
| Flusser等，Blur Invariants for Image Recognition，IJCV 131:2298–2315，2023，[出版社](https://link.springer.com/article/10.1007/s11263-023-01798-7) | 一般模糊不变量的理论与识别；全文核对 | 给出神经网络潜变量近似满足何种编辑模型的实证，区分代数性质和真实图像性质 |
| Cao等，MarkNull，USENIX Security 2026，4721–4740，[会议页面](https://www.usenix.org/conference/usenixsecurity26/presentation/cao) | 噪声—潜变量关联；第8节防御使用反演再生成LPIPS误差；全文核对 | “使用两路潜变量”已有先例；关联测量与最终防御判定分别比较 |
| Lu等，Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection，ICCV 2025，19248–19257，[正式论文记录](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Tracing_Copied_Pixels_and_Regularizing_Patch_Affinity_in_Copy_Detection_ICCV_2025_paper.html)，[作者全文](https://arxiv.org/html/2602.17484v1) | PixTrace追踪编辑像素位置，CopyNCE以重叠面积监督局部对应；读方法、实验和附录。论文同时给出描述子与配对匹配器，并在NDEC研究相似实例 | 局部匹配、几何一致性和局部对比训练已有较强先例。后续局部恢复机制须比较同等查询/参考信息及存储成本。[作者仓库](https://github.com/eddielyc/CopyNCE)与ViT-S描述子/匹配器权重链接已核实；249访问Google Drive超时，尚无本项目实测结果 |
| Wang、Sun、Yang，A Benchmark and Asymmetrical-Similarity Learning for Practical Image Copy Detection，AAAI 2023，[DOI](https://doi.org/10.1609/aaai.v37i3.25366) | 出版社摘要说明困难负例与非对称相似度；本轮未读完整实现 | 普通随机负图之外，需要相似实例的独立测试；不能只靠十张普通图声称极低误认率 |
| Xie等，Dual-Guard，2026预印本，[全文](https://arxiv.org/html/2604.19090v1) | 初始GS水印与最终潜变量指纹；Full模式保存每图参考潜变量，进行分块内容检查；方法和威胁模型核对 | 参考潜变量、初始/最终两路证据和局部认证已有公开设计。原生载体不改、记录尺寸及恢复性质需明确各自价值 |
| Yang等，ForenMark: Semantic keyed watermarking with evidence recovery for latent diffusion images，Knowledge-Based Systems，2026，[DOI](https://doi.org/10.1016/j.knosys.2026.117084)，[出版社](https://www.sciencedirect.com/science/article/pii/S0950705126018101) | 出版社摘要及开放文章预览：DINO引导嵌入位置，提示词语义键，低置信度时调用恢复模块；未核验完整作者实现 | 语义键与恢复分支组合已有工作。发布实例与提示词语义的区别必须通过同提示词实例及已有伪造攻击测量 |
| Juels、Sudan，A Fuzzy Vault Scheme，ISIT 2002扩展摘要，[作者全文](https://people.csail.mit.edu/madhu/papers/2002/ari-conf.pdf)，[作者出版记录](https://www.arijuels.com/publications/) | 用有足够重叠的无序集合恢复同一秘密 | 把全局码改成局部特征集合，再接密钥恢复，也有直接密码学先例；须证明新的可恢复区域或观测机制的收益 |

## 当前实验检验什么

本项目的可检验假设是：合法处理对内容潜变量与原生反演噪声造成足够接近的变化，因而两路的相对关系可形成跨未知编辑的实例恢复区域。真实解码与原生载体使用恢复材料完成验证。单路、普通拼接、配对打乱、连续距离和SSCD对照用于分开各项贡献。

512维频谱池化的结果暴露出实例信息丢失，因此补充完整16384维相位作为信息量诊断。该控制使用32倍维数，只用于定位信息损失；全部结果保留，未据它重新调整冻结实验。能区分普通独立实例，和能拒绝携带原水印的既有Imprint输出，是两项不同要求。

## 下一步机制的验收问题

1. 未知编辑下，哪些可观测证据支持“这是同一张图”？应分别测量未登记处理、同提示词实例和完整预算攻击。
2. 若校正器将查询拟合回参考，它能否把其他实例也拟合进去？应保留实例差异的独立检查，而非仅报告拟合残差下降。
3. 相同合法通过数下，直接内容门控、成熟拷贝检测和实际密钥恢复的误认是否不同？差异应来自可解释的新区域或协议性质。
4. 密钥恢复是否增加了有效信息，还是实现了已有距离判定？比较完整接受集合、辅助数据、查询成本和秘密输入。

这些问题用于确定后续实验的价值。局部特征、额外纠错、双通道或恢复模块的名称本身不构成创新依据。

## CopyNCE公开资产范围

核实版本为`c2223d18abf3289202666a0d2b9db1c3a7f1b483`。[该版本README](https://github.com/eddielyc/CopyNCE/blob/c2223d18abf3289202666a0d2b9db1c3a7f1b483/README.md)发布224×224 ViT-S描述子和匹配器两个基础检查点，并说明其他配置的权重受发布流程限制。基础检查点、经微调检查点、局部裁剪集成及分数归一化应分别标注；不能将基础检查点称为复现论文全部最高分配置。

249本轮访问作者GitHub发生TLS连接中断，Google Drive权重链接连接超时。没有下载模型到本机，也没有用其他模型代替CopyNCE。本轮已完成的强对照仍是同图SSCD单参考、九模板float32/int8及各候选的连续距离门控。
