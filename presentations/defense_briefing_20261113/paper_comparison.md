# 三篇防御论文：机制、比较对象与攻击覆盖

核对依据：SemBind 的 ICML 2026 最终版，PGID 的作者 v5（2026-11-06），SEAL 的 ICCV 2025 正文及补充材料。下表整理原论文的实验；本项目已执行的工作见[三个基线报告](../../reports/three_baselines_20261113/report.md)。

| 方法 | 防御放在哪里 | 主要解决的威胁 | 论文中的主要比较对象 |
|---|---|---|---|
| SemBind | 在已有载体的生成与检测流程中加入语义符号掩码 | 跨内容的黑盒水印伪造 | Tree-Ring、Gaussian Shading、PRC、Gaussian Shading++，分别比较原方法与 SemBind 版本 |
| PGID | 在检测端替换一次性反演，增加引导式反演—去噪循环 | 扰动引起的水印伪造与去除 | Tree-Ring、Gaussian Shading、PRC、T2SMark，分别比较普通反演与 PGID 版本 |
| SEAL | 从描述语义直接构造分块初始噪声，检测时核对内容与噪声 | 噪声复用伪造与局部内容篡改 | WIND、Tree-Ring、Gaussian Shading；再生成去除另比较 HiDDeN、Stable Signature、TrustMark、WOUAF |

## SemBind

辅助图经视觉编码器和语义网络得到二值码，扩展、置换后调制底层水印噪声。验证时从候选图重新提码，解除调制，再调用底层检测器。设计依靠相关内容的码保持接近、无关内容的码分离。

主实验测试 **Imprint 与 Reprompt**。附录 D 测试 **CtrlRegen** 再生成去除，附录 G 测试 **WMCopier** 伪造；两项附加攻击使用 TR、GS 及其 SemBind 版本。常规编辑包括 JPEG、亮度、模糊、噪声、中值滤波和缩放。[论文 §4、附录 D/G](https://proceedings.mlr.press/v306/zhang26hj.html)

研究切入点：训练中的同提示词一致性有助于辅助图与最终图对齐；进一步进行发布实例认证时，需要区分语义相近但不同的图像。PRC/GS++ 的部分绑定是另一项可单独消融的设计取舍。

## PGID

先记录待测图的普通反演轨迹，再执行多轮非对称反演—去噪循环：引导潜变量偏离受扰动轨迹，每轮去噪比反演多走若干步，最后完整反演并调用原检测器。PGID-F 与 PGID-R 使用不同参数，分别用于伪造复核和去除恢复。

主实验测试 **Imprint Forgery 与 Imprint Removal**；附录 D 更换攻击使用的调度器；附录 E 测试 **VAE-Forgery**（TR、GS、T2SMark）及 **Average-Removal**（TR）。主实验还区分代理模型与提供者模型匹配或不匹配的情况。[论文 §4–5、附录 D/E](https://arxiv.org/html/2605.09319v5)

研究切入点：扰动修复对载体、模式和参数存在依赖；普通编辑后的合法图保留也是需要评价的一端。本项目的合法图误拒观察来自[额外九条件实验](../../reports/pgid_published_revalidation_20261112/report.md)。原论文附录 G 将无扰动的重新生成类攻击列为局限。

## SEAL

辅助图经 BLIP-2 生成描述，再由微调文本编码器提取语义向量。SimHash 与私有 salt 为各个噪声块派生种子。检测时从候选图描述生成参考噪声，与反演噪声逐块比较；Spatial Test 进一步分析异常块的空间组织，帮助发现局部编辑。

原论文测试 **私有模型噪声复用伪造**、**Cat Attack** 局部对象插入，以及补充 §11 的 **Latent Forgery**。去除测试包括再生成和隐写分析残差估计，另测旋转、JPEG、裁剪缩放、模糊、噪声与亮度变化。再生成对比中的额外方法见补充 §13.3。[论文 §3–4](https://openaccess.thecvf.com/content/ICCV2025/html/Arabi_SEAL_Semantic_Aware_Image_Watermarking_ICCV_2025_paper.html)、[补充材料](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Arabi_SEAL_Semantic_Aware_ICCV_2025_supplemental.pdf)

研究切入点：全局描述可能对细粒度实例变化不敏感，重新提取也可能产生漂移；局部分块检测还受几何变化影响。本项目已执行描述语义判定与参考漂移诊断，空间篡改检测保留为后续验证项。

## 汇报采用的研究主线

围绕最终发布图学习实例表示，让允许编辑保持稳定，让同提示词的其他实例分离。随后在同样的实例码和记录权限下，比较直接内容门控与容错密钥恢复。既有攻击的原始实现用于防御评测，方法改动集中在防御端。
