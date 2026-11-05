# 语义水印防御端基线清单与漏洞对比（2026-11-06 更新）

**研究视角：** 本清单服务于 **基于 SemBind 的语义水印防御与鲁棒性提升研究（面向 CCF-A 类会议）**。我们将 2025–2026 年针对扩散语义水印伪造（Reprompting / Imprinting / LFA）的顶会防御与最新预印本分为 **生成端新水印方案** 与 **检测端投影/判别层** 两类，梳理其代码可用性、本仓库复现状态以及我们发现的**防御缺陷/改进切入点**。

---

## 1. 两类防御机制的严格区分（决定论文对比表与协同设计）

| 防御类型 | 核心机制 | 代表工作 | 与我们研究（SemBind+）的关系 |
|---|---|---|---|
| **生成端语义绑定方案（New Watermarking Schemes）** | 在水印生成/嵌入阶段将图像或提示词语义与潜变量/轨迹绑定；验证端先提取语义解绑再检测 | **SemBind**（ICML 2026）、**ISTS**（CVPR 2026 Findings）、**SEAL**（ICCV 2025）、**PAI**（ICLR 2026）、**SLICE**（BMVC 2026） | **直接对标基线与核心改进对象**：我们在 SemBind 基础上解决其哈希碰撞、公开特征对齐绕过、空间盲区、硬二值化鲁棒性悬崖及 $2\times$ 生成开销问题 |
| **检测端投影/几何判别层（Detection-Layer Defenses）** | 不改变原水印（如 GS）嵌入方式，在检测端通过引导反演去噪或几何偏移判别伪造 | **PGID**（NeurIPS 2026）、**Geometry / GeoDistort-WM**（ICML 2026） | **横轴对比 + 纵深协同组件**：一方面单独对比其高步数攻击退化与有状态限制，另一方面可将轻量流形投影与 SemBind+ 软解绑结合构建纵深防御 |

---

## 2. 顶会防御基线与缺陷对比矩阵

| # | 防御方案 | 会议 / 年份 | 类型 | 官方代码状态 | 本仓库复现状态 | 核心缺陷与我们的改进空间 |
|---|---|---|---|---|---|---|
| 1 | **SemBind** | ICML 2026 | 生成端语义掩码绑定 | ⚠️ 部分公开（`XinZhang1999/SemBind@337c9f90`，仅发布 masker 与反演） | ✅ **Masker 位级零误差复现**（Hamming=0）；已按论文实现完整绑定/解绑链路（[`sembind.py`](../scripts/defense/sembind.py)）并完成 30 对实测 | **核心基座**：(1) 1024 位对比哈希存在跨图自然碰撞（最小 Hamming 29/1024），30 步 Imprint 下 2/28 样本被攻破（最高 BA 0.9844）；(2) 依赖公开 `DINOv2 [CLS]`，可被无密钥特征对齐攻击绕过；(3) 无空间局部性；(4) 硬二值化在失真下易翻转，降低 $\sigma$ 则泄漏未绑定坐标；(5) 需辅助图生成（$2\times$ 开销与语义漂移） |
| 2 | **PGID** | NeurIPS 2026 | 检测端引导反演去噪 | ✅ 完整公开（`MinhQDuong/PGID@eb7fb0dd`） | ✅ **已跑通复现**（[`repro_pgid_20261103`](../reports/repro_pgid_20261103/report.md)，Det 完全命中论文 0.00） | **无语义绑定，随攻击预算单调失守**：在 50/100/150 步攻击下防御后 BA 从 `0.5781 → 0.6211 → 0.6641` 单调上升，距阈值 `0.7070` 仅差 4.3pp；易受流形残差预补偿自适应攻击突破 |
| 3 | **ISTS** | CVPR 2026 Findings | 实例特定时间/频域选择 + 双边检测 | ✅ 完整公开（`hala64/ISTS@562ec0fa`，内置 `--atk_type imprinting`） | 🔄 权重与代码已盘点，待在独立环境中跑对比基线 | **参数熵低且不防全潜变量复制**：用 `CLIP + K-Means(1024)` 选 $t \in [10,20], l \in [-12,12]^2$，离散空间极小；双边检测 $\min(s_+, s_-)$ 仅防符号取反移除，无法阻止复制完整潜变量的 Imprint 伪造 |
| 4 | **Geometry** (`GeoDistort-WM`) | ICML 2026 | 检测端全局/局部几何距离 | ❌ 防御代码未发布（`chengyi-chris/geodistort-wm@72880c9` 仅含重跑入口） | ✅ **按论文公式实现并完成网格扫描**（[`repro_geometry_paper_task_20261103`](../reports/repro_geometry_paper_task_20261103/report.md)） | **有状态存储依赖 + `L-SPD` 数值退化**：需保存/检索每张原图的初始潜变量 $z_T$；实测 `G-Cos` AUC=1.000，但局部指标 `L-SPD` 在论文 $16\times 16$ 网格下 AUC 仅为 **0.480（随机猜测）**，$32\times 32$ 网格因协方差秩亏（$\text{rank}\le 3$）被 $\epsilon I$ 主导而退化 |
| 5 | **SEAL** | ICCV 2025 | 生成端 `BLIP-2` 文本描述 SimHash 分块绑定 | ✅ 公开（`Kasraarabi/SEAL@92d31b31`） | ✅ 已完成源码审计（[`SEAL_EVALUATION_CONTRACT.md`](SEAL_EVALUATION_CONTRACT.md)）与测试脚本（[`seal_smoke.py`](../scripts/defense/seal_smoke.py)） | **模型栈沉重且受限于文本重描述稳定性**：需额外加载 15.8 GB `BLIP-2` 与句向量模型并执行 $2\times$ 扩散生成；轻度图像失真或编辑引起的描述措辞变化即可破坏分块噪声匹配 |
| 6 | **PAI** | ICLR 2026 | 去噪轨迹前 5 步乘性偏转 + 稳健 PCA | ✅ 部分公开（`QingyuLiu/PAI@9ebc58d1`，缺 `pca_reference.npy`） | ✅ 已完成源码与连续性审计（[`PAI_EVALUATION_CONTRACT.md`](PAI_EVALUATION_CONTRACT.md)） | **中心化数据库依赖**：验证端必须按文件名从本地登记表读取生成时的密钥与时间戳 RNG 状态，无法进行无状态独立验证 |
| 7 | **SLICE** | BMVC 2026 Oral | 多语义因子分区绑定 | ❌ 未公开代码 | ✅ 已完成论文理论边界核查 | 依赖预设语义因子（subject/environment/action/detail）显式分离假设，尚无公开实现 |

---

## 3. 重点跟踪的最新预印本（2026 年 9–10 月，启发防御设计）

| 工作 | arXiv 编号 | 核心思想 | 对我们设计 SemBind+ 的借鉴与区分 |
|---|---|---|---|
| **Latent Coordinate Sign Clamping** | `2610.40204`（2026-10-29） | 结合幅值加权多数投票、BCH/Chase 软判决译码与感知哈希 + CLIP 上的 Reed-Solomon 模糊承诺（Fuzzy Commitment） | **高度相关**：验证了“幅值加权软译码 + 纠错模糊承诺”在抗伪造与鲁棒性上的潜力。区别在于其针对事后多比特嵌入，而我们在原生无损扩散水印（SemBind-GS）上构建**语义 logit 裕度 + 潜变量幅值联合软解绑**与**密钥化空间-语义绑定** |
| **CoverLock / Residual Transferability** | `2609.32241`（v2 2026-10-24） | 分析残差可迁移性，通过增强水印证据对载体图的依赖来抵抗跨图残差迁移 | 与我们对 SemBind“跨图掩码一致率/多数投票放大”的分析互为印证 |
| **PP-Mark** | `2610.41170`（2026-10） | 可公开验证与形式化不可伪造水印 | 提供公开验证威胁模型参考；我们聚焦于保持扩散语义水印高鲁棒性的同时实现抗伪造绑定 |
| **Gaussian Shading++** | `2504.15026v2` | 双通道设计、伪随机纠错码、软判决解码与公钥签名 | 侧重密码学签名与双通道编码，未绑定图像视觉语义（仍面临同图反演重生成/移植风险），可与我们的语义绑定层正交结合 |

---

## 4. 论文实验对比规范

1. **生成端方案与检测端防御分层汇报**：主表分别展示生成端语义绑定方案（Vanilla GS、SEAL、ISTS、SemBind、**SemBind+ [Ours]**）与检测端/组合方案（PGID、Geometry、**SemBind+ w/ PGID**），保证威胁模型与信息权限公平。
2. **双轴评估指标（Security × Robustness × Efficiency）**：
   - **安全性（Anti-Forgery Security）**：在标准黑盒伪造（Reprompting、Imprint 30/150 步、Jain LFA）与自适应伪造（DINOv2-CLS 对齐攻击、碰撞筛选攻击、局部拼接攻击）下的伪造位准确率（Forged BA $\downarrow$）与伪造通过率（ASR / FAR $\downarrow$，越接近 0 越好）；
   - **鲁棒性（Distortion & Edit Robustness）**：在 JPEG 压缩、高斯模糊、缩放、随机裁剪、色彩抖动、高斯噪声及重生成编辑下的合法水印位准确率（Clean/Distorted BA $\uparrow$）与真阳性检测率（TPR@FPR=$10^{-6}$ $\uparrow$）；
   - **生成与验证效率（Efficiency & Kualitas）**：单图生成 NFE / 秒数、FID $\downarrow$、CLIP-Score $\uparrow$ 以及是否无状态（Stateless Verification）。
