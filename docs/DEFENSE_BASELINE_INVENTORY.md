# 针对 Imprint 伪造方向的防御基线清单

核查日期：2026-11-03。判据：**顶刊顶会 + 有可运行代码 + 直接针对黑盒语义水印伪造**。
检索入口：Müller et al. CVPR 2025 攻击基线（arXiv 2412.03283）的完整引用列表，共 44 篇可见引用（30 防御类 / 14 攻击复现类），逐篇筛选。

## 0. 结论先说

**有缺口。** 此前评估的三项（SemBind / PGID / Geometry）不是全集。**ISTS（CVPR 2026 Findings）是最大的漏项**：官方仓库完整、代码里直接内置 `--atk_type imprinting`，且用了比我们更强的白盒威胁模型。

另外 CoverLock 与 Sign-Clamping 两篇新防御**只有预印本、无会议**，按"顶刊顶会"判据暂不列为基线，但需跟踪。

## 1. 先分清两类防御（决定比较是否公平）

这个区分之前没有写清楚，会影响论文里的对比口径：

| 类型 | 含义 | 我们的攻击怎么比 |
|---|---|---|
| **检测层 / 后处理** | 不改水印方案，在检测端加投影或判别 | 保持 GS + Imprint 不变，只在评估端加该层 |
| **新水印方案** | 替换 GS 的嵌入方式（语义绑定、动态注入等） | 必须换掉水印方案，重新跑攻击；这是**方案级对比**，不是防御层对比 |

- 检测层：**PGID、Geometry**
- 新方案：**SemBind、ISTS、PAI、SEAL、SLICE**

把新方案和检测层放在同一张表里比"检测率"是不成立的，必须在论文里分开。

## 2. 顶会防御基线清单

| # | 防御 | 会议 | 类型 | 官方代码 | 复现状态 |
|---|---|---|---|---|---|
| 1 | **ISTS** | CVPR 2026 Findings | 新方案 | ✅ 完整（`hala64/ISTS` @`562ec0fa`） | ❌ **未评估（最大缺口）** |
| 2 | **PGID** | NeurIPS 2026 | 检测层 | ✅（`MinhQDuong/PGID` @`eb7fb0dd`） | ✅ 已复现（Det 完全一致，BA +0.04–0.05） |
| 3 | **SemBind** | ICML 2026 | 新方案 | ⚠️ 部分（`XinZhang1999/SemBind` @`337c9f90`） | ⚠️ masker 位级复现；绑定链路未发布 |
| 4 | **Geometry** | ICML 2026 | 检测层 | ❌ **未发布**（README: "will be released soon"） | ⚠️ G-Cos 复现（AUC 1.000）；L-SPD 不可复现 |
| 5 | **SEAL** | ICCV 2025 | 新方案 | ✅（`Kasraarabi/SEAL` @`92d31b31`） | ❌ 阻塞：需未缓存 BLIP2/embedding 与固定语义协议 |
| 6 | **PAI** | ICLR 2026 | 新方案 | ✅（`QingyuLiu/PAI` @`9ebc58d1`） | ❌ 阻塞：缺稳健 PCA 参考与拟合协议 |
| 7 | **SLICE** | BMVC 2026 Oral | 新方案 | ❌ 作者页无 Code 链接 | ❌ 阻塞：无公开实现 |

## 3. ISTS 详情（下一步重点）

- 论文：*Towards Robust Content Watermarking Against Removal and Forgery Attacks*，Yifan Zhu / Yihan Wang / Xiao-Shan Gao
- arXiv 2604.06662，CVPR 2026 Findings
- 仓库：`github.com/hala64/ISTS`，单次 initial commit `562ec0fa936a7852831534a841a3ec73ed28a3c0`（2026-04-08）
- 方法：实例特定水印 + 双边检测。按 prompt 语义用 CLIP + K-Means(1024) 选择**注入时间 t∈[10,20] 与频域偏移 l∈[−12,12]²**；检测时取正反两向匹配度较小者，专门对抗"把水印推向反方向"的 imprint 移除。
- **它评估的攻击里直接包含 Imp-Forgery（Müller 的 imprint 伪造）**，还有 Imp/Avg/VAE 三种移除与三种伪造，且设定为**白盒（攻击者可用同一模型）**，比我们的黑盒更强。
- 依赖：python 3.12、diffusers 0.33.1、torch 2.6.0、transformers 4.51.2、numpy 1.26.4；`open_clip` 已内置；训练好的分类器在 `./results`
- 环境可行性：服务器 **pip 网络可用**（已验证），可建独立 env；模型侧只需 SD2.1（本地已有 Manojb 快照），**不需要下载新模型**

**风险**：diffusers 0.33.1 与本机 0.31.0 不同，且 ISTS 的 `inverse_stable_diffusion.py` 与 SemBind 同源，可能撞上同一个 `convert_model_output` 签名漂移。必须建独立 env 按它 pin 的版本跑，不要混用。

## 4. 预印本（暂不列为基线，需跟踪）

| 工作 | arXiv | 与我们的关系 |
|---|---|---|
| CoverLock / Residual Transferability | 2609.32241（v2 2026-10-24） | 提出"增强水印证据对封面图依赖"来抗残差迁移；与我们的"语义掩码一致率"是同一现象的两面 |
| Latent Coordinate Sign Clamping | 2610.40204（2026-10-29） | 双层伪造防御（感知哈希 + CLIP 上的 RS 模糊承诺），报告 98.8% 伪造检测准确率；**直接针对 GS 符号结构**，最贴近我们的攻击面 |
| PP-Mark | 2610.41170 | 可公开验证、形式化不可伪造 |

Sign-Clamping 这篇值得特别盯——它五天前才挂出来，且攻击面对得最准。

## 5. 顶会但与伪造方向不同（不作基线）

- **TokenTrace**（CVPR 2026，2602.19019）：多概念归属，不是抗伪造
- **PSyDUCK**（IEEE WIFS 2025）：把信息藏进去噪过程以提升容量，不是抗伪造
- **Latents-Inv**（IJCAI 2026）：鲁棒性导向，非抗伪造
- **PDMark**（Neural Networks 2026）：可否认水印，威胁模型是胁迫披露，不同

## 6. 建议评估顺序

1. **ISTS** —— 唯一"顶会 + 完整代码 + 内置 imprinting 评估"的缺口，优先
2. **SEAL / PAI** —— 代码都在，卡在材料缺项；若能从作者仓库补齐参考协议即可解阻
3. **SLICE** —— 等作者放代码
4. **Geometry** —— 等作者放代码；在此之前只报 G-Cos AUC，L-SPD 不引用
5. **Sign-Clamping** —— 若在正式会议前有代码，纳入

## 7. 必须写进论文的口径

- 新方案与检测层分开报告，不混表
- 每一项标注"官方代码 / 打补丁的官方代码 / 按论文重写"三档之一
- 只报 AUC 的项（Geometry）不产生接受/拒绝，`D=null`
- 未复现的项不得写成"该防御无效"
