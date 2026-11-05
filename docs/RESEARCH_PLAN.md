# 面向 CCF-A 的语义绑定水印防御与鲁棒性提升研究计划

**更新日期：** 2026-11-06  
**核心目标：** 以 **SemBind（ICML 2026）** 为基础，针对其在自然语义碰撞、公开骨干无密钥特征对齐、局部空间盲区、硬二值化鲁棒性悬崖及双倍生成开销等方面的漏洞，设计并验证新一代语义绑定水印防御框架（**SemBind+**），在抗伪造能力（含自适应攻击）与失真/编辑鲁棒性上全面超越现有 SOTA 防御（SemBind、PGID、ISTS、Geometry、SEAL、PAI），冲击 CCF-A 类顶会（CVPR / ICCV / ICML / NeurIPS / CCS / S&P / USENIX Security）。

---

## 1. 论文核心故事线（Storyline）

1. **现状与痛点（Why Existing Defenses Fail）**：
   - 潜空间扩散水印（如 Gaussian Shading、Tree-Ring）易受黑盒伪造攻击（CVPR 2025 Reprompting / Imprinting）和单图 VAE 优化攻击（Jain et al. LFA）威胁。
   - 2025–2026 年提出的两类防御均存在关键漏洞：
     - **以 SemBind（ICML 2026）为代表的语义掩码绑定方案**：
       1. **跨图哈希碰撞与投票放大（V1）**：对比学习训练的 1024 位语义哈希存在非平凡跨图碰撞（实测 190 对不同图像最小 Hamming 距离仅 29/1024，一致率 97.2%），经 GS 的 $64 \times 256$ 多数投票放大后，在仅 30 步普通黑盒 Imprint 攻击下即出现 2/28 样本被直接伪造攻破（最高 BA 0.9844）。
       2. **公开骨干特征瓶颈（V2）**：私有 masker 完全建立在公开 `DINOv2-giant` `[CLS]` 特征之上，攻击者无需获知私有 masker 权重或秘密置换 $\pi$，仅需在攻击损失中对齐公开 `DINOv2 [CLS]` 即可实现“无密钥特征对齐绕过”。
       3. **全局 `[CLS]` 空间盲区（V3）**：丢弃全部空间 patch tokens，无法抵抗局部区域篡改与拼接伪造。
       4. **硬二值化鲁棒性悬崖与 $2\times$ 生成漂移（V4）**：边界 logits（$|h_i|\approx 0$）在轻度失真下易翻转（实测最多 78/1024 位翻转）；降低掩码率 $\sigma < 1$ 则直接暴露 $(1-\sigma)L$ 个未绑定坐标；且两阶段生成带来 $2\times$ 延迟与跨轨迹语义漂移。
     - **以 PGID（NeurIPS 2026）/ Geometry（ICML 2026）/ ISTS（CVPR 2026）为代表的同期方案**：
       - PGID 无语义与密码学绑定，随攻击优化步数增加防御后 BA 单调攀升（150 步达 0.6641，距阈值仅 4.3pp）；
       - Geometry 依赖有状态原图潜变量检索，且局部 `L-SPD` 指标在 $16\times 16$ 网格下实测 AUC 仅 0.480（随机猜测），$32\times 32$ 网格因秩亏退化；
       - ISTS 离散时间/频域参数空间熵过低，双边检测仅防符号反转移除而不防全潜空间复制伪造。

2. **我们的方法贡献（SemBind+：Three Synergistic Pillars）**：
   - **贡献一：置信度感知的软解绑与加权 LLR 译码（Margin-Aware Soft Unbinding & LLR Decoding）**
     - 保持 $\sigma = 1.0$（100% 坐标全绑定），利用语义 logit 裕度 $|h_i|$ 与反演潜变量幅值 $|z_{\text{inv}}|$ 对 $64 \times 256$ 重复投票做对数似然比加权，并给出加权集中不等式下的信噪比增益与抗伪造安全性理论界。
   - **贡献二：密钥化多层空间-语义联合绑定（Keyed Multi-Granular Spatial-Semantic Binding）**
     - 对多层 DINOv2 特征施加验证端密钥化随机正交投影（抵御基于公开骨干的无密钥特征对齐攻击）与平衡码本约束（消除自然碰撞长尾）；同时融合全局 token 与空间 patch tokens 进行分块潜空间绑定，实现局部篡改检测与定位（Tamper Localization）。
   - **贡献三：单遍轨迹自绑定与流形净化协同（Single-Pass Trajectory Binding & Detection-Layer Synergy）**
     - 在单次扩散去噪早期步通过 Tweedie 干净图像估计 $\hat{x}_{0|t}$ 提取语义掩码，将生成成本从 $2\times$ 降至 $\sim 1.15\times$ 并消除辅助图漂移；验证端结合轻量级流形去噪投影抵御高步数自适应攻击。

---

## 2. 分阶段实验执行路线图

### 阶段 1：漏洞实证与基线补齐（Vulnerability Audit & SOTA Baseline Completion）
- **任务 1.1（SemBind 跨图碰撞与 150 步基线）**：在 `confirmation100` 全部 100 张载体（4950 对跨图组合）上完成 `SemanticMasker` 碰撞分布统计；补齐 30 对逐对独立身份（Per-Pair Identity）下 150 步 Imprint 对 SemBind-GS 的完整基线测试。
- **任务 1.2（SemBind 无密钥 DINOv2-CLS 对齐自适应攻击验证）**：实现加入公开 `DINOv2 [CLS]` 对齐损失的自适应 Imprint 攻击，定量证明原版 SemBind 在不泄露 masker 权重与置换 $\pi$ 时的脆弱性。
- **任务 1.3（ISTS 与 PGID 扩展评估）**：在服务器独立环境跑通 ISTS（CVPR 2026 Findings，`hala64/ISTS`）的 `Imp-Forgery` 基线，完成 PGID $n \ge 5$ 与流形预补偿对照。

### 阶段 2：SemBind+ 防御算法开发与模块消融（Method Development & Ablation）
- **任务 2.1（Margin-Aware Soft Unbinding）**：在 `scripts/defense/` 下实现基于连续 logit 与潜变量幅值的软判决解码器，无需重新训练模型即可在 6 类常见失真（JPEG、高斯模糊、缩放、裁剪、亮度、高斯噪声）与黑盒伪造上验证 TPR/FPR 与鲁棒性提升。
- **任务 2.2（Keyed Multi-Layer Spatial-Semantic Masker）**：实现密钥化多层特征投影与空间分块绑定，验证对“DINOv2-CLS 对齐攻击”的防御率（FPR/ASR $\to 0$）及局部拼接篡改定位能力。
- **任务 2.3（Single-Pass Tweedie Binding）**：实现单遍去噪早期语义提取绑定，对比两阶段生成与单遍生成的自检位准确率、FID、CLIP-Score 与单图耗时。

### 阶段 3：全面对比实验与论文成稿（Full Benchmark & Paper Writing）
- 在统一协议（逐图独立密钥/nonce、固定 FPR=$10^{-6}$ 阈值校准、100 对测试集）下完成 **Vanilla GS、SemBind、PGID、ISTS、Geometry、SEAL 与 SemBind+（Ours）** 的全矩阵对比表与理论证明整理。

---

## 3. 实验服务器与资产规范

- **服务器与工作目录**：`liyanlei@172.18.185.249`，主目录 `/data0/liyanlei/watermark_research`。
- **核心已就绪模型与数据资产**：
  - 扩散模型缓存：`Manojb/stable-diffusion-2-1-base`、`stabilityai/stable-diffusion-xl-base-1.0`；
  - SemBind 模型权重：`models/sembind/dinov2-giant`、`models/sembind/sembind_masker.pth`（886 MB，位级复现已验证）；
  - 第三方防御基准代码：`third_party/SemBind`、`third_party/PGID`、`third_party/SEAL`、`third_party/semantic-forgery`、`third_party/geodistort-wm`；
  - 标准测试集：`datasets/confirmation100_relay_v11/rgb512`（100 张 512×512 PNG 载体与配对提示词，SHA256 已核验）。
- **GPU 使用规范**：启动任何 GPU 任务前核对 `nvidia-smi` 空闲状态与物理卡 UUID，通过 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 与 `CUDA_VISIBLE_DEVICES=<UUID>` 绑定单卡，不干扰服务器上其他用户的进程。
