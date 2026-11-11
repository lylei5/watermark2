# PAI：参考来源与验证流程契约

适用范围：以下源码结论针对文中固定版本和原核查日期；本轮未重新逐行认证全部来源。当前部署、实验与复现状态见[基线证据清单](DEFENSE_BASELINE_INVENTORY.md)。

核查日期：2026-10-17。范围：官方代码静态阅读、文件哈希与语法树检查，以及 ICLR 2026 最终论文的验证和攻击设置。本次没有导入上游模块、加载模型或执行检测。

**结论：普通验证的数值规则已经明确；公开版本尚不足以直接运行完整稳健验证。PAI 是另一种水印生成机制，现有 GS 样本不能作为其合法水印参考。**

## 固定来源

- [官方会议记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/51e2aaa731e8a5edfbda95fd8bd8948b-Abstract-Conference.html)确认 ICLR 2026；[最终 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/51e2aaa731e8a5edfbda95fd8bd8948b-Paper-Conference.pdf)共27页，SHA256 `b676bdacd4bbdd33b8d51f337fdcc528ed22d3aa6a69b841aec1b57d4ce8d484`。同时核查 [arXiv:2601.06639v1](https://arxiv.org/html/2601.06639v1) 对应章节。
- 官方仓库 `https://github.com/QingyuLiu/PAI`，固定 commit **`9ebc58d1a106e7155347f0d1f186e67313a4b0cf`**。以下所有源码结论均指此版本。
- 完整递归树包含11个文件：9个 Python、README、appendix.pdf。10个源码/说明文件及仓库两版附录的Git blob均已核对。树中未发现许可证文件；本项目只发布审计与来源哈希，不复制上游代码。当前仅main分支，无tag或release；唯一公开issue是指标询问且无回复，没有补充材料链接。详情见[来源记录](../reports/pai_source_evidence.json)。

已遍历当前main可达的全部5个提交，并核对父链到初始提交。9个Python文件只在当前最后提交出现；前四个提交只有README及至多一个附录PDF，没有另一套历史验证实现。两版附录均7页，均未发现嵌入附件或外链，提取文本中未见PCA参数说明。当前仓库附录§3列11类攻击；会议最终版A.4.2/A.6.4包含额外PCA空间攻击，不能把仓库附录当作最终版的完整材料包。历史对象、文件哈希和PDF结构预检见[补充证据](../reports/pai_history_and_endpoints.json)。这个检查只覆盖公开main历史，不证明私有材料不存在。

## 实际数据流

| 环节 | 已确认实现 | 固定源码位置 |
|---|---|---|
| 水印生成 | `encode.py` 构造 PAI，随机生成形状为 `[1,C,64,64]` 的密钥；以时间戳设置 NumPy RNG，再由 uniform 值和密钥的正态CDF构造初始噪声 | [encode.py:56–98](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L56-L98) |
| 轨迹偏转 | 指定步骤的预测干净潜变量逐元素乘 `1 + key * M_INJECT`；反演对应分支做除法。这里不是给 GS 增加一个后处理检测器 | [DDIM_watermark.py:59–93](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/DDIM_watermark.py#L59-L93) |
| 原始记录 | PNG 单独保存；`watermark.npy` 保存路径、列表和按文件名索引的字典，条目含密钥、提示词和时间戳 | [encode.py:100–124](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L100-L124) |
| 参考关联 | 读取候选清单指向的原始NPY，以候选文件的 basename 查找原始字典；由记录的时间戳重新生成 `[4,64,64]` uniform 值，并取原密钥 | [dataset/watermark_dataloader.py:94–117](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/dataset/watermark_dataloader.py#L94-L117) |
| 验证反演 | 从密钥和 uniform 值重建目标初始噪声；候选经 VAE 编码后，空提示词、guidance=1，带密钥反向偏转；输出参考噪声和恢复噪声 | [decode.py:49–76](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/decode.py#L49-L76) |

因此，参考潜变量来自**评估端的原密钥和登记时间戳重建**，不是参考图的普通代理反演。代码通过清单及 basename 关联候选与目标身份，没有实现任意来图的全库检索或可靠身份认证。生成代码将时间戳写入NPY，未见写入PNG元数据的操作。重命名、跨目录同名、缺清单时的处理未定义，不能自行猜测所属参考。

## 数值规则与完整接受事件

普通验证为展平全部非batch维度后的均方差：`mean((reference - recovered)^2) < 1.76`，等于阈值不通过。这是活动布尔判定，见 [utils.py:66–67](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/utils.py#L66-L67) 与 [verify_vanilla.py:12–26](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/verify_vanilla.py#L12-L26)。最终论文 §5.1/Eq.5 与 A.6.1 则给普通阈值 **1.761**；两者有数值差异，不能自动改成一致。

稳健验证的代码流程为：

1. 每图展平 `reference - recovered`，用读取的 `PCA.transform` 投影。
2. 对 ownership 和 attack 两套参数分别计算 `sqrt((z-mean) @ cov @ (z-mean))`。
3. 阈值为 `sqrt(chi2.ppf(1-alpha, d))`，`alpha=0.05`、`d`来自投影输出维数；严格小于时位于对应区域。
4. 输出 ownership 区域内比例和 `1 - attack区域内比例`。

代码依据：[verify_robust.py:12–49](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/verify_robust.py#L12-L49)。变量虽叫 `cov`，计算中没有求逆；其保存值是否本来就是精度矩阵 **unknown**，没有依据将其判为算法错误。

**关键缺项：** `pca_reference.npy` 不在递归树中，PCA拟合、均值/矩阵估计及校准集划分脚本也未见。无法核对投影维度、参数、校准样本和独立测试边界。两个验证脚本各自读取 `ours.npy`；稳健入口没有调用普通验证、应用其掩码或输出逐图融合判定。普通通过与稳健归属通过如何在实际结果中组合，尚未由可运行官方端到端入口确认。

论文最终版 §5.1–5.2 描述先做普通门控，再进行 PCA 验证；归属区域与无攻击区域是两个不同对象。A.6.1 的稳健阈值2.448是距离尺度，而 §5.2 对平方距离使用卡方阈值。所有权被接受与攻击未被标记应分别保存；完整内容认证不能仅由 ownership 比例代替。源码自身没有落实一个统一的内容认证布尔值。

最终版Table 2(b)对Reprompting报告O-ACC=97.00%、A-ACC=100.0%。按A.6.1的归属定义，前者名义上对应约3%的错误归属；后者描述报警表现。这不是一项已复现的3%“未报警伪造成功率”。原始逐图结果、样本对应和舍入信息缺失，无法据表恢复精确联合成功计数；将报警作为拒绝条件也需要额外的部署策略证据。这里分别记录错误归属与未报警的错误归属；原引比较契约未随当前仓库保存，不作为可访问依据。

## 静态运行阻塞与设置差异

| 项目 | 观察与影响 |
|---|---|
| 类属性缺失 | `get_noise_pred`读取 `self.SDXL`，但9个Python文件中未见此属性赋值，构造函数也未设置。原样入口在进入该分支时缺少所需状态。[DDIM_watermark.py:26–38、99–104](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/DDIM_watermark.py#L26-L104)。未通过运行触发错误。 |
| 元数据字段不对齐 | 生成的两个记录字典没有 `origin_img_path`；验证数据集第117行及后处理第80行却读取此键。未发现生成该字段的配套转换入口。[encode.py:109–124](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L109-L124)、[post_process.py:73–91](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/post_process.py#L73-L91)。 |
| batch记录 | 默认batch=6，但密钥只扩成batch=1，保存时访问 `key[i]`。含多图批次存在越界风险。[encode.py:51、73–75、101–117](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L51-L117)。 |
| 模型身份 | 入口仅给字符串 `stable-diffusion-2-1`，未指定组织、revision或本地快照；各组件分别 `from_pretrained`，也没有 `local_files_only`。[encode.py:56–67](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L56-L67)、[DDIM_watermark.py:40–57](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/DDIM_watermark.py#L40-L57)。不能据名称证明现有 Manojb 快照与作者权重等价。 |
| 精度与VAE | 加载没有显式dtype/autocast；VAE取后验mean并使用配置内scaling_factor，输入归一化到[-1,1]。未实际测量运行dtype。[DDIM_watermark.py:133–162](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/DDIM_watermark.py#L133-L162)。论文A.6.1称mixed precision。 |
| 步数与位置 | 入口50步、偏转5步、start_wm_step=1、强度0.1；`update_step`按零基索引选择1到5，而论文A.6.1写前5步。精确偏转时序仍须对齐。[encode.py:35–51](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L35-L51)、[DDIM_watermark.py:59–62](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/DDIM_watermark.py#L59-L62)。 |

这些是固定公开版本的工程与协议缺项，不是实测安全漏洞。没有为绕开阻塞修改上游或改变阈值。

## 已有自适应评估及项目边界

最终论文 §6.5、A.4.2、A.6.4 确实包含时间戳修改、白盒候选密钥优化和白盒PCA空间攻击；后者冻结随机候选密钥而非使用真实密钥。具体算法及实验比较已有描述，不能声称未考虑自适应攻击。此次递归树没有这些攻击的专门实现入口，论文结果未独立复现。

当前项目的目标是SDXL+GS，代理为SD2.1；PAI论文A.4.2/A.6.1采用SD2.1目标、SDXL代理。两者需要分别建立任务集和身份材料。现有图像可以作为外部输入的文件，但不能充当合法PAI正样本；现有GS密钥、消息、原始噪声也不能代替PAI密钥/时间戳/PCA参数。

攻击端仍只允许参考图、任务要求、代理模型及自身状态。PAI密钥、目标反演、初始化参考、PCA参数与判定留在评估端；公开时间戳只有在协议明确允许且实际提供时才属于攻击输入。公开模型架构不赋予目标内部调用权限。预先固定候选哈希再检测的交付方式可复用，现有统计工具的内容标签和完整判定仍保持unknown。

## 理论结论的适用边界

最终版A.3.1/Eq.28–31对邻近无效密钥推导二阶矩差异，推导需要处理共享时间戳的初始化相关性，以及不同轨迹预测项之间的相关性。Eq.30中的初始化二阶矩项不能仅凭两个初始化量各自服从标准正态就省略交叉项；若记比例因子为 a、两初始化量为 X/Y，则恒等式是 `E[(aX-Y)^2] = E[a²X²] + E[Y²] - 2E[aXY]`。a 本身还依赖密钥，相关性需要在同一概率空间说明。这是待满足的证明条件，不是实测绕过结论。

源码初始化 `F(K,U)=sqrt(-2 log U) cos(2πΦ(K))`（[encode.py:81–89](https://github.com/QingyuLiu/PAI/blob/9ebc58d1a106e7155347f0d1f186e67313a4b0cf/encode.py#L81-L89)）在固定 `0<U<1` 时对K连续，且导数绝对值不超过 `sqrt(-2 log U) sqrt(2π)`。因此邻近密钥下的初始化差异趋近0，不能用相互独立的噪声替代它们的联合关系。已做[标准库标量检查](../scripts/check_pai_initialization_continuity.py)：全部输入为合成常数，只核对连续性界及代数交叉项，[数值结果](../reports/pai_history_and_endpoints.json)不含模型、反演或检测。

更一般地，如果固定图片的验证分数对密钥连续，且合法密钥在严格接受区域内部，那么局部邻域内会有其他密钥仍位于接受区域。这是有条件的拓扑性质，不提供未知密钥的恢复方法，也未验证真实样本满足该前提。最终版A.5/Table 5已有密钥扰动与误接受率分析，因此还需明确可区分密钥的容差定义；不能声称作者没有考虑相近密钥。当前项目攻击者修改图片，评估者固定受害者密钥；“固定图片、改变验证密钥”的分析属于另一种问题，不能把它直接计为本项目黑盒图像伪造成功。

## 实验准入状态

官方PAI单样本仍需：精确模型快照、密钥/时间戳与候选关联、稳健PCA文件及拟合来源、逐图普通门控与两类稳健输出、上述接口缺项的官方对应实现。公开历史与附件核查没有补齐这些材料；PAI暂不进入模型实验队列，完整接受结果为null。出现新的官方材料时再重新核对，不反复检查同一版本；当前授权下不拟合替代PCA、不新增模型下载，也不把普通阈值实验写成完整防御复现。
