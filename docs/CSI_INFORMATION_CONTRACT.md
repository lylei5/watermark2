# CSI 公开实现的信息权限与比较契约

适用范围：以下源码结论针对文中固定版本和原核查日期；本轮未重新逐行认证全部来源。当前部署、实验与复现状态见[基线证据清单](DEFENSE_BASELINE_INVENTORY.md)。

核查日期：2026-10-16 UTC。**公开 CSI 库包含检测反馈驱动的候选搜索，不能原样作为本项目零目标反馈的攻击基线。** 同时，它提供了有最终布尔判定的 SEAL 研究重实现；这与 SEAL 原作者代码的缺项是两个不同事实，不能互相替代。

## 来源和核查范围

[第一作者主页](https://zhenggao-30.github.io/)的论文 Code 链接指向 [ZhengGao-30/CSI-Attack](https://github.com/ZhengGao-30/CSI-Attack)。固定 main commit 为 `8723e362835a6a61efd11b61f677b3321face3fa`，提交日期 2026-08-22；本次获取 28 个源码/说明/许可证文件，共 260162 字节，逐个匹配 Git blob SHA 并记录 SHA256，见 [来源清单](../reports/csi_source_evidence.json)。读取 LICENSE 正文，许可为 Apache-2.0。未导入、安装或运行该库，未运行模拟 selftest，未下载权重或调用 LLM。

[`NOTICE.md:14–31`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/NOTICE.md#L14-L31)说明这是由论文和实验代码整理的库，不是原始实验目录的逐字节档案；水印数学模块是研究重实现。以下代码判断仅针对这个固定版本。它不能单独证明早期论文每一项实验的精确权限和参数。

## 检测反馈的实际数据流

固定版本 [`csi/attack.py:171–193`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L171-L193)把失败分区名或检测分数/阈值组装成建议；[`256–265`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L256-L265)将 feedback 传给下一轮 LLM；[`371–379`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L371-L379)实际调用检测器、按失败结果反馈，并在启用选项时成功即停。反馈最终进入提示词，见 [`csi/llm.py:194–201,231–244`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/llm.py#L194-L201)。

因此，检测在此不只是离线日志：它可影响后续候选；可选提前停止默认关闭，不改变失败反馈已经存在的事实。同批已提出的候选不会被这条反馈回溯修改；下一轮是否发生取决于剩余预算。源码使用“black-box”描述不等于我们定义的“零目标反馈”。

| 接口 | `code` 模式 | `paper` 模式 |
|---|---|---|
| 文本/视觉过滤参考 | 从已交付水印图重新 caption 的整句 embedding | 对 caption 保留 anchor tokens 后编码 |
| 第三层 CSW 过滤 | 禁用 | 使用候选 caption embedding 与 `semantic_reference(emb)` 比较 |
| 目标检测反馈 | 失败后加入下一轮建议；可选成功早停 | 同一反馈路径 |
| 指定属性要求 | proposer 保留主体并变化次要细节，不接收 intent 参数 | proposer 接收明确的 Modification Target |
| 已报告结果关系 | 库说明为其已验证路径；本项目未复现 | 库说明没有已发表运行使用这一完整组合 |

模式差异依据 [`attack.py:20–33,229–239,335–369`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L335-L369)、[`llm.py:157–178,206–230`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/llm.py#L157-L178)。不能用 `paper` 模式结果冒称已复现 `code` 模式，或认为保持主体的内容多样化已经满足任意指定属性任务。

## 参考和模型权限

SEAL 插件先生成未加水印草图，以其 caption embedding v 构造水印噪声，保存 `ctx={embedding:v}`。`semantic_reference(emb)`直接返回这个 v，见 [`csi/schemes/seal.py:191–219`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/schemes/seal.py#L191-L219)。所以 `paper` 模式的第三过滤层使用生成端保存的绑定向量；它不是仅从交付水印图重新 caption 得到的攻击端参考。`code` 模式虽也取得该返回值，但第三层禁用，不据此声称这个向量参与了该模式的候选选择。

攻击 run 同时持有 key、watermark 对象和 emb.ctx，见 [`attack.py:197–239`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L197-L239)。这证明共享控制器有这些权限；不自动证明 proposer 读取了密钥字节。正式比较必须分别审计实际信息流和权限隔离。

标准 Tools/工厂路径让水印与攻击使用同一 diffusion backend 和 captioner；attack_encoder 可单独提供，默认复用 encoder。依据 [`csi/tools.py:34–42`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/tools.py#L34-L42)、[`csi/campaign.py:182–190`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/campaign.py#L182-L190)以及 [`csi/schemes/__init__.py:53–57`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/schemes/__init__.py#L53-L57)。这与本项目 SDXL 目标、SD2.1 代理失配设置不同；通用 API 可另行配置，不代表已有报告采用了分离模型。

当前 SD backend 只从参考图 DDIM 反演取得 zT；返回 `eps_schedule=None`，生成函数没有消费 eps 参数。见 [`csi/backends/sd.py:138–154,177–204`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/backends/sd.py#L177-L204)。因此论文 Eq.2 的 CSW_Noise/per-step schedule 抽象，在这个后端不是读取目标原始全轨迹的证据。

## 成功定义和预算

默认 Budget 为最多 60 轮、每轮 10 个 prompt、每参考最多生成 5 张图、成功早停关闭，见 [`csi/record.py:352–358`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/record.py#L352-L358)。文本过滤失败不消耗图像预算，生成后即计数，后续视觉过滤失败也占预算；重复 prompt 跳过，见 [`attack.py:273–333`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/attack.py#L273-L333)。生成预算、LLM 提案数和目标查询数须分列。

`CampaignResult.asr_per_prompt` 是至少一个候选成功的参考数除以全部参考数，不按原图检测通过先筛分母；另报 genuine TPR。单 run 在默认不开早停时，ASR 分母却是实际到达检测器的候选数，排除了过滤失败的候选。依据 [`campaign.py:70–83`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/campaign.py#L70-L83)、[`record.py:523–558`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/record.py#L523-L558)。不能把二者或论文简略 ASR 定义与我们一任务一候选的联合成功率混在同一列。

候选 success 只读取 `detect.detected`，见 [`record.py:224–226`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/record.py#L224-L226)；没有额外的指定属性达成判定。caption 相似度过滤可约束主体，但不等价于独立核查目标编辑完成。

## SEAL 判定可执行，但属于新的重实现对象

该库 SEAL 检测会重新 caption 候选，生成语义参考噪声，并反演候选；计数 `m=sum(patch_l2<=t_patch)` 后 `m>=threshold` 接受，见 [`seal.py:221–230,282–307`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/schemes/seal.py#L282-L307)、[`types.py:110–135`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/types.py#L110-L135)。它确实是语义绑定的完整布尔路径，不是 SEAL 原库的穷举码字分支。

但它以 SplitMix64 标记码字、NumPy RNG 生成噪声；源码明确标注与参考实现不同，见 [`seal.py:12–21,93–147`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/schemes/seal.py#L93-L147)。不能把其向量、常数和随机状态用于替补 SEAL 原库或几何验证器。

活动校准代码在同一 clean 集上取得 patch quantile、确定图像级阈值并计算该集合的 FPR；正常图也在校准中计算 TPR，见 [`calibrate.py:78–116`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/calibrate.py#L78-L116)、[`seal.py:232–254`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/csi/schemes/seal.py#L232-L254)。这些是校准集指标，不能自动视为独立测试保证；campaign 另有 clean 检查，但调用者仍需保证数据划分。

README 的 SEAL 49/60 使用该重实现自己的工作点；NOTICE 区分它与另一个代码库的结果。论文 Table 1 的 81% 与当前 README 的近似数值，不能视为逐样本重现证明。这里仅记录作者发布材料，未取得完整原始运行归档，也未自行复现。[`README.md:353–395`](https://github.com/ZhengGao-30/CSI-Attack/blob/8723e362835a6a61efd11b61f677b3321face3fa/README.md#L353-L395)。

## 对本项目的具体影响

1. 将此版本归入“有检测反馈的候选搜索”比较项；原始论文的精确查询协议仍未由原始实验档案独立确认。只在出处和完整协议核实后用于防御评测。
2. 候选生成、过滤与反馈均属于已发表基线的待核对协议，不作为本项目创新方向；不创建或改进攻击方法。
3. CSI 的 SEAL 插件可作为另行命名、固定版本的候选评估实现，但不能用于宣称已补齐原作者 SEAL 或几何防御。当前 BLIP2、embedding 权重和真实 LLM 提案依赖仍不满足本轮离线缓存/不新增模型的边界；只进行静态审查。

下一项可独立推进的工作是将公开 CSI、原 Reprompting 和 Jain 的预算、信息输入及成功分母冻结成统一比较表，并追踪最新相关工作，明确尚未被已有机制覆盖的研究问题。此处不修改防御阈值，不执行新攻击或模拟成功率。
