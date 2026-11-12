# SEAL 官方评估契约核查

适用范围：以下源码结论针对文中固定版本和原核查日期；本轮未重新逐行认证全部来源。当前部署、实验与复现状态见[基线证据清单](DEFENSE_BASELINE_INVENTORY.md)。

当前状态（2026-11-13）：**尚未完整复现SEAL防御基线**。249上已有BLIP2与caption-embedding权重文件，分片索引所列文件均存在；本次只检查存在性、文件大小和分片闭合，没有加载模型或完成逐权重哈希。见[资产核验](../reports/phase1_seal_status_20261113/assets.json)。尚无原生生成、语义判定与完整攻击评测结果。

以下源码审查完成于2026-10-16 UTC。公开实现包含语义绑定分数和另一个具有布尔判定的穷举检测器，两者不能混作同一评估对象。现有GS图像也不能追认为SEAL原生样本。

## 来源与范围

固定源码 commit：`92d31b31b93a6e373fe88a584c640beafb68c6fa`，官方 [Kasraarabi/SEAL](https://github.com/Kasraarabi/SEAL/tree/92d31b31b93a6e373fe88a584c640beafb68c6fa)。以下全部源码链接均固定到该完整 SHA。检查了完整递归目录及 17 个选定源码/说明文件；文件哈希见 [证据清单](../reports/seal_source_evidence.json)。未发现许可证正文。未复制上游实现到本仓库，未执行入口、安装脚本或模型加载。

论文实际阅读范围是 [arXiv:2503.12172v1](https://arxiv.org/html/2503.12172v1) §3.2、§3.3、§5–6、附录 10。ICCV 最终 PDF 本次请求返回 403，不能将 v1 核查写成最终版核查。

v1 §3.2 Algorithm 1–3：由候选图像 caption 的语义向量和含秘密 salt 的哈希规则构建参考噪声；同时反演候选图。逐块距离低于 τ 的数量 m 达到 m_match 时接受。该参考由候选内容产生，与我们已有 GS 实验的原始目标噪声不是同一对象。Tampering Detection 段另讨论对每块全部码字的检查；不能用它替代语义绑定判定。§5–6 已考虑 private-model regeneration、Cat Attack 及更强伪造，不能称作者未考虑攻击适应性。

## 源码中实际执行的两条路径

| 路径 | 实际输入与计算 | 实际输出/判定 |
|---|---|---|
| 语义绑定 | 候选 caption → 归一化 sentence embedding → `generate_initial_noise`；候选 VAE 均值编码后做空提示词反演；逐块比较两份噪声 | `SEAL.py` 和 `image_transformations.py` 记录 L2 数组，所审入口未将 `--threshold` 接入最终布尔判定 |
| 穷举码字 | 候选反演；逐块枚举全部 2^b 个码字，取最小 L2；`sentence_model` 参数未用于候选 caption | `image_transformations_exhaustive.py` 真正执行 `m=sum(l2<tau)`、`m>=m_match`，默认 τ=2.3、m_match=12 |
| 穷举 TPR 评估 | 与穷举路径相同的无 caption 匹配计数 | `image_transformations_exhaustive_tpr.py` 用随机图分数的 99 百分位作阈值，随后在同一随机分数集合上报告 FPR |

逐项源码依据：

- 语义绑定流程：[`SEAL.py:67–119`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/SEAL.py#L67-L119)；[`image_transformations.py:22–36,94–118`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/image_transformations.py#L22-L36)。`SEAL.py:29` 的 threshold 定义和整个主函数核查未见使用；分数数组不等于部署判定。
- 穷举规则：[`image_transformations_exhaustive.py:26–95`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/image_transformations_exhaustive.py#L26-L95)，参数 [`110–111`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/image_transformations_exhaustive.py#L110-L111)，活动调用 [`161–167`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/image_transformations_exhaustive.py#L161-L167)。这里有真实接受/拒绝实现，但其语义权限和数学对象与 caption 路径不同。
- TPR 阈值来源与 FPR 分母：[`image_transformations_exhaustive_tpr.py:157–184`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/image_transformations_exhaustive_tpr.py#L157-L184)。这段未见独立校准/测试划分；不能将其经验百分位和默认常数移植为语义防伪造工作点，也不能当作独立测试 FPR。

## 参考噪声和数值约定

[`utils.py:481–557`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/utils.py#L481-L557) 实现 SHA256 派生种子、逐块随机投影 bit 和噪声生成。输入 embedding 先归一化；噪声形状为 `[1,4,64,64]`，网格边长 `ceil(sqrt(k))`，默认 k=1024 给出 32×32 块，每块 4×2×2 个值；逐块重置随机种子再生成高斯噪声。调用点常用 seed=42，所审 helper 没有独立 secret-salt 参数。与论文 secret salt 的部署对应关系尚需确认；这不是已证实的密钥泄漏或安全漏洞结论。

[`utils.py:559–582`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/utils.py#L559-L582) 为未按块维数归一化的欧氏范数，较小距离表示较好匹配。这个阈值不能套用到我们展平后的全局余弦。复现需固定随机数算法、设备、软件版本、caption/embedding 与模型快照；仅记录 seed=42 不够。

## 当前依赖和执行边界

[`SEAL.py:34–52`](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/SEAL.py#L34-L52) 默认扩散模型为 `stabilityai/stable-diffusion-2-1-base`、float16、revision=`fp16`，额外加载 `Salesforce/blip2-flan-t5-xl` 和 `kasraarabi/finetuned-caption-embedding`，并读取 `Gustavosta/Stable-Diffusion-Prompts`。穷举 TPR 脚本默认扩散模型改为 Manojb 镜像，但两个语义模型依然加载。2026-11-13实地检查已确认`/data0/liyanlei/watermark_research/models/seal/blip2-flan-t5-xl`与`caption-embedding`内存在权重、配置和分词文件。资产可用性与模型实际加载、官方快照同一性、端到端结果仍须分别验证，不能默换embedding或目标模型。

相同代码段在 `online=false` 默认下仍调用 `wandb.init`；GPU 自动选择也没有本项目的 UUID 绑定。安装脚本包含系统安装和依赖重装。因此入口不能直接作为受控复现命令执行。需要明确关闭外部日志、固定离线缓存和设备限制，再验证其行为。

SEAL 的水印由语义嵌入决定生成噪声。阶段 02–09 的普通 GS 图可以用作格式或对照输入，不能追认为 SEAL 正常水印样本。原 GS BA 阈值、目标原始噪声和几何参考关联机制也不能代替 SEAL 判定。

## 对研究选择的影响

SEAL 是具有实质代码的可考察对象，不是仅有 README 的空仓库。但采用它做主评估会改变水印生成机制，并引入额外语义模型、固定工作点和公开/秘密随机状态的协议需求。当前完整几何与 SemBind GS-S 的缺项仍应分别保留；不能用 SEAL 的穷举通过率宣称绕过语义防伪造，也不能用现有 GS 诊断冒充 SEAL 结果。

可核查的下一项材料需求是：完整语义路径的固定 τ/m_match 及其校准来源、secret salt 的实际实现、模型精确快照、一个正常样本从 caption 到最终标签的官方测试向量。当前只有静态材料审查，没有已发送的作者联系或新的实验结果。

## 相关攻击库中的 SEAL 重实现

[CSI 公开库](CSI_INFORMATION_CONTRACT.md)包含重新 caption 候选、生成语义参考并作补丁计数布尔判定的完整路径，但明确使用不同的码字哈希和 RNG，并自行校准工作点。它是可单独命名的评估对象，不补齐上述原库缺项。其结果没有在本项目运行，不能转记为我们对原始 SEAL 的复现。

## 公开问题跟踪的补充核查

2026-10-16 读取官方仓库公开 issues 及相关评论，没有发送消息。[Issue #4](https://github.com/Kasraarabi/SEAL/issues/4) 同样询问未接入的检测阈值，所读评论尚未提供维护者的完整验证入口。[Issue #1](https://github.com/Kasraarabi/SEAL/issues/1#issuecomment-2826397737) 虽已关闭，但“距离阈值设为 3、按 Lemma 3.2 定匹配阈值”的回复来自 `author_association=NONE` 的参与者，不能认作作者确认的固定协议。[Issue #5](https://github.com/Kasraarabi/SEAL/issues/5) 是第三方复现疑问，不能作为本项目实测或论文错误证据；[Issue #6](https://github.com/Kasraarabi/SEAL/issues/6) 的许可证询问仍开放。

维护者在 [Issue #2 的回复](https://github.com/Kasraarabi/SEAL/issues/2#issuecomment-2893240256) 明确解释：Cat Attack 表中的 TR/GS 为 0 表示未识别篡改/伪造，而非原水印已被移除或 bit accuracy=0。后续表格应分别报告篡改识别与原 WM 通过，不能沿用含糊的“检测率”。这些公开讨论补充解释，但仍未闭合语义路径的固定工作点。
