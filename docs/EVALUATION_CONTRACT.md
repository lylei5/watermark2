# 评估对象与威胁模型契约

适用范围：以下源码结论针对文中固定版本和原核查日期；本轮未重新逐行认证全部来源。当前部署、实验与复现状态见[基线证据清单](DEFENSE_BASELINE_INVENTORY.md)。

核查日期：2026-10-16（Asia/Shanghai）。范围：公开论文最终会议版、固定代码和既有实验记录的静态核查。未 import 上游入口，未加载模型、下载权重或运行新实验。来源字节哈希、文件清单和状态见 [审计证据](../reports/evaluation_contract_evidence.json)。

## 结论

**原 GS Reprompting 的局部判定可明确；完整几何防御与 SemBind GS-S 的可执行契约仍未闭合。** 当前不能报告这两种防御的攻击接受率。缺失字段保持 unknown/null，不以自行选择的阈值或替代参考补齐。

本轮确定的进展：核对了两个 ICML 最终出版 PDF；SemBind 最终版 E.1/Table 17 包含端到端代理 masker 评估，超出了此前 v1 附录 D 的阅读范围；完成 Jain GS 优化函数与评估入口的源码边界检查；明确了候选交付、参考关联和判定缺项。历史阶段报告保留，当前研究入口按这些证据更新。

## 1 来源与版本

| 对象 | 固定来源 | 状态 |
|---|---|---|
| 基础攻击 | [CVPR 2025 正文](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf)，补充材料 C；semantic-forgery `ca68950cd7b53b8438a49580f134b2a0db28d778` | 既有代码复现；本轮重读关键流程 |
| 几何防御 | [ICML 2026 PMLR](https://proceedings.mlr.press/v306/lee26bm.html)，出版社 24 页 PDF；geodistort-wm `72880c9bb1cee0b04bcb2433deb2ab2cc95ff622` | HEAD 与阶段 03 相同；本地源码干净；未发现许可证正文 |
| SemBind | [ICML 2026 PMLR](https://proceedings.mlr.press/v306/zhang26hj.html)，出版社 24 页 PDF；`337c9f90a150157ac1d410208ae20e6530ca8cdd` | HEAD 固定；枚举所有文件并检索 Python/notebook 符号；逐行查看关键路径；LICENSE 正文为 MIT |
| Jain LFA | [2504.20111v1](https://arxiv.org/html/2504.20111v1)；watermark_forgery_removal `04433962437cbf08ad701b7b43124de07df0b666` | 固定 HEAD，核查 GS 入口、核心优化与评估；未跑代码 |
| CSI | [2602.21593v1](https://arxiv.org/html/2602.21593v1)，WWW 2026 | 核查 §2/3；作者公开库 `8723e362835a6a61efd11b61f677b3321face3fa` 的检测反馈与两种模式已查明，详见 CSI 契约 |

几何最终版与 v1 的参考、指标、AUC 和 C.3–C.4 段落作了针对性核对，没有声称全文逐字一致。SemBind 以最终版为当前依据，预印本附录编号不沿用。仅下载论文和小型源码文本；无模型或数据集下载。GitHub 匿名 API 出现限额，使用已有 gh 认证读取公开仓库，未绕过访问控制。两个已固定上游的内容和 submodule pins 不变。

## 2 原 GS 判定与历史样本

固定配置是 256 位消息、64 次重复、l=1、offset=0。活动接受规则是 `bit_accuracy > 0.70703125`，是严格大于。它来自旧代码十万用户阈值表，不能自行解释为本项目已经实测验证了 FPR=10⁻⁶、全用户检索归属或几何判定。

证据：commit `ca68950cd7b53b8438a49580f134b2a0db28d778`，[`utils/utils.py:9–12,58–115`](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/utils/utils.py#L58-L115)，[`run_reprompting.py:124–205`](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/run_reprompting.py#L124-L205)。原始 CSV 的参考行和攻击行分别使用 step=-1、1。

阶段 02 没有保存目标原始噪声；阶段 04/05 保存了真实目标原始噪声、参考图反演、攻击图反演，共 27 份张量。阶段 04 的数据不能重标为阶段 02 当时保存的数据。阶段 04/05 可提供将来“已知配对参考”诊断的输入，但不能据此证明几何论文的真实部署检索已实现。

从相同消息/key/nonce 只能确定采样约束，不能确定连续噪声的幅值；种子也不能脱离随机调用顺序、库版本和状态恢复证明。证据：同 commit [`utils/wm/gs_provider.py:145–186`](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/utils/wm/gs_provider.py#L145-L186)。现有 8 个种子共享基础水印设置，是开发诊断；不能视为跨密钥的最终测试。

## 3 几何防御的参考与完整规则

### 3.1 参考来源

最终会议版 §4.3.1、定义 4.5/4.8、C.4 明确比较参考水印潜变量 `zT` 与待验证图目标反演得到的 `zhatT`。§4.1 同时提出不逐图保存原始生成特征的部署要求。公开材料尚未给出足以连接这两点的检索、重建或认证元数据机制。

代码在受控实验中持有目标生成时的 `wm_zT`：commit `72880c9bb1cee0b04bcb2433deb2ab2cc95ff622`，[`run_reprompting.py:203–217`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_reprompting.py#L203-L217)，再取得目标反演 [`284–314`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_reprompting.py#L284-L314)。这是实验中已知对应关系，不是任意外来候选的参考查找算法。

正常水印入口在提示词循环前建立原始噪声并复用；Reprompting 入口每轮建立原始噪声。前者见同 commit [`run_watermark.py:125–168`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_watermark.py#L125-L168)。因此不能把所有公开实验都概括成“逐图一份独立原始噪声”。

候选图到参考原始噪声的持久化关联、多个用户/候选参考的搜索策略、关联错误处理、参考是否需由提交者指定：**unknown**。若未来由实验控制器直接提供正确参考，应明确称“已知参考配对评估”，不能冒称零存储的任意图部署验证。

### 3.2 分数的已知部分

会议版定义 4.5：`G-Cos = <zT, zhatT> / (||zT||₂ ||zhatT||₂)`；SAD 为其 arccos。余弦高表示方向更一致，SAD 高表示更大偏移。定义 4.8：局部 token 中心化协方差采用 `1/M` 求和并加 `εI`，局部距离为 `||log(P^(-1/2) Q P^(-1/2))||F`。§5.1 指定 16×16 空间网格和 top-k 均值，k=5，L-SPD 大表示差异更大。

这些是论文数学说明；公开代码尚不能确定 token 的轴/patch 到 tensor 的映射、ε 数值、零范数处理、特征值截断、矩阵对数实现、精度、局部分数究竟按何种参考配对聚合，以及最后如何融合两个分数。相同符号名不是数值等价证据。

公开树缺 `utils/pipe/pipe_utils.py`、`utils/finger_utils.py`、`utils/spd_utils.py`。几何调用和张量保存被注释，活动代码不计算几何分数。证据：同 commit [`run_reprompting.py:23–33`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_reprompting.py#L23-L33)、[`345–438`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_reprompting.py#L345-L438)。原引本地完整文件证据未随当前仓库保存，不能作为可访问附件；其余来源清单见[契约审计证据](../reports/evaluation_contract_evidence.json)。注释阈值 0.125、0.37 等不是正式可执行协议。

### 3.3 判定闭合状态

论文 §5.1、Tables 2/4 报告每个分数的 AUC，C.4 将几何检查放在水印验证之前；定理 4.10 给出理论拒绝区域，但没有足够信息实例化部署规则。当前活动分支仍只有原水印判断，见同 commit [`run_reprompting.py:410–425`](https://github.com/chengyi-chris/geodistort-wm/blob/72880c9bb1cee0b04bcb2433deb2ab2cc95ff622/run_reprompting.py#L410-L425)。

| 完整判定字段 | 状态 |
|---|---|
| 原 WM 判定 | GS 严格 BA>0.70703125，源码可确定 |
| 几何得分可执行函数 | 缺失；论文公式不能自动补成作者实现 |
| 部署阈值及严格/非严格比较 | unknown |
| 理论拒绝区域 | B.3/Theorem 4.10：SAD>τs AND LGI>τl；阈值为低于各自伪造期望的正数 |
| 部署中 G-Cos 与 L-SPD 的融合函数 | unknown；理论区域与代码实际分数、阈值尚未对应 |
| 校准样本、划分、阈值选择目标 | unknown |
| 几何检查与 WM 的完整最终接受逻辑 | 论文给出顺序；可执行布尔规则 unknown |
| 校准后在未见测试集上的误拒/伪造接受 | 本项目尚无结果 |

附录 B.3 的 [定理适用条件核查](../reports/geometry_proof_obligations.md)进一步区分了正交误差增长率、相对谱偏差和集中界维度尺度。闭式例子只检查推理条件，没有构造受限攻击或替代验证器。

可以用形式化占位说明所缺：`Accept(x)=F(WM(x), Geometry(Invert_target(x), RetrieveReference(x)))`，其中 `F`、`RetrieveReference` 和完整 `Geometry` 尚未确定。这不是建议实现一个新规则。

## 4 SemBind 的对象与公开实现

最终会议版 §3.2–3.3：嵌入端由同提示词的辅助干净图得到语义码，重复/截断、按 mask ratio 保留、按秘密 `Kperm` 置换，映射到 ±1，逐元素乘原水印噪声。验证端从候选自身得到掩码，使用同置换，对目标反演噪声再乘该掩码解除绑定，然后调用底层水印解码与判定。这里需要的是 masker、置换密钥和水印验证身份，不要求用原辅助图作几何配对；没有证据表明应把其与 G-Cos 当作一个组合防御。

主实验的 SD2.1、DPMSolver、50 步、CFG 7.5 与 exact inversion，不能用我们当前 SDXL/DDIM 冒充。论文 inverse order=0；固定代码的 `exact_inversion` 默认 `inv_order=1`、`guidance_scale=3.0`、`decoder_inv=True`，且同时有两个步数字段。因此必须获得/固定实际调用参数，不能直接采用函数默认值。证据：commit `337c9f90a150157ac1d410208ae20e6530ca8cdd`，[`inversion.py:8–31`](https://github.com/XinZhang1999/SemBind/blob/337c9f90a150157ac1d410208ae20e6530ca8cdd/inversion.py#L8-L31)、[`95–151`](https://github.com/XinZhang1999/SemBind/blob/337c9f90a150157ac1d410208ae20e6530ca8cdd/inversion.py#L95-L151)；论文 §4.1。

代码确有网络和演示：[`model.py:109–135`](https://github.com/XinZhang1999/SemBind/blob/337c9f90a150157ac1d410208ae20e6530ca8cdd/model.py#L109-L135) 给出二值化；[`semantimark.ipynb:66–96`](https://github.com/XinZhang1999/SemBind/blob/337c9f90a150157ac1d410208ae20e6530ca8cdd/semantimark.ipynb#L66-L96) 加载 masker、生成随机噪声图像、提取哈希。对全部跟踪 `.py/.ipynb` 作绑定/水印/检测符号检索，并逐行查看 notebook、model、inversion 关键路径后，未见完整 GS-S 绑定/解绑/阈值评估入口；`src/prc.py` 的 PRC 置换不等于 SemBind 的秘密语义掩码置换。README 所称渐进发布与此吻合，但“不完整”结论来自实际文件检查。

README 已给 masker 的 Google Drive 下载链接；这应记为“链接公开、未下载、内容与训练配置尚未验证”，不能说没有权重链接。[`README.md:93–125`](https://github.com/XinZhang1999/SemBind/blob/337c9f90a150157ac1d410208ae20e6530ca8cdd/README.md#L93-L125)。底层 GS 阈值、消息/密钥关联、秘密置换生成/保存、最终判定比较符、准确模型 revision 和数值一致性仍未闭合。不能把当前 GS 的 0.70703125 直接搬过来填空。

论文把部署 masker 视为私有（最终版 E.1）。科研复现持有公开 checkpoint 是评估者能力；若攻击者也使用同一目标 masker、置换或目标内部反演，应单列为更强访问设置。公开模型身份不授予调用目标实例或使用目标密钥的权限。

最终版 E.1/Table 17 的端到端实验明确存在：5 个独立代理 masker，100 COCO prompts，GS-S Det.=0.00，BA=0.4988–0.5104。文字称攻击者嵌入自己的 GS 水印，但目标验证身份与消息/密钥对应未由公开实现澄清，故不能把这直接等同本项目的单图目标消息迁移设置。E.2 是像素覆盖；F 扩展提示词和 FLUX；G 比较 WMCopier。这些证据修正“只有组件测试”的不完整定位，却不意味着所有自适应策略已被穷尽。

## 5 与已有攻击的区别必须落在可验证的输入和计算上

| 方法 | 实际攻击计算 | 目标反馈的已知用途 | 与本项目主设置的关系 |
|---|---|---|---|
| 原 Reprompting | 代理反演后按指定提示词生成 | 固定入口验证/日志；不按分数改候选 | 已有单候选必要基线 |
| 原 Reprompt+ | 3 提示词；GS 再 3 次重采样 | 实验按任一通过计成功 | 多候选预算；不能当作零反馈的最佳候选选择器 |
| 原 Imprint | 优化代理 z0，经代理整段反演匹配参考 | 固定源码周期目标验证；无分数早停 | 固定预算版本可作为边界可控的既有基线；正文早停描述与代码分开 |
| Jain GS LFA | 直接优化像素，经代理 VAE 编码匹配参考，加像素 MSE | 核心函数无目标查询；外层生成/最终验证共享目标 | 单图 VAE 一致性已存在；需分离外层和核心、冻结其预算 |
| CSI 公开库 | LLM 候选、参考图反演、语义过滤；code/paper 两模式 | 失败检测分数/分区反馈下一轮；可选成功早停 | 有检测反馈；默认同 backend、多候选 any-success，与本项目零反馈单候选不同 |

原 Imprint 源码：commit `ca68950cd7b53b8438a49580f134b2a0db28d778`，[`run_imprint_forgery.py:186–251`](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/run_imprint_forgery.py#L186-L251)。正文 §3.1 写按目标准确率停止，但这一固定源码没有该活动分支。论文采用 150 步等报告预算；不能一概宣布所有 Imprint 都依赖查询择优，也不能忽略正文描述。

Jain 核心：commit `04433962437cbf08ad701b7b43124de07df0b666`，[`Gaussian-Shading/my_utils.py:261–299`](https://github.com/anubhav1997/watermark_forgery_removal/blob/04433962437cbf08ad701b7b43124de07df0b666/Gaussian-Shading/my_utils.py#L261-L299)。损失为代理 VAE mode 表示的 MSE 加 `lambda × pixel MSE`，按固定 iters 梯度更新并裁剪像素到 [-1,1]，非每步目标验证。外层 [`forgery.py:199–253`](https://github.com/anubhav1997/watermark_forgery_removal/blob/04433962437cbf08ad701b7b43124de07df0b666/Gaussian-Shading/forgery.py#L199-L253) 同进程持有水印生成器与目标 pipeline，不能把整段入口直接给受限攻击进程。

另有必须处理的静态指标问题：上述外层将 `final_p` 以 `<=0.05` 累计 ASR，但 [`watermark.py:80–92,152–164`](https://github.com/anubhav1997/watermark_forgery_removal/blob/04433962437cbf08ad701b7b43124de07df0b666/Gaussian-Shading/watermark.py#L152-L164) 的 `eval_watermark` 返回位匹配比例 `correct`，内部计数使用 `>= tau`。因此这个 GS 入口打印的 ASR 方向不能直接用于本项目结论。它是确定的源码语义不一致；不据此断言论文所有表格错误，未运行或改写上游。

[CSI 信息契约](CSI_INFORMATION_CONTRACT.md)已固定作者公开库 `8723e362835a6a61efd11b61f677b3321face3fa`：code 模式仅有文本/视觉过滤，随后查询检测器并将失败反馈下一轮；paper 模式增加的 CSW 参考来自生成端保存的草图 caption embedding。SD 后端只复制反演所得 zT，eps_schedule=None。公开库不是原始实验归档，故当前源码权限已确定，原始论文每项实验的精确访问条件仍需原始记录。删去反馈或替换参考会形成另一个明确命名的比较变体，不能冒称原样复现。

本项目使用上述信息契约界定已发表基线的可比范围。创新研究只针对防御；不新增候选生成、损失、反馈或选择策略，不以提高伪造接受率为研究目标。

### 5.1 单图 VAE 基线的精确数值约定

Jain 的 GS 核心参数名容易混淆：`clean_img` 是希望保留的新内容载体，`generated_image` 是已带水印的参考。固定 commit `04433962437cbf08ad701b7b43124de07df0b666` 的 [`my_utils.py:261–299`](https://github.com/anubhav1997/watermark_forgery_removal/blob/04433962437cbf08ad701b7b43124de07df0b666/Gaussian-Shading/my_utils.py#L261-L299) 将两者的 `posterior.mode()` **除以** VAE scaling factor，最小化该表示的 MSE 加到原载体的像素 MSE；梯度更新后只裁剪到 [-1,1]，不是围绕载体的显式 L∞ 半径投影。`delta>0` 时额外加入低频损失；默认 delta=0。核心固定迭代，不读取目标检测分数。

这与我们的 VAE-only 重建不同：阶段 09 没有优化载体，也没有把新内容映射到参考表示，只对参考本身编解码并测量目标反演。不能把阶段 09 叫作 Jain 攻击复现。

设 VAE 均值为 μ、scaling factor 为 s，Jain 表示为 μ/s，而当前官方 baseline 编码为 z=sμ（commit `ca68950cd7b53b8438a49580f134b2a0db28d778`，[`pipe_provider.py:154–162`](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/utils/pipe/pipe_provider.py#L154-L162)）。本项目的代数检查得到 `MSE(μ₁/s, μ₂/s)=s⁻⁴ MSE(z₁,z₂)`。因此直接改为乘 s 而沿用 λ 和梯度步长会改变数值优化；若以统一正倍数归一化损失，像素系数应对应 λs⁴，普通梯度下降步长也需相应换算。这是单位约定核查，不认定原实现错误，不自动修改或调参。

[`forgery.py:75–100`](https://github.com/anubhav1997/watermark_forgery_removal/blob/04433962437cbf08ad701b7b43124de07df0b666/Gaussian-Shading/forgery.py#L75-L100) 的入口默认 λ=2、alpha=5/255、n_iters=15000，不能拿 helper 的默认 iters=10 当作论文预算。入口 `vae_optimization` 默认 SD1.4，且主模块加载模型和远端提示词；本轮未执行。未来公平比较需要明确载体来源、代理权限、精确损失单位、迭代/时间预算及离线目标判定，不能直接继承它错误方向的 GS ASR 汇总。

### 5.2 参数来源与真实交付对象

历史载体比较记录（原链接文件未随当前仓库保存，以下细节尚待重新核对固定源码）记载：同一固定Jain commit的README GS命令使用λ=10000，与裸CLI默认2不同。入口先检测浮点输出再存PNG；独立evaluation/eval_gs.py虽读取PNG计算质量，其ASR仍来自文件名保存的旧分数，并非重新执行PNG水印检测。共同比较须使用冻结后的PNG与统一评估实现；原入口汇总不能直接当作文件交付成功率。该来源差异尚不能证明论文实际运行路径或表格有误。

## 6 主威胁模型与交付接口

攻击输入为一张已知有效水印参考 PNG、预定内容要求、代理模型/公开算法、预定随机种子和预算。参考的原生成提示词不默认提供；输出提示词是攻击者任务输入。目标公开名称可以作为明确注明的已知信息，但目标实例调用、目标 VAE/UNet 权重使用权限与它分开；若使用完全相同目标模型作代理，应列为单独更强设置，不能混进 mismatch 主结果。

目标水印 key/message/nonce、SemBind 目标 masker/秘密置换、原始噪声、生成 RNG、目标内部反演、各验证分数、参考检索数据库属于评估端。科研控制器为了生成样本持有这些信息，不代表攻击算法获准读取；旧归档已公开的值不应被硬编码到后续攻击，正式测试应有未见的目标材料。

候选先固定再评估：攻击输出目录只交付 `candidate.png` 和 manifest，含文件 SHA256、任务 ID、参考图片 SHA256、攻击代码/代理版本、公共参数、种子、预算、退出状态。评估端核对哈希后读取候选；评估后不改变本任务候选或参数。目标原始潜变量关联表仅由评估端持有。重新编码图片会改变文件哈希，应视为新产物而非静默替换。

单候选与已发表基线自带的多候选协议分别报告；仅沿用原方法固定的候选数、选择规则和预算，any-of-k 与 single-output 分列。不用目标分数淘汰失败种子、调参或挑图。现有目录/脚本划分不是权限隔离，当前 `permission_isolation_implemented=false`；本轮没有部署隔离系统。

评估结果应分别保留参考有效性、内容要求达成、旧 WM 结果、各几何分数、完整防御接受、归属身份、运行状态。攻击失败、运行失败、未知判定分开。只有规则闭合后才能统计最终接受率；正常误拒需用正常图和预声明良性变换测量，伪造接受需明确分母与参考有效条件。AUC 可单独报告，但不替代固定工作点，不从测试集择优反推阈值。

## 7 与既有数据的兼容性及下一步

| 对象 | 可复用 | 尚缺或会改变含义 |
|---|---|---|
| 原 GS | 当前 PNG、配置、原始 CSV；严格阈值可追溯 | 未见提示词/密钥的最终测试尚未建立 |
| 已知配对几何诊断 | 阶段 04/05 的真实目标 zT 和反演张量、图像哈希 | 官方完整函数、数值等价、参考部署机制、固定判定；阶段 02 原噪声未保存 |
| SemBind GS-S | 工程记录规范、部分工具与代理基线思想 | 新的绑定生成过程、masker/DINOv2、置换和底层 GS-S 协议；旧普通 GS 图不能追认为 SemBind 样本 |

现有目标 `stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b`，代理 `Manojb/stable-diffusion-2-1-base@0094d483a120f3f33dafbd187ea4aa60d10de75c`；512、float32、自定义 DDIM、50/50 步、生成 CFG 7.5、反演空 prompt/CFG 1。它们保持为历史事实，不为配齐论文配置而修改。

当前持续推进完整官方单样本材料的静态闭合：参考来源、数值规则、固定工作点与最终标签齐备后，才进入完整防御接受率评估。所需澄清包括参考检索、局部分块/正则、分数融合、WM 布尔关系、校准/测试划分与模型精确快照。这里是材料需求，不是已发送的作者联系。

补充的 [SEAL 契约](SEAL_EVALUATION_CONTRACT.md)已经定位到真正执行布尔判定的穷举分支，但它不等于语义防伪造分支；缺失依赖和语义固定工作点仍阻止本轮完整复现。已完成的目标往返与部分 VAE 对照只用于误差来源研究，不替代任何官方最终判定。


## 8 普通GS的解码与逐图状态契约

[阶段22源码和数学审查](../reports/22_decoder_contract/report.md)区分目标连续距离、加密取位、解密重复投票及严格阈值。参考BA=1不保证每列投票有大边距；代理符号一致也不直接推出目标一致。目标解密后边距属于评估端，不能作为攻击权重。

原论文v2 Appendix A说明实验逐图更新key与nonce。阶段17固定offset0、阶段21各两图复用offset1/2，是开发配对而非该逐图配置的复现。未来确认集须分别冻结消息/用户身份、逐参考key/nonce、生成RNG和任务分组；同参考各方法共享相同评估状态以保持配对。正式测试材料不得因采用公开固定表就声称已保密隔离。既有原始配置和结果保留。
