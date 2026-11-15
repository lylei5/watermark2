# 统一对比的适用性与原作者实现清单

核查日期：2026-11-15。这份清单界定可执行对象和缺项，不包含新攻击结果。检测校准、运行预算和最终阈值还需实验验收；当前运行状态见 `project_state.json`。

## 防御和载体

| 防御 | GS | TR | PRC | 其他原文载体 | 必须单列的差异 |
|---|---|---|---|---|---|
| SemBind | 支持 | 支持 | 支持 | GS++：完整实现未验收 | 发布语义网络和反演组件；绑定按论文补齐。原生 GS 空间重复与 PGID 扁平重复不同；原生 PRC 512 位与 PGID 256 位不同。 |
| PGID-F | 支持 | 支持 | 支持 | T2SMark | GS/TR/T2S 的 k=15、s=3、gamma=0.001；PRC 的 k=7、s=3、gamma=0.001。 |
| PGID-R | 支持 | 支持 | 支持 | T2SMark | k=10、s=1、gamma=0.045。F/R 各自作为固定检测器，不能按攻击标签切换。 |
| SEAL | 不适用 | 不适用 | 不适用 | 原生语义分块噪声 | 独立生成系统；语义检测、空间篡改检测和公开 seed42/论文私有 salt 路径分别标明。 |

“同载体”面板必须固定底层载体实现、布局、载荷、身份与生成配置。原生配置存在差异时，先报告各自原生系统，不能把同名 GS 或 PRC 当作已经严格对齐。共同载体适配只涉及原作者载体的参数和防御输入输出，不改动攻击的优化、重采样或选择规则。适配验收前不填写同载体横向排名。

## 攻击并集

依据 SemBind 的 [ICML 正式版](https://proceedings.mlr.press/v306/zhang26hj.html)、[PGID v5](https://arxiv.org/html/2605.09319v5)及 SEAL 的 [ICCV 正式版](https://openaccess.thecvf.com/content/ICCV2025/papers/Arabi_SEAL_Semantic_Aware_Image_Watermarking_ICCV_2025_paper.pdf)和[补充材料](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Arabi_SEAL_Semantic_Aware_ICCV_2025_supplemental.pdf)。PGID 的 NeurIPS2026 接收信息目前依据作者仓库，和已取得正式论文集的文献区分。

| 条件 | 在三篇中的实际位置 | 发表资格与原作者实现 | 预算及执行状态 |
|---|---|---|---|
| Imprint-F | SemBind §4.3、PGID 主实验 | [CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.html)；[semantic-forgery](https://github.com/and-mill/semantic-forgery/tree/ca68950cd7b53b8438a49580f134b2a0db28d778) | 完整作者循环151次实际更新，保存索引50/100/150；SD2.1/SD1.5分别报告。已有小样本不计入新主表。 |
| Imprint-R | PGID 主实验 | 同上，原 `run_imprint_removal.py` | 同样保留完整更新，不成功即停或减少预算。 |
| Reprompt | SemBind §4.3 | 同一 CVPR2025 论文和原 `run_reprompting.py` | 原作者50步反演和生成；原提示词及预定代理，完整预算。 |
| Reprompt+ | CVPR2025 原论文扩展；作为统一补充条件 | 作者 [README](https://github.com/and-mill/semantic-forgery/blob/ca68950cd7b53b8438a49580f134b2a0db28d778/README.md)给出协议，**没有独立的完整入口** | GS：3提示词×3次原生重采样；TR：3提示词。项目只编排原作者单候选函数，明确这项编排差异；不扩展至 PRC，不修改重采样布局。 |
| 使用目标模型权重的重新生成伪造 | SEAL 正文 §5/Table2 | ICCV2025；原 [regeneration.py](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/regeneration.py) | 目标模型反演与异提示词生成；单列权重可访问条件，不能混入仅代理模型的 Reprompt。 |
| Cat Attack | SEAL §4、§5、补充§13.1 | ICCV2025；原 [cat_attack.py](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/cat_attack.py) | 正文随机对象比例0.3–0.6；附录分档另列。采用原对象、粘贴及位置规则，不新增对象搜索或优化。 |
| Averaging / Steganalysis removal | PGID 附录E；SEAL 补充Table3 | [NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/67b2e2e895380fa6acd537c2894e490e-Abstract-Conference.html)；[showlab 原码](https://github.com/showlab/watermark-steganalysis) | 保留5/10/20/50/100/200/500/1000/2000/5000的公布预算；5000对配对图的灰盒条件与非配对黑盒条件分开。作者 CLI 有参数调用不匹配，需通过直接调用未修改的作者函数完成入口验收。 |
| Regeneration removal | SEAL 正文§5、补充§13.3 | [NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/10272bfd0371ef960ec557ed6c866058-Abstract-Conference.html)；[WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker) | 与 SEAL 的 `regeneration.py` 伪造任务不同。原攻击配置、实际去噪预算与 SEAL 表格的精确对应尚待核清；未冻结前不执行。 |
| WMCopier | **SemBind 正式版附录G/Table24** | [NeurIPS2025 论文](https://openreview.net/pdf/27c00eec6f2cdb6052b2b199df682c0b65efabf8.pdf)；[原作者代码](https://github.com/holdrain/WMCopier) | SemBind 报告1000张辅助图、10000训练步、100张测试图。需进一步核对作者完整训练/推理配置，训练与测试来源隔离，不能用更短训练代替。 |
| VAE / Latent Forgery（Jain 等） | PGID 附录E；SEAL 补充§11 | 已核实[预印本](https://arxiv.org/abs/2504.20111)及作者代码；尚未核实顶会/顶刊正式发表记录 | **待发表资格核验，暂不执行**。被防御论文引用不等于满足用户的攻击来源限制。 |
| 代理 masker 与像素覆盖自适应攻击 | SemBind 正式版附录E | 在正式版中确有实验 | **按用户要求排除所有自适应攻击**；说明范围限制，不以“全攻击覆盖”掩盖此项。 |

SEAL 引用的早期题名 “Steganalysis on Digital Watermarking” 与 NeurIPS2024 的 “Can Simple Averaging Defeat Modern Watermarks?” 可由作者仓库关联到同一 arXiv 编号2406.09026。Regeneration removal 的正式会议年份为2024，不能照抄 SEAL 参考文献中的2025年份。

## 共同失真和原生失真

共同条件使用 [ICML2024 WAVES](https://proceedings.mlr.press/v235/an24a.html) 原函数，固定 commit `1477635d306d6f8c77a588e6441bd4ba301a1f7a` 和公布强度，不按结果搜索强度。覆盖旋转、裁剪缩放、擦除、亮度、对比度、模糊、噪声和压缩。主条件及各强度在运行前冻结。

SemBind 原文还含 JPEG70、亮度factor1、GaussianBlur radius3、噪声sigma0.01、MedianFilter7、缩放0.5；SEAL 原文列旋转75度、JPEG25、裁剪0.75、模糊8、噪声0.1、亮度6。保留原生复现附录，明确原函数的实际语义：SEAL 的噪声函数存在共享标量/uint8行为，不能标为独立高斯噪声；SemBind 的 factor1不凭名称视为有效扰动。

## 检测分数与校准接口

| 路径 | 可用于统一 FPR 的对象 | 校准限制 |
|---|---|---|
| GS | 相对已声明身份的消息匹配数/256 | 严格/非严格边界固定；离散点不能精确达到目标时取保守点，并报告实际 FPR。原 BA>181/256 另列。 |
| TR | 原作者 p-value 或原实现连续匹配分数 | 固定分数方向、归一化及比较符；不同实现的数值阈值不能互搬。 |
| PRC Detect | 原 Detect 统计量及原有 `false_positive_rate` 参数 | 参数只是名义水平，须用自然图经验校准；导出统计量前与原布尔函数逐条验收。 |
| PRC Decode / Detect OR Decode | 原解码成功事件及联合规则 | 解码通过可能形成 FPR 下限。若无法达到目标，标明不可达，不悄悄删掉 Decode 分支；精确消息恢复单列。 |
| SEAL semantic | 固定每块距离2.3下的匹配块数 | 整数阈值、独立自然负例验证；阈值12不能直接叫经验1e-6。 |
| SEAL spatial | 原空间热图、百分位和连通块分数 | 与语义验真区分；空间检测的 AUC 不能充当语义检测器的固定 FPR 结果。 |
| 原生载体 AND 登记SSCD | 原生分数与逐图登记相似度的完整联合规则 | 需要逐图存储，定义记录已知或全库检索；不得在攻击集上择阈值，也不称理论上限。 |

自然负例规划为两个各20000图的来源隔离集合；COCO官方test2017作为当前候选资产，必须先完成下载、解压校验、图像有效性及重复审计。开发耗时的8张旧自然图不进入这些集合。主表500个来源的队列还需与全部历史开发来源核查重叠。未满足这些条件时，工作点保持“未校准”。
