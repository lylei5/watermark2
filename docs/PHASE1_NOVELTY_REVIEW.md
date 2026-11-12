# 轨迹语义方向的贡献边界

截至2026-11-13，单纯将水印放到去噪中后期，或从同一生成过程提取内容，不能作为独立创新点。当前100提示词实验是稳定性与区分力测量；它没有实施轨迹内绑定，也没有测得防御增益。

| 一手来源 | 已有机制与当前判断 |
|---|---|
| [ROBIN，NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/073c8584ef86bee26fe9d639ec648e28-Abstract.html) | 在中间扩散状态植入水印，再用优化的提示引导隐藏水印。中途植入及后续保真引导已有先例。 |
| [AEON，WACV2026](https://openaccess.thecvf.com/content/WACV2026/html/Muneer_AEON_Adaptive_Embedding_Optimized_Noise_for_Robust_Watermarking_in_Diffusion_WACV_2026_paper.html) | 论文集摘要明确中间过程中的语义哈希适配、后期嵌入及一致性优化。当前仅核对出版记录和摘要，不据此宣称逐式等价或复现其数值。 |
| [IRIS v1](https://arxiv.org/html/2608.03539v1)，[当前状态](https://arxiv.org/abs/2608.03539) | 初稿从未加水印的轨迹终点读取内容，再返回同一轨迹后期嵌入内容派生环。2026-08-26的v2已撤回，注明实质修订；只作为公开思路的查新线索，不能作为经验证的性能证据或顶会基线。 |
| [Guidance Watermarking，ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b9501301a7a0f99e57406cc901f4e3ce-Abstract-Conference.html) | 用已有解码器梯度和增强引导生成；简单增加一致性损失或多增强引导不能单独构成新机制。 |
| [SemBind，ICML2026](https://proceedings.mlr.press/v306/zhang26hj.html) | 训练目标使同提示词图像获得相近码。轨迹内稳定与实例区分是两回事；须同时测量同提示词不同随机种子。 |

## 对方向A的判定要求

第35/50步P5及配对优势只判断必要条件。即使通过，也不能把相对较低的跨实例码一致率直接视为拒绝能力：原GS的重复译码本来就会纠正一部分符号差异，语义距离未必转化为检测差异。

若要立项，贡献必须落到一个具体而可证伪的问题：在中途写入会改变后续图像的条件下，如何保持内容绑定的自洽，并让合法编辑与不同实例在同一检测工作点上分离。需要检验写入前后码漂移、正常图绝对接受、已发表完整攻击接受及计算成本。仅有正常轨迹的高一致率、节约一次生成或替换语义网络，均不足以完成该论证。

当前不创建新的载体或攻击变体。方法推进由冻结的测量结果、完整基线及上述查新边界共同决定；不能为了获得好结果而放宽已定判据。
