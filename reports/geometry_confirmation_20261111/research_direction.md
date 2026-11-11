# 从语义绑定到编辑后认证：贡献边界与下一项检验

目前可被数据支持的机制是：固定的强语义绑定在合法裁剪后可能仍保留身份信息，但坐标失配使默认译码无法使用这些信息。查询驱动的坐标恢复，在不改变生成图像和默认身份判定基准的情况下，恢复了新来源的合法接受。校准集上，简单降低 BA 阈值不能达到同样的合法接受与错误身份结果。

这尚不足以宣称新的先进防御。校准时两个几何判定方案完全等效；独立确认全组通过382/384、分组381/384。全组是预先固定的主要方法，分组结果尚无收益，不增加额外结构。

## 与已有防御的关系

| 工作 | 已有机制 | 本项目必须回答的问题 |
|---|---|---|
| [SemBind，ICML 2026](https://proceedings.mlr.press/v306/zhang26hj.html) | 语义掩码绑定潜变量水印；通过掩码比例调节鲁棒性与防伪权衡 | 保持同一绑定设置和图像，验证端能否恢复合法编辑的证据？ |
| [GaussMarker，ICML 2025](https://proceedings.mlr.press/v267/li25ae.html) | 空间和频率双域水印，以及可学习的 Gaussian Noise Restorer | 潜变量恢复本身已有先例；需明确原型无需新嵌入、恢复器训练或参考图像的信息条件，并做原生方法比较。 |
| [ROAR，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_ROAR_Reducing_Inversion_Error_in_Generative_Image_Watermarking_ICCV_2025_paper.html) | 再生成优化降低潜变量域反演误差，MoE 恢复网络处理像素域失真 | 降低反演误差不是新贡献；应分离坐标对应误差、数值反演误差和语义不一致。 |
| [Gaussian Shading，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.html) | 分布保持的潜变量水印与多数译码 | 相同合法检出目标下，几何恢复是否优于放宽译码阈值，并能否经受其官方失真设置？ |

上述文献支持比较方向，不支持直接使用跨论文表格数字进行排名。GaussMarker 和 ROAR 的原生生成、恢复和计算预算需要分别复现；当前未完成这些方法的同条件比较。

## 同步与语义绑定的直接先例

[SynTag（ICCV2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_SynTag_Enhancing_the_Geometric_Robustness_of_Inversion-based_Generative_Image_Watermarking_ICCV_2025_paper.pdf)通过VAE解码器微调嵌入同步特征，用预测网络恢复几何关系；因此“校正后反演”不是新问题。[SyncSeal全文](https://arxiv.org/html/2509.15208v1)也用额外同步水印和几何预测提升已有水印，并明确区分保持原像素位置的填黑裁剪与真正改变坐标的裁剪，指出穷举同步的计算及误报代价。

本项目当前信息条件是：已有生成图像保持不变；检测只使用查询图、声明身份和密钥；一次查询反演后，在潜变量中恢复对应关系并校正搜索代价。这个条件差异值得检验，但不是优越性证据。不能用项目251个固定裁剪姿态的结果代替这些方法覆盖的旋转、透视等广泛变换。

[SEAL（ICCV2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Arabi_SEAL_Semantic_Aware_Image_Watermarking_ICCV_2025_paper.pdf)已经通过语义嵌入、局部敏感哈希和密钥化噪声实现内容相关水印。内容认证和分块匹配本身也不能作为新贡献。IRIS的[arXiv当前记录](https://arxiv.org/abs/2608.03539)仍标记撤回；仅记录概念相关性，不作为有效性已确认的基线，也不引入其中的攻击。

当前可证伪的研究命题为：**保持强内容绑定时，合法编辑引入的坐标失配可以与内容不一致区分，从而在不放宽身份接受门限的情况下恢复部分合法认证证据。** 六条件确认支持其中的裁剪恢复部分；官方九条件结果检验范围边界。额外拒绝伪造、跨模型推广和原生方法比较仍须独立证据，不能用这个命题预先替代实验结论。

## 已冻结的独立检验

64 个来源、384 张现有视图的确认已完成，保留全部来源与失真分母。主要方法、四个 BA 校准工作点和分组次要对照均在评分前固定。全组通过382/384，默认319/384，两者均0/24,192错误身份接受，详见[确认报告](report.md)。此轮检验同条件的新来源推广，不把它称作新失真或跨模型验证；确认集已经使用。

## 官方失真实现审计

Gaussian Shading 官方 `image_distortion` 的 `random_crop_ratio` 保留随机窗口并将窗口外区域填黑；当前项目的 `crop90` 则裁剪后放大。这两个条件不能合并命名或互相替代。官方 `run.sh` 固定九项参数：crop 0.6、drop 0.8、brightness 6、JPEG25、Gaussian blur4、Gaussian std0.05、median7、salt-and-pepper0.05、resize0.25。crop/drop 参数按上游代码作用于边长，不应改写成同数值的面积比例。

已启动对同一组 64 个来源分别调用九项上游函数，固定提交 `09c678fadc7545acf7be12647ddf2a5e66f6a9dc`，不修改函数或强度；预计 576 张查询及 36,864 次身份判决。该轮和六条件确认分别报告，并注明共享来源。来源：[官方实现](https://github.com/bsmhmmlf/Gaussian-Shading/blob/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/image_utils.py)、[官方参数](https://github.com/bsmhmmlf/Gaussian-Shading/blob/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/scripts/run.sh)。

只有在更完整的既定条件下保持收益，并排除更强简单对照、验证错误身份与已发表伪造基线后，才考虑防御优越性主张。若收益局限于当前裁剪后放大条件，应把结果定位为认证失效的机制与修复证据。
