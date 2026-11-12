# SEAL原生语义检测契约

本契约以[ICCV2025最终论文](https://openaccess.thecvf.com/content/ICCV2025/html/Arabi_SEAL_Semantic_Aware_Image_Watermarking_ICCV_2025_paper.html)、[最终补充材料](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Arabi_SEAL_Semantic_Aware_ICCV_2025_supplemental.pdf)及[官方固定源码](https://github.com/Kasraarabi/SEAL/tree/92d31b31b93a6e373fe88a584c640beafb68c6fa)为依据。PDF已取得并阅读，文件哈希见[收据](../reports/three_baselines_20261113/paper_receipts.json)。动态状态只维护于project_state.json。

## 语义链路

1. 按prompt生成预览图，使用Salesforce/blip2-flan-t5-xl描述图像。
2. 作者kasraarabi/finetuned-caption-embedding编码caption并作L2归一化。
3. 使用官方generate_initial_noise，k=1024、b=7、公开代码投影seed=42生成噪声，再按同prompt生成合法水印图。
4. 对待测图重新caption和编码，重新生成参考噪声；使用VAE mode及官方forward_diffusion反演该待测图。
5. 每块为4×2×2个潜变量，采用未归一化欧氏距离。最终算法3与补充§9.1规定：d<2.3的块数至少12则接受。

[官方SEAL.py](https://github.com/Kasraarabi/SEAL/blob/92d31b31b93a6e373fe88a584c640beafb68c6fa/SEAL.py)只累计距离，没有接入最终标签。项目入口仅补判定和保存产物，不重新实现其数值函数。原始`--threshold=50`是未使用参数，不能拿来替代论文规则。

## 数值约定和已知差异

生成沿用模型自带PNDMScheduler，默认请求50步、CFG7.5、FP16；反演用官方backward_ddim的逆向调用，空prompt、CFG1、请求50步及已加载调度器的时间步。准确的实际时间步和模型快照应随产物保存，不能写成生成/反演均为原生DDIM。

公开函数逐块重置随机种子，caption投影和噪声生成必须保留设备、RNG与依赖。论文算法2含secret salt；公开helper没有对应独立参数。本项目保留公开算法并明确该差异，不擅自加入salt后宣称原码一致。

249已完成独立环境导入、全部SEAL权重哈希与BLIP2/作者embedding的上游LFS匹配，见[来源对照](../reports/three_baselines_20261113/source_equivalence.json)。实际环境为torch2.5.1、diffusers0.20.2、transformers4.47.1、sentence-transformers3.3.1；并非作者完整依赖锁。SD2.1镜像固定于0094d483a120f3f33dafbd187ea4aa60d10de75c；与原stabilityai fp16快照的同一性仍需证据。

## 空间检测和评估范围

无caption的穷举码字检测是另一条路径，不能替代上述语义检测。补充§9.2的空间篡改分数使用热图80百分位和连通块数量，AUC不等于一个已经校准的固定检测器。图像扰动入口中的旋转75度、JPEG25、裁剪0.75、模糊8、噪声0.1和亮度6可以按官方函数独立验证。

[原生100提示词协议](../protocols/seal_native_20261113_v3.json)覆盖正常语义生成与无水印对照。模型导入、权重匹配、正常链路、良性鲁棒性、完整既有攻击及论文数值复现是分别验收的层次。阈值12/2.3不能直接称为FPR1e-6；也不能用原GS图、GS比特准确率或穷举通过率冒充SEAL语义结果。

## 最终论文的私有salt路径

`seal_paper_simhash.py`逐项实现最终Algorithm2：用hash(i,j,salt)采样每个投影向量，再用hash(bits,i,salt)采样对应噪声块。盐为私有32字节，没有公开默认值。论文未指定序列化与具体Gaussian PRNG；这里明确使用带域前缀的SHA256、固定宽度整数编码、摘要前64位作为设备内私有PyTorch Generator的seed。这是论文规格实现，不能声明与作者未公开实现逐字节相同。

[CPU组件检查](../reports/three_baselines_20261113/seal_paper_simhash_checks.json)已验证同盐同输入精确复现、独立对象重放、换盐改变投影与噪声、拒绝空盐、保持全局RNG状态。没有执行攻击或形成图像性能结论。[独立100提示词协议](../protocols/seal_paper_native_20261113.json)与公开代码对照使用同一批预览图，并在运行中核对预览PNG哈希；两条结果分别报告。

作者embedding本地资产原缺少仅记载版本、空prompts和cosine设置的config_sentence_transformers.json。[完整配置对照](../reports/three_baselines_20261113/configuration_audit.json)已校验现有文件的上游Git blob。另建包含该原始配置的固定目录，未改变已冻结运行的资产；249上3个文本的768维CPU编码逐值一致，最大差0，见[验证](../reports/three_baselines_20261113/seal_embedding_config_check.json)。该检查不替代图像验证。
