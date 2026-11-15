# SemBind与PGID共同自然图完整提取队列

**完整负例提取已排队，等待SEAL合法图评估正常结束；尚未执行本批GPU提取，也没有本批校准FPR结果。** 冻结输入为同一40000图清单，20000校准、20000验证。预定得到160000份特征：SemBind40000份，PGID默认反演、F、R各40000份。所有数值实验只在249执行。

| 路径 | 每图完整预算 | 本批适用范围 |
|---|---|---|
| SemBind原生反演与语义编码 | 50步反演、100次VAE优化，CFG3；发布的语义网络 | 缓存原始噪声与语义码，后续按冻结GS/TR/PRC载体和身份评分 |
| PGID默认反演 | 50次UNet调用 | GS/TR的DDIM对照 |
| PGID-F | k15、s3、gamma0.001，完整298次UNet调用 | GS/TR固定F配置 |
| PGID-R | k10、s1、gamma0.045，完整200次UNet调用 | GS/TR固定R配置 |

PRC的PGID-F为k7，且原作者PRC管线另有DPM配置。这批PGID特征不冒充该专用路径；PRC专用提取及共同载体适配仍需另行验收。缓存特征可以省去同一提取器对多载体身份的重复推理，但不意味着不同防御可共用同一噪声结果。

## 执行和核验

前驱使用进程退出事件，必须检查SEAL合法图评估的正常完成记录。之后检查三张空闲授权GPU，按UUID绑定，排除GPU6；最多三个工作进程。先在既有八张开发图上分别运行SemBind及PGID三种配置，与历史原始特征逐项比较：噪声容差atol/rtol均1e-5，语义码要求完全相同。这个GPU数值核验尚待执行；任何不通过都会阻止自然图批次启动。

自然图按100张一批，共800个方法批次；这里的分批只是恢复和调度边界，仍然处理全部40000图。每张先核对原文件哈希，再执行共同512×512预处理并核对像素哈希。存储反演结果、语义码、来源/划分、NFE、耗时、显存和产物哈希。已完成记录支持哈希核验后的复用；失败保留记录并停止启动新的批次，不因分数删换图片。

CPU预检已验证两个脚本可编译、PGID管线接口存在、八张自然图共同预处理的像素一致、完整PGID循环调用数一致，CUDA未初始化。依赖、模型文件与缓存revision已固定；运行前后校验模型哈希。既有已完成实验继续使用各自冻结源码，不受公共入口增加分批/预处理/计数功能影响。

## 成本与结论边界

依据[八图完整耗时预检](../unified_benchmark_preflight_20261115/report.md)，SemBind每图44.59秒，40000图约495GPU小时；PGID默认/F/R合计每图39.31秒，约437GPU小时。两部分合计约932GPU小时，三卡持续运行约13天量级，尚不含其他载体专用配置、攻击、等待和额外开销。这是小样本耗时外推，实际进度以执行记录为准。预算保持完整；自然图提取不读取或优化最终攻击结果。

本批只提取特征。完整校准还需冻结原载体实现、身份与解码规则，随后只在校准划分拟合阈值、在验证划分报告实际FPR及区间。没有阈值、合法检出与攻击结果之前，不填写防御优劣排名。

协议：[unified_null_features_20261116.json](../../protocols/unified_null_features_20261116.json)，SHA256 `e0ebe9aad974a310eac591a915b946a4eab997ffa4883d2a80bf463a7b078d22`。

服务器运行目录：`/data0/liyanlei/watermark_research/runs/unified_null_features_20261116`。完成后生成`feature_records.jsonl`和`feature_extraction_complete.json`。可核查[入队记录](launch.json)、[入队时状态](pipeline_state.json)、[CPU预检](null_feature_preflight.json)、[模型哈希](model_files.json)、[协调器](../../scripts/defense/queue_unified_null_features.py)和[完整提取入口](../../scripts/defense/unified_runtime_probe.py)。
