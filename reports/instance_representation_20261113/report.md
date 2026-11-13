# 发布实例表征诊断

固定100个已有开发提示词，对比 SemBind 码、DINOv2 连续特征和 SSCD 拷贝描述子。输入、模型、源码、提示词族拆分及阈值选择规则见[冻结协议](../../protocols/instance_representation_20261113_v2.json)。机制和相关工作见[研究设计](../../docs/INSTANCE_BINDING_RESEARCH.md)。

每个来源包含辅助图、同提示词独立seed图、原GS图、GS-S最终图，以及最终图的九种独立GS官方处理，共1300张输入/表征。全部推理与统计在249执行，GPU7。当前完成状态以project_state.json和服务器完成产物为准；本报告将在评分审计完成后记入结果。

首次启动在第一张图的处理种子初始化处失败：种子超出NumPy的32位范围，未产生特征产物。新目录v2将种子基数设为2026111300，其余输入和预声明比较保持相同。原始[失败记录](initial_execution_failure.json)与服务器冻结目录保留。
