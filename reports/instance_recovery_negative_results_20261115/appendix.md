# 实例恢复负结果的附录证据

以下链接保留各运行的原始报告、协议和逐样本产物。此目录整理历史证据，不重新生成数据，不改变原有阈值或方法选择。

| 证据 | 完整材料 | 统计与实现边界 |
|---|---|---|
| 部分序恢复 | [报告](../order_capsule_confirmation_20261115/report.md)、[完成记录](../order_capsule_confirmation_20261115/complete.json) | 100 个既有来源，3800 配对，三设置共 11400 次解码；主设置预先固定。最新运行 native_calls=0，载荷为身份和像素哈希 |
| 部分序完整开发攻击 | [报告](../order_capsule_development_20261115/report.md) | 1010 配对、3030 次解码；TR 合法 99/100，Reprompt+ 两代理 0/10、1/10；同合法直接参考各 0/10，九模板 SSCD 为 2/10、3/10 |
| 饱和白坐标作用 | [报告](../order_saturation_ablation_20261115/report.md) | 确认后的配对诊断；400 次解码，不是独立确认 |
| 前向内容封装 | [报告](../transported_capsule_observability_20261115/report.md)、[100 来源迁移](../capsule_source_transfer_20261115_v3/report.md) | 区分正确密钥返回与后置残差拒绝；已知来源的不同条件并非独立样本 |
| 共同频谱 | [报告](../shared_channel_binding_20261114/report.md)、[结果表](../shared_channel_binding_20261114/result_tables.md) | 17730 次恢复判定包含 690 次内部配对消融；同一来源三组投影不是三组独立数据 |
| 空间证据 | [报告](../spatial_witness_recovery_20261114/report.md) | 区域隔离 TR 合法 68/100；270 项匹配比较中 2 项优于 CopyNCE，但同合法 SSCD 同样拒绝相关攻击 |
| 可见性证据 | [报告](../visible_witness_recovery_20261114/report.md) | TR 合法 99/100，basic Reprompt 两代理接受 8/10、9/10；准确恢复不能替代来源正确性 |
| 多尺度完整候选预算 | [报告](../reprompt_plus_recovery_20261115/report.md)、[匹配比较](../reprompt_plus_recovery_20261115/matched.json)、[完成记录](../reprompt_plus_recovery_20261115/complete.json) | 240 张原作者候选，来源数为 GS/TR 各 10；主要 GS 内容比较只含 3 个评估来源；未标注目标提示词实现程度 |
| 强对照 | [SSCD 与原生迁移](../recovered_carrier_transfer_20261113/report.md)、[CopyNCE](../copynce_control_20261114/report.md) | CopyNCE 匹配器读取完整参考图；SSCD 单参考和模板的存储、阈值分别保留 |
| 方法与先例 | [部分序关系](../../docs/ORDER_CONSTRAINED_RECOVERY.md)、[实例恢复机制](../../docs/INSTANCE_BINDING_RESEARCH.md)、[内容密钥可行性](../../docs/CONTENT_KEY_FEASIBILITY.md) | 顺序匹配、纠错、模糊提取器、编辑估计与分布式信源认证均有先例；本项目不将其重新命名为创新 |

## 解释范围

相同合法计数比较中的阈值在评估合法图上确定，属于描述性权衡分析，不能冒充独立自然负例校准的固定 FPR 结果。一个来源的多次编辑、候选和检查点必须作为同一个统计簇处理。不同报告的样本和校准规则不同，不汇总成单一总体准确率。

直接参考可以模拟恢复器的结论以相同信息和权限为前提，不推出无状态系统与有状态参考系统等价。当前尚无证据支持恢复层具有独立存储、计算或秘密访问收益。未公开模型权重、图像、私钥、辅助记录内容和原生秘密材料；完整私有资产留在 249。

## 冻结范围

实例密钥恢复、部分序恢复、前向密钥封装、共同频谱恢复、空间/可见性/多尺度证据恢复不再开展新变体或参数搜索。允许为论文归档进行只读审计、已有产物校验和必要的结果整理；若未来恢复此方向，需要新的明确研究任务。三种防御的基线实验不依赖这些恢复器。
