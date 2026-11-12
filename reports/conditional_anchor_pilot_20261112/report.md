# Conditional-anchor defense pilot

All240 genuine-query scores independently recomputed. Six new neutral prompts, four embedding arms, ten conditions, full50-step generation/inversion and100-step decoder optimization.

| Arm | Accepted | Mean BA | Binding mismatch | LPIPS to auxiliary |
|---|---:|---:|---:|---:|
| full | 53/60 | 0.882292 | 0.095052 | 0.725946 |
| anchor_low | 53/60 | 0.873437 | 0.058757 | 0.370816 |
| independent_low | 54/60 | 0.863346 | 0.080241 | 0.766647 |
| anchor_random | 49/60 | 0.873177 | 0.063802 | 0.666053 |

Prespecified feasibility gates: {'binding_vs_independent': True, 'binding_vs_random': False, 'genuine_retention': False, 'reference_fidelity': True}. Advance: False.
LPIPS is a paired auxiliary-image fidelity diagnostic, not FID or a general quality guarantee. Full carrier has64 repetitions; partitioned arms have32. Only the partitioned controls isolate anchor sharing.

## 机制诊断与研究决定

共享锚未通过四项冻结门槛，因此未启动该方案的Imprint扩展。不得把绑定差异与LPIPS改善表述为整体防御优势。

服务器保存的240条潜变量进行事后诊断：

| 载体 | 原判定 | 登记语义码诊断 | 普通线性幅值加权 |
|---|---:|---:|---:|
| full | 53/60 | 59/60 | 57/60 |
| anchor_low | 53/60 | 59/60 | 56/60 |
| independent_low | 54/60 | 58/60 | 55/60 |
| anchor_random | 49/60 | 57/60 | 53/60 |

登记码仅用于定位绑定相关损失，不是可部署检测器。幅值加权只是已有思路的诊断对照，不是GS++完整复现，也未使共享锚超过同样加权的full。完整逐条记录见[channel_diagnostic.json](channel_diagnostic.json)。这项诊断没有改变原实验门槛或最终结论。

后续已完成的[内容耦合分位实验](../conditional_quantile_pilot_20261112/report.md)记录相同完整容量下的耦合与独立采样对照。

按源图检查，共享低频锚相对独立锚在4/6来源上减小绑定差异，在6/6来源上降低配对LPIPS；第5号来源的绑定差异反而增加0.040039，真实接受数由9/10降到7/10。均值改善不能解释为所有来源均获益。见[source_mechanism.json](source_mechanism.json)。
