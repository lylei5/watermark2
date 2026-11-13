# 三个防御基线的证据清单

当前基线为SemBind、PGID、SEAL。[验收契约](THREE_BASELINE_REPRODUCTION.md)区分代码实现、正常图、完整既有攻击和论文数值复现；动态状态只维护于[project_state.json](../project_state.json)。

| 基线 | 已执行证据 | 图像验收范围与限制 |
|---|---|---|
| SemBind | 发布的生成/反演/语义网络源码字节核验；[100图1000条件语义码一致性](../reports/defense_revalidation_20261111/report.md)差异0；[300次掩码解绑定](../reports/three_baselines_20261113/sembind_binding_checks.json)通过；PRC6条CPU原码往返通过 | 原GS空间重复的GS-S完成100提示词闭环，合法100/100、辅助无水印0/100；PRC-S完成6例完整图像反演，合法6/6且消息精确恢复，无水印0/6；TR-S图像验收同为合法6/6、无水印0/6。严格确定性后端PRC-S完整重放也为6/6。旧扁平GS项目适配器的攻击成绩不转记为新原生GS-S结果。完整作者绑定源码未公开，按用户确认从论文补齐。 |
| PGID | [GS的10例PGID-F完整预算](../reports/defense_revalidation_20261111/report.md)和[TR/PRC各10例完整Imprint](../reports/pgid_carrier_coverage_20261113/report.md)完成；三个检查点原生均10/10接受，PGID后0/10。新旧源码可执行AST一致；四载体CPU往返均2/2通过 | [TR/PRC合法图和封面440条判定](../reports/recovered_carrier_transfer_20261113/report.md)完成，合法TR60/100、PRC29/100，原生100/100、64/100。PGID-R完整151更新10例：各检查点原检测0/10、防御后10/10。原工作点与描述性等合法通过数比较分别记录。 |
| SEAL | [固定源码、权重及配置](../reports/three_baselines_20261113/report.md)已核验；论文Algorithm2私有salt组件检查通过 | 原公开代码与论文salt路径均完成100提示词图像验收，合法图分别83/100、91/100，无水印均0/100；两路径七类良性处理共1400条语义判定完成。候选自身caption派生参考噪声；语义检测与无caption穷举检测分开。 |

三个原生路径保持各自载体、生成/反演器及检测规则；当前没有同工作点的横向排名。原GS的空间重复、PGID的扁平重复、SEAL的分块语义噪声属于不同生成协议。若做共同条件比较，需额外冻结数据、信息权限、合法检出率和预算。

每项结果保留逐来源分母、输入输出哈希、完整更新次数和失败记录。源码索引50/100/150对应PGID发布代码的实际51/101/151更新，不能混写成相同实际预算。理论1e-6不能替代经验自然图FPR，重复同一来源的多个检查点也不是独立样本。

内容绑定强对照另含SSCD单参考、中心化、float32/int8九模板，以及[CopyNCE作者公开描述子与匹配器](../reports/copynce_control_20261114/report.md)。它们在同像素、同来源校准下与实例恢复方法比较，参考信息和存储成本单列。其他方法出处见[文献](LITERATURE.md)、[信息契约](EVALUATION_CONTRACT.md)和[完整报告索引](../reports/README.md)。
