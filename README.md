# 扩散模型水印防御研究

本项目研究扩散水印的内容绑定防御，同时建立可复核的SemBind、PGID、SEAL原生基线。[实时状态](project_state.json)记录运行队列；[验收契约](docs/THREE_BASELINE_REPRODUCTION.md)规定实现和数值复现的范围。

当前原型研究[部分序约束的密钥恢复](docs/ORDER_CONSTRAINED_RECOVERY.md)：查询提供位置之间的亮度关系，恢复器用这些关系求解128位随机密钥，确认并实际解封认证载荷。返回流程没有后置图像残差门槛；原生GS/TR/PRC载体保持原样。

[100实例与未见编辑验证](reports/order_capsule_confirmation_20261115/report.md)完成11400次部分序解码及4599次SSCD图像推理。已知条件合法图实际返回993/1000，噪声95/100、六倍亮度98/100；旧流程最终为894/1000、45/100、70/100。旧噪声密钥本已100/100正确，因此噪声最终通过的改善需与移除残差门槛的效果分开理解。同合法993/1000时，九模板SSCD误收同提示词其他实例18/100，部分序恢复和直接部分序参考拟合均为0/100。

此前未执行的WAVES对比度、旋转、裁剪后缩放各五档强度，部分序恢复分别500/500、3/500、18/500；九模板SSCD均500/500。来源图曾用于项目，留出的是编辑类型。[饱和坐标消融](reports/order_saturation_ablation_20261115/report.md)实际完成400次解码，删除纯白坐标没有改变任何配对判定；亮度收益对应对固定区域亮度拟合限制的绕过。空间对应和恢复层相对直接参考的独立价值仍是研究重点。

[完整开发攻击实验](reports/order_capsule_development_20261115/report.md)保留原作者Imprint、basic Reprompt、Reprompt+输出。主设置TR合法99/100，Reprompt+两代理接受0/10、1/10来源；同合法九模板SSCD为2/10、3/10，直接参考拟合为0/10、0/10。本轮新编辑只验证内容绑定与密钥载荷，未新增原生水印检测。完整失败证据保留在[报告索引](reports/README.md)。

已发表攻击覆盖包括[原作者Reprompt+](reports/reprompt_plus_recovery_20261115/report.md)的GS/TR两代理、各10来源、9/3候选预算，共240张图与480条唯一原生/PGID判定。TR的PGID合法60/100，Reprompt+各接受8/10来源；[basic Reprompt](reports/author_reprompt_control_20261114/report.md)另含PRC。原论文大样本多目标矩阵与三个防御的完整共同工作点比较继续单独验收。

[2026-11-13研究汇报](presentations/defense_briefing_20261113/README.md)保存当时的演示文稿与讲稿，当前研究结果以上述报告为准。

## 三个防御基线

| 基线 | 实现依据 | 验证与差异记录 |
|---|---|---|
| SemBind | 官方语义网络和DPM生成/精确反演；按最终论文补齐平铺、比例选择、秘密置换和符号绑定 | 原GS OTP空间重复的GS-S；PRC-S采用sigma0.5。作者未发布完整绑定入口，论文补齐部分明确标注。[规格](docs/THREE_BASELINE_REPRODUCTION.md) |
| PGID | 固定作者`dc4ad96`，原生PGID-F/R及载体判定 | GS的F/R及TR/PRC的F各10例完整151次更新；TR/PRC合法图对照完成。GS、TR、PRC、T2SMark原码CPU往返均通过。[结果与范围](docs/DEFENSE_BASELINE_INVENTORY.md) |
| SEAL | 固定作者`92d31b3`及ICCV2025最终正文、补充材料 | 原公开seed42路径与论文secret-salt路径分别实现、分别验证；距离阈值2.3、匹配块数12。[契约](docs/SEAL_EVALUATION_CONTRACT.md) |

源码版本、模型哈希、依赖和组件检查见[来源报告](reports/three_baselines_20261113/report.md)。模型导入、组件通过、正常图通过和完整论文数值复现分别判断，不用排队任务或少量样本替代后者。

[实现差异表](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)集中列明作者原码、论文补齐逻辑、内存适配、密钥约定和未验收项。

已验收的正常图结果：GS-S为100/100；PRC-S、TR-S各6/6，无水印各0/6；SEAL公开代码为83/100、论文私有salt实现为91/100，对应无水印对照均为0/100。严格后端PRC-S重放同样6/6。这些是开发数据上的原生设置结果，不是同工作点排名或1e-6经验误报保证。

## 实现入口

| 功能 | 入口 |
|---|---|
| 部分序密钥恢复 | [order_constrained_capsule.py](scripts/defense/order_constrained_capsule.py)，真实128位求解、确认和认证解封 |
| 未见编辑与强对照 | [run_order_recovery.py](scripts/defense/run_order_recovery.py)、[审计](scripts/defense/audit_order_confirmation.py)，WAVES原码、SSCD九模板、两类直接参考拟合 |
| 饱和贡献与恢复机制核验 | [audit_order_saturation.py](scripts/defense/audit_order_saturation.py)、[机制汇总](scripts/defense/summarize_order_mechanism.py)，同记录配对、失败标签、存储与全强度 |
| 前向密钥封装与恢复 | [transported_key_capsule.py](scripts/defense/transported_key_capsule.py)，同编辑传输、可用坐标求解、认证解封 |
| 完整攻击与强对照 | [run_transported_capsule.py](scripts/defense/run_transported_capsule.py)、[汇总](scripts/defense/summarize_transported_capsule.py)，冻结校准与同通过数比较 |
| 来源迁移与新模型推理 | [validate_capsule_source_transfer.py](scripts/defense/validate_capsule_source_transfer.py)，固定阈值、原登记复用、同像素SSCD/CopyNCE |
| SemBind语义码 | [semantic_code.py](scripts/defense/semantic_code.py) |
| SemBind论文掩码 | [sembind_binding.py](scripts/defense/sembind_binding.py)，私有密钥；sigma为0至1 |
| 原GS/GS-S图像闭环 | [run_phase1_probe_worker.py](scripts/defense/run_phase1_probe_worker.py)，冻结100提示词协议 |
| PRC/PRC-S图像闭环 | [run_sembind_prc_native.py](scripts/defense/run_sembind_prc_native.py) |
| SEAL完整语义链路 | [run_seal_native.py](scripts/defense/run_seal_native.py)；论文salt见[seal_paper_simhash.py](scripts/defense/seal_paper_simhash.py) |
| 作者PGID完整预算入口 | [run_pgid_native.py](scripts/defense/run_pgid_native.py)，原样执行上游run.py |
| 实例密钥登记与恢复 | [instance_coset_recovery.py](scripts/defense/instance_coset_recovery.py)，认证记录、实例方向、128位真实恢复 |
| 恢复材料的原生检测 | [verify_recovered_native_keys.py](scripts/defense/verify_recovered_native_keys.py)，原作者检测器与哈希核验反演 |
| TR/PRC恢复迁移 | [recovered_carrier_transfer.py](scripts/defense/recovered_carrier_transfer.py)，冻结阈值、完整来源分母 |
| 共同频谱真实密钥恢复 | [evaluate_shared_channel.py](scripts/defense/evaluate_shared_channel.py)，两路/单路/拼接/SSCD配对比较 |
| CopyNCE公开模型 | [copynce_control.py](scripts/defense/copynce_control.py)，作者描述子与配对匹配器 |
| 作者Reprompt完整步骤 | [author_reprompt_control.py](scripts/defense/author_reprompt_control.py)，冻结两代理、50步反演与50步生成 |
| Reprompt+恢复与强对照 | [evaluate_reprompt_plus_recovery.py](scripts/defense/evaluate_reprompt_plus_recovery.py)，完整候选预算、同图联合判定与同合法通过数比较 |
| Reprompt原生与PGID验证 | [evaluate_author_reprompt.py](scripts/defense/evaluate_author_reprompt.py)，原生反演分别执行 |
| 空间与可见性密钥恢复 | [spatial_witness_recovery.py](scripts/defense/spatial_witness_recovery.py)，实际RS错误/擦除译码及材料解封 |
| 同像素强对照汇总 | [summarize_author_reprompt.py](scripts/defense/summarize_author_reprompt.py)、[compare_visible_witness.py](scripts/defense/compare_visible_witness.py)，绝对合法通过数匹配 |
| 编辑类型留出校准 | [analyze_heldout_edit_calibration.py](scripts/defense/analyze_heldout_edit_calibration.py)，登记外五类处理逐类从校准中排除 |
| 完成后逐条审计 | [summarize_native_baseline.py](scripts/defense/summarize_native_baseline.py) |

每个运行在249的`runs/<run>/`保存冻结源码、协议哈希、逐样本产物和退出状态。当前代码可继续维护；重放既有结果使用该运行的冻结版本。工作进程拒绝覆盖已有结果。

SemBind生成和反演均为50步，inverse order0、VAE decoder优化100步。GS同时记录原源码166位、理想独立公平位下167位、历史182位三个工作点。理论阈值不是经验FPR保证；原生PGID、SemBind和SEAL的不同工作点不能直接用于方法排名。

## 证据和执行约束

[完整报告索引](reports/README.md)保留已完成实验和失败结果。[内容密钥评估](docs/CONTENT_KEY_FEASIBILITY.md)记录生成后登记的协议、实例区分及内容熵问题。恢复实现覆盖冻结SSCD与内容/噪声频谱；新的神经码器训练尚未执行。

所有模型推理和CPU评分仅在SSH主机249的`/data0/liyanlei/watermark_research`执行。只使用空闲GPU7、5、4，最多三张，始终排除GPU6；以进程退出事件接续任务，不持续轮询。模型、图像和私钥留在服务器，代码与可公开证据每完成一个单元就提交并推送main。

防御评测只使用顶会顶刊已经发表的攻击及其既定实现、完整预算，不创造或改进攻击。报告真实合法检出率、错误身份/未加水印负例、伪造接受、图像质量和计算成本，保留失败的合法参考。

- [机器状态](project_state.json)、[交接说明](GPT_CONTEXT.md)、[研究计划](docs/RESEARCH_PLAN.md)
- [基线证据](docs/DEFENSE_BASELINE_INVENTORY.md)、[第三方来源](THIRD_PARTY.md)、[操作规则](AGENTS.md)
