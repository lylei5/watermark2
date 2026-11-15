# 扩散模型水印防御研究

本项目研究扩散水印的内容绑定防御，同时建立可复核的SemBind、PGID、SEAL原生基线。[实时状态](project_state.json)记录运行队列；[验收契约](docs/THREE_BASELINE_REPRODUCTION.md)规定实现和数值复现的范围。

当前原型研究[随编辑前向变换的密钥恢复](docs/TRANSPORTED_KEY_CAPSULE.md)：用查询中的共享编辑关系同时约束辅助记录与密钥基函数，在剩余可用信息中恢复128位密钥。原生GS/TR/PRC载体保持原样。[成立条件](docs/FORWARD_RECOVERY_CONDITIONS.md)与[查新](docs/RECOVERY_MECHANISM_PRIOR_ART.md)分别说明数学范围和直接先例。

[完整开发攻击实验](reports/transported_capsule_observability_20261115/report.md)完成5050次解码、1158次材料解封及317次原生检测。主设置在TR合法95/100时，两代理Reprompt+均0/10；同通过数的单参考SSCD各1/10，九模板SSCD各0/10。直接参考图前向拟合也达到0/10，当前判别收益主要来自编辑关系。

[100实例来源迁移](reports/capsule_source_transfer_20261115_v3/report.md)完成2600次查询解码、100次登记核验，并重新执行1300图的SSCD和CopyNCE。固定阈值下，主设置合法894/1000、同提示词不同实例0/100；单参考SSCD为899/1000、0/100。主设置准确密钥覆盖963/1000，噪声组100/100可恢复但残差后仅45/100通过。跨来源合法覆盖、未见编辑和密钥层的独立贡献仍是研究重点。更早的失败和完整逐样本证据保留在[报告索引](reports/README.md)。

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
