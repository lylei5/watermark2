# 扩散模型水印防御研究

本项目研究扩散水印的内容绑定防御，同时建立可复核的SemBind、PGID、SEAL原生基线。[实时状态](project_state.json)记录运行队列；[验收契约](docs/THREE_BASELINE_REPRODUCTION.md)规定实现和数值复现的范围。

最新的[实例编辑方向恢复实验](reports/instance_edit_quotient_20261113/report.md)已跑通128位密钥恢复、原生验证材料解封和原作者检测。八方向版本在三组投影下恢复473–477/500个合法查询，不同实例误恢复均0/50；相同登记信息的多模板SSCD是主要强对照。按共同90%校准目标，八方向实际恢复为455/460/457，多模板SSCD为448，分母均500；按来源统计的不确定性与存储成本一并报告。TR/PRC完整Imprint、合法图及实际恢复迁移的执行状态见[project_state.json](project_state.json)。

[完整方法说明](docs/INSTANCE_COSET_RECOVERY.md)给出最终图登记、实例恢复区域、BCH译码和提供方认证。三轮方法实验均保留，95%校准目标下的实际恢复能力仍待提高；多模板SSCD在该工作点为476/500、不同实例0/50。先前的[表征比较](reports/instance_representation_20261113/report.md)、[载体与内容配对测量](reports/carrier_content_channel_20261113/report.md)和[全局误差方向实验](reports/instance_nuisance_geometry_20261113/report.md)记录机制依据。

[2026-11-13研究汇报](presentations/defense_briefing_20261113/README.md)包含中山大学模板演示文稿、逐页讲稿和实验来源。

## 三个防御基线

| 基线 | 实现依据 | 验证与差异记录 |
|---|---|---|
| SemBind | 官方语义网络和DPM生成/精确反演；按最终论文补齐平铺、比例选择、秘密置换和符号绑定 | 原GS OTP空间重复的GS-S；PRC-S采用sigma0.5。作者未发布完整绑定入口，论文补齐部分明确标注。[规格](docs/THREE_BASELINE_REPRODUCTION.md) |
| PGID | 固定作者`dc4ad96`，原生PGID-F/R及载体判定 | GS的F/R各10例完整151次更新已验收。GS、TR、PRC、T2SMark原码CPU往返均通过。[结果与范围](docs/DEFENSE_BASELINE_INVENTORY.md) |
| SEAL | 固定作者`92d31b3`及ICCV2025最终正文、补充材料 | 原公开seed42路径与论文secret-salt路径分别实现、分别验证；距离阈值2.3、匹配块数12。[契约](docs/SEAL_EVALUATION_CONTRACT.md) |

源码版本、模型哈希、依赖和组件检查见[来源报告](reports/three_baselines_20261113/report.md)。模型导入、组件通过、正常图通过和完整论文数值复现分别判断，不用排队任务或少量样本替代后者。

[实现差异表](docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)集中列明作者原码、论文补齐逻辑、内存适配、密钥约定和未验收项。

已验收的正常图结果：GS-S为100/100；PRC-S、TR-S各6/6，无水印各0/6；SEAL公开代码为83/100、论文私有salt实现为91/100，对应无水印对照均为0/100。严格后端PRC-S重放同样6/6。这些是开发数据上的原生设置结果，不是同工作点排名或1e-6经验误报保证。

## 实现入口

| 功能 | 入口 |
|---|---|
| SemBind语义码 | [semantic_code.py](scripts/defense/semantic_code.py) |
| SemBind论文掩码 | [sembind_binding.py](scripts/defense/sembind_binding.py)，私有密钥；sigma为0至1 |
| 原GS/GS-S图像闭环 | [run_phase1_probe_worker.py](scripts/defense/run_phase1_probe_worker.py)，冻结100提示词协议 |
| PRC/PRC-S图像闭环 | [run_sembind_prc_native.py](scripts/defense/run_sembind_prc_native.py) |
| SEAL完整语义链路 | [run_seal_native.py](scripts/defense/run_seal_native.py)；论文salt见[seal_paper_simhash.py](scripts/defense/seal_paper_simhash.py) |
| 作者PGID完整预算入口 | [run_pgid_native.py](scripts/defense/run_pgid_native.py)，原样执行上游run.py |
| 实例密钥登记与恢复 | [instance_coset_recovery.py](scripts/defense/instance_coset_recovery.py)，认证记录、实例方向、128位真实恢复 |
| 恢复材料的原生检测 | [verify_recovered_native_keys.py](scripts/defense/verify_recovered_native_keys.py)，原作者检测器与哈希核验反演 |
| TR/PRC恢复迁移 | [recovered_carrier_transfer.py](scripts/defense/recovered_carrier_transfer.py)，冻结阈值、完整来源分母 |
| 完成后逐条审计 | [summarize_native_baseline.py](scripts/defense/summarize_native_baseline.py) |

每个运行在249的`runs/<run>/`保存冻结源码、协议哈希、逐样本产物和退出状态。当前代码可继续维护；重放既有结果使用该运行的冻结版本。工作进程拒绝覆盖已有结果。

SemBind生成和反演均为50步，inverse order0、VAE decoder优化100步。GS同时记录原源码166位、理想独立公平位下167位、历史182位三个工作点。理论阈值不是经验FPR保证；原生PGID、SemBind和SEAL的不同工作点不能直接用于方法排名。

## 证据和执行约束

[完整报告索引](reports/README.md)保留已完成实验和失败结果。[内容密钥评估](docs/CONTENT_KEY_FEASIBILITY.md)记录生成后登记的协议、实例区分及内容熵问题。当前实现使用冻结SSCD，已完成实际恢复；新的码器训练尚未执行。

所有模型推理和CPU评分仅在SSH主机249的`/data0/liyanlei/watermark_research`执行。只使用空闲GPU7、5、4，最多三张，始终排除GPU6；以进程退出事件接续任务，不持续轮询。模型、图像和私钥留在服务器，代码与可公开证据每完成一个单元就提交并推送main。

防御评测只使用顶会顶刊已经发表的攻击及其既定实现、完整预算，不创造或改进攻击。报告真实合法检出率、错误身份/未加水印负例、伪造接受、图像质量和计算成本，保留失败的合法参考。

- [机器状态](project_state.json)、[交接说明](GPT_CONTEXT.md)、[研究计划](docs/RESEARCH_PLAN.md)
- [基线证据](docs/DEFENSE_BASELINE_INVENTORY.md)、[第三方来源](THIRD_PARTY.md)、[操作规则](AGENTS.md)
