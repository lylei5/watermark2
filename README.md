# 扩散模型水印防御研究

本项目研究扩散水印的内容绑定防御，同时建立可复核的SemBind、PGID、SEAL原生基线。[实时状态](project_state.json)记录运行队列；[验收契约](docs/THREE_BASELINE_REPRODUCTION.md)规定实现和数值复现的范围。

最新的[载体与内容配对测量](reports/carrier_content_channel_20261113/report.md)重新执行130张图的检测，全部原生BA重放一致。10个完整151更新Imprint输出均通过原生GS、均被最终图SSCD门控拒绝。载体错误统计可预测部分内容漂移；等维数通用潜变量统计的预测更准，已纳入方法对照。[处理误差方向实验](reports/instance_nuisance_geometry_20261113/report.md)进一步表明，全局处理方向跨实例迁移有限；研究继续围绕[最终实例的恢复区域](docs/INSTANCE_BINDING_RESEARCH.md)推进。TR/PRC完整攻击与配对合法图任务状态见project_state.json。

[100提示词表征诊断](reports/instance_representation_20261113/report.md)比较三种表征与两种参考方式。评估侧SSCD最终图参考保留428/450张处理图，同提示词不同实例误匹配1/50；SemBind码对应420/450、42/50。[二值化实验](reports/sscd_binary_fe_20261113/report.md)将1024位码的95%校准恢复半径从385–390降到214–220位。普通子空间对照进一步达到187–208位；实际密钥恢复仍是后续方法工作。

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
| 完成后逐条审计 | [summarize_native_baseline.py](scripts/defense/summarize_native_baseline.py) |

每个运行在249的`runs/<run>/`保存冻结源码、协议哈希、逐样本产物和退出状态。当前代码可继续维护；重放既有结果使用该运行的冻结版本。工作进程拒绝覆盖已有结果。

SemBind生成和反演均为50步，inverse order0、VAE decoder优化100步。GS同时记录原源码166位、理想独立公平位下167位、历史182位三个工作点。理论阈值不是经验FPR保证；原生PGID、SemBind和SEAL的不同工作点不能直接用于方法排名。

## 证据和执行约束

[完整报告索引](reports/README.md)保留已完成实验和失败结果。[100提示词轨迹测量](reports/phase1_trajectory_native_gs_20261113/report.md)未通过冻结门槛；[内容密钥评估](docs/CONTENT_KEY_FEASIBILITY.md)记录后续研究所需解决的生成时序和实例区分问题。新防御训练待基线验收后推进。

所有模型推理和CPU评分仅在SSH主机249的`/data0/liyanlei/watermark_research`执行。只使用空闲GPU7、5、4，最多三张，始终排除GPU6；以进程退出事件接续任务，不持续轮询。模型、图像和私钥留在服务器，代码与可公开证据每完成一个单元就提交并推送main。

防御评测只使用顶会顶刊已经发表的攻击及其既定实现、完整预算，不创造或改进攻击。报告真实合法检出率、错误身份/未加水印负例、伪造接受、图像质量和计算成本，保留失败的合法参考。

- [机器状态](project_state.json)、[交接说明](GPT_CONTEXT.md)、[研究计划](docs/RESEARCH_PLAN.md)
- [基线证据](docs/DEFENSE_BASELINE_INVENTORY.md)、[第三方来源](THIRD_PARTY.md)、[操作规则](AGENTS.md)
