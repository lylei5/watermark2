# 文献、来源与历史实验结论审计

日期：2026-11-11。目标是让防御研究的文献与判断可追溯。本次核对核心文档、14项文献元数据、第三方来源和选定历史实验。服务器CPU复核已有记录，没有启动模型实验或创建、修改攻击方法。

## 文献核验

完整题名、编号、作者和一手链接见[文献记录](../../docs/LITERATURE.md)。

| 条目 | 经原始来源核验的记录 |
|---|---|
| SemBind | arXiv:2601.20310；ICML2026正式论文集 |
| Geometry | arXiv:2606.29807；完整题名含副标题A Geometric Distortion Perspective；ICML2026正式论文集 |
| semantic-forgery | arXiv:2412.03283；CVPR2025 |
| PAI | arXiv:2601.06639；Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection；ICLR2026正式论文集可确认 |
| PGID | Progressive Guided Inversion and Denoising；arXiv:2605.09319，作者记录注明NeurIPS2026接收 |
| SEAL | Semantic Aware Image Watermarking；ICCV2025 |

原编号2602.07512对应无人机目标检测、2602.11702对应电子结构计算、2503.05660对应磁化纹理光学切换、2509.22253对应四维空间曲面的twistor lifts，均不是所引用水印论文。另修正ISTS的展开、Sign Clamping和CoverLock的题名及方法描述；PP-Mark接受状态和SLICE日程收录与正式论文集区分。上述元数据核验不代表全文每项主张已经验证。

## 历史结果与结论

| 对象 | 核验结果 | 结论边界 |
|---|---|---|
| PGID | [42条逐行记录](raw_evidence_audit.json)从初始/恢复位串重算BA、阈值判定、分组人数与汇总，全数一致；n=5已完成 | 防御后50/100/150检查点均0/5接受；均值0.5625/0.59921875/0.665625；原n=1均值不可写作n=5，也不支持防御失效 |
| SemBind masker | 原始JSON的20条记录复核：原图/失真图官方与封装位差均0；失真距离均值20.55，标准中位数8 | 历史代码对偶数样本取上中位观测值，曾写9；函数已修复。跨图190对只有汇总，本次未独立重算；最小非零距离29不证明语义碰撞或完整防御绕过 |
| Geometry | [已保存30任务分数的汇总复算](geometry_summary_check.json)：六组AUC和均值一致 | 未从原始潜变量重算分数，未重核论文表格；AUC相同不证明完整复现，L-SPD公式适配结果不证明作者方法缺陷 |

[PGID报告](../repro_pgid_20261103/report.md)、[SemBind组件报告](../repro_sembind_masker_20261103/report.md)和[Geometry报告](../repro_geometry_paper_task_20261103/report.md)已直接修正正文，原始数据保留。SemBind后续反演组件状态链接到2026-11-10报告，历史环境失败不再用作当前状态。

中位数修复在249用函数本体检查偶数、奇数、单元素输入，见[检查记录](statistics_check.json)。[历史数据核验脚本](audit_historical_evidence.py)可在249读取原始CSV/JSON复跑；输出只含脱敏记录和哈希，不发布原始payload位串。

## 来源与研究范围

[服务器目录与Git元数据观察](server_sources.json)及本地`.gitmodules`/Git索引支持[第三方来源清单](../../THIRD_PARTY.md)。PGID、SEAL和两个子模块有可核对HEAD；SemBind、ISTS目录本次没有可用Git元数据；没有观察到PAI目录。目录或HEAD存在不证明完整权重、干净工作树或方法复现。

[研究计划](../../docs/RESEARCH_PLAN.md)与[基线复现清单](../../docs/DEFENSE_BASELINE_INVENTORY.md)只保留防御假设及其所需对照。撤回无证据的普适漏洞、完整复现、性能和理论保证；删除与研究约束冲突的自定义攻击开发计划。已有标准基线的事实与原始结果继续保留。源码契约仍用于解释已发表基线的信息权限，不作为改进攻击的计划。

## 核查覆盖与剩余缺口

- 核查根目录研究入口、`docs/`现有文档，并修订上述三份历史报告。对非第三方Markdown/JSON进行错误编号、展开和来源地址检索；剩余旧编号只出现在本审计的纠错说明中。
- 当前已完成实验与未证明SOTA的边界在README、GPT_CONTEXT、PROJECT_STATE和机器可读状态中保持一致；本次没有重跑近期模型评测。
- 既有四份源码契约标明原核查时间和固定版本。本轮没有重新逐行验证全部上游源码、每一篇论文全文、所有历史报告和全部原始张量；已有“已核验”文字不视为全库真实性认证。
- 三条失效的内部证据链接已处理：缺失的载体契约标为未复核摘录、Geometry文件清单注明附件缺失并指向现存证据、PAI比较契约的不可访问引用改为正文区分。需要采用未复核摘录中的参数时重新核对固定源码。PGID原论文表值、模型字节等价性、原生基线闭环和跨身份独立性仍需补证。
- 下一阶段依据开发数据开展防御机制分析；新方法的结果需要新的独立确认集。不得用本次文档修正替代实验验证或宣称防御领先。
