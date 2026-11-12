# 研究证据索引

本页索引已完成证据。当前任务状态以 [project_state.json](../project_state.json) 为准；各目录内的 plan、protocol 和执行快照是对应实验的冻结记录，不是待执行队列。失败、部分失败和否定结果保留。不同来源、载体和检测工作点之间不可直接排名。

## DPM 组件研究与基线重验

| 研究 | 已完成证据与边界 |
|---|---|
| [汉明恢复半径](phase1_hamming_radius_20261113/report.md) | 保留至少95%配对的描述性半径同时纳入82%/90%同提示词实例；非攻击/FE结果 |
| [100提示词轨迹](phase1_trajectory_native_gs_20261113/report.md) | 生成阶段完成；第35步P5为0.904639/0.892383，未通过门槛；完整验证仍运行 |
| [固定公开输入](phase1_public_inputs_20261113/report.md) | 100张COCO图核验与100条过滤I2P提示词；无攻击运行 |
| [PRC与模型资产](phase1_backend_audit_20261113/report.md) | PRC原码8/8 CPU往返；SD1.5/CLIP权重核验；未完成PRC-S图像实验 |
| [随机错误身份工作点](phase1_wrong_key_null_20261113/report.md) | 300万次随机身份；167位0次误报，95%上界1.22963e-6；不代表固定用户自然图FPR |
| [第一阶段规格审计](phase1_contract_20261113/report.md) | 最终论文设置、原GS布局与理论检测工作点 |
| [连续语义区域](continuous_geometry_screen_20261113/report.md) | 24 条干净图验证；绑定和 BA 门槛未通过；未执行攻击 |
| [完整码字条件传输](semantic_transport_pilot_20261112/report.md) | 240 条合法查询；主候选 55/60，重排对照 57/60；未执行攻击 |
| [最小反射区域](decision_region_pilot_20261112/report.md) | 240 条合法查询；主候选 44/60，full 59/60；未执行攻击 |
| [固定语义反馈](semantic_feedback_pilot_20261112/report.md) | 180 条合法查询；58/56/57，分母均为 60；未执行攻击 |
| [共享语义证据](shared_semantic_evidence_20261112/report.md) | 120 条合法查询及 216 条完整 Imprint 验证；未建立匹配合法检出率收益 |
| [重复结构诊断](replica_moments_development_20261112/report.md) | 已消费数据的开发诊断；未建立独立优势 |
| [条件分位载体](conditional_quantile_pilot_20261112/report.md) | 240 条合法查询；候选 48/60，full 59/60；未执行攻击 |
| [共享内容锚](conditional_anchor_pilot_20261112/report.md) | 候选 53/60，独立对照 54/60；未通过门槛 |
| [条件内容似然](conditional_content_likelihood_20261112/report.md) | 候选合法 30/40，校准 BA 39/40；没有新增最终伪造拒绝 |
| [完整 Imprint 重验](imprint_fresh_revalidation_20261112/report.md) | 144 条复核；项目 SemBind 组件适配 0/4、GS 4/4 最终接受 |
| [SemBind 组件九条件](sembind_published_revalidation_20261112/report.md) | 配对项目适配与反演设置诊断 |
| [SemBind/GS 完整链路组件](sembind_chain_revalidation_20261111/report.md) | 18 条干净图记录全部接受；不是完整原生复现 |
| [PGID 九条件](pgid_published_revalidation_20261112/report.md) | 标准 90/90，PGID 后 61/90；开发评估 |
| [防御组件重验](defense_revalidation_20261111/report.md) | 100 图语义组件及 PGID 官方完整预算重验 |

## 历史证据

[历史证据审计](repository_audit_20261111/report.md)记录适用条件和纠正后的分母。历史 DDIM 项目适配、短预算探针及旧确认集不构成原生 SemBind 的最新成绩。已执行源码可从各报告所列的服务器冻结目录和 Git 历史定位。

- [官方 DPM 组件检查](sembind_official_audit_20261110/report.md)
- [完整预算 Imprint](full_imprint_20261111/report.md)
- [几何校准](geometry_calibration_20261111/completed/report.md)、[六条件确认](geometry_confirmation_20261111/report.md)、[官方九条件](gs_published_conditions_20261111/report.md)
- [自然负例](unwatermarked_validation_20261111/report.md)
- [仓库同步与源码核验](repository_sync_20261113/report.md)

- [三个原生基线来源核验](three_baselines_20261113/report.md)：固定源码、SEAL环境和上游权重；完整链路状态见project_state.json。
