# 研究证据索引

本页索引已完成证据。当前任务状态以 [project_state.json](../project_state.json) 为准；各目录内的 plan、protocol 和执行快照是对应实验的冻结记录，不是待执行队列。失败、部分失败和否定结果保留。不同来源、载体和检测工作点之间不可直接排名。

## DPM 组件研究与基线重验

| 研究 | 已完成证据与边界 |
|---|---|
| [统一对比来源核验与完整成本](unified_benchmark_preflight_20261115/report.md) | 8张共同开发图、40份完整反演产物全部核验；普通DDIM/PGID-F/PGID-R/SemBind/SEAL为3.70/21.27/14.34/44.59/2.45秒每图；论文攻击并集及实现缺项已列明，非统一FPR结果 |
| [实例恢复负结果与附录](instance_recovery_negative_results_20261115/report.md) | 当前研究线冻结；汇总真实恢复、强对照、几何泛化、完整候选预算及存储证据，不新增实验或宣称总体不可能性 |
| [部分序恢复与未见编辑](order_capsule_confirmation_20261115/report.md) | 11400次部分序解码、7552次精确返回；已知合法993/1000，同合法九模板SSCD错收18/100、直接部分序与恢复均0/100；未见对比度500/500、旋转3/500、裁剪缩放18/500 |
| [饱和白坐标消融](order_saturation_ablation_20261115/report.md) | 400次真实解码及100次登记核验；保留或删除白坐标均亮度98/100、噪声95/100，200配对判定无变化 |
| [部分序完整开发攻击](order_capsule_development_20261115/report.md) | 3030次解码、934次精确返回；TR合法99/100，Reprompt+两代理0/10、1/10，同合法九模板2/10、3/10、直接参考0/10、0/10 |
| [部分序恢复开发探针](order_capsule_probe_20261115/report.md) | 40配对，首次取符号亮度10/10、噪声6/10；12个独立约束优化投影核对；既有前10实例开发数据 |
| [100实例前向恢复迁移](capsule_source_transfer_20261115_v3/report.md) | 2600次查询解码及100次登记核验；SSCD/CopyNCE1300图重新推理；主设置合法894/1000、同提示词不同实例0/100，SSCD899/1000、0/100 |
| [可观察前向恢复](transported_capsule_observability_20261115/report.md) | 5050次解码、2026次准确密钥、317次原生检测；TR95/100时Reprompt+两代理0/10，与九模板SSCD相同 |
| [前向密钥封装](transported_capsule_20261115/report.md) | 3030次实际解码、683次准确密钥；全部预设工作点合法覆盖不足；残差与直接参考预测误差等价 |
| [Reprompt+实际恢复与强对照](reprompt_plus_recovery_20261115/report.md) | 960次恢复、473次准确返回、202次恢复材料原生调用；TR合法95/100、Reprompt+接受8/10与7/10，同合法SSCD各1/10；96项精确合法计数匹配完成 |
| [作者Reprompt+固定预算](author_reprompt_plus_20261115/report.md) | 两代理、GS/TR各10来源；GS9/TR3候选预算完整生成240张；480条唯一原生/PGID判定，PGID来源接受GS各10/10、TR各8/10；TR首提示词20/20像素一致 |
| [多尺度可观察证据恢复](observable_evidence_recovery_20261115/report.md) | 3080次真实恢复、1394次准确返回、388次恢复材料原生调用；可观察规则与固定交集770/770相同；360项同合法通过数比较无SSCD优势 |
| [作者Reprompt与原生防御](author_reprompt_control_20261114/report.md) | 两代理、三载体、60张完整步骤候选；120条原生/PGID判定；六强对照及360次冻结SSCD真实恢复；全论文大样本矩阵待完成 |
| [空间证据恢复](spatial_witness_recovery_20261114/report.md) | 2310次实际恢复、749次准确返回、302次恢复材料原生调用；区域隔离TR合法68/100，270项同合法通过数比较无SSCD优势 |
| [可见性恢复](visible_witness_recovery_20261114/report.md) | 1540次实际恢复、655次准确返回、361次恢复材料调用；TR合法99/100但Reprompt8/10、9/10；180项同合法通过数比较完成 |
| [TR/PRC验证材料恢复](recovered_carrier_transfer_20261113/report.md) | 280次新反演、3360次实际恢复；2179次成功均准确；440条PGID合法图/封面对照完成，TR合法60/100、PRC29/100；SSCD单参考95/100、64/100；210组描述性共同合法通过数比较完成 |
| [共同频谱实例恢复](shared_channel_binding_20261114/report.md) | 17040次真实恢复及690次划分内部配对消融；4143次成功全部准确；共同相位TR最大恢复19–21/100，SSCD83–84/100；288组描述性比较均未超过强对照 |
| [CopyNCE作者模型对照](copynce_control_20261114/report.md) | 440条图像观测、710个记录配对；冻结GS95%匹配器联合TR98/100、PRC59/100；描述子/SSCD/int8模板、来源bootstrap与五类编辑留出校准完整比较 |
| [TR/PRC完整Imprint](pgid_carrier_coverage_20261113/report.md) | 每载体10例完整151更新；原生每检查点10/10接受，PGID与固定SSCD均0/10；140条判定与80张图哈希核验 |
| [配对编辑观测与密钥恢复](paired_channel_recovery_20261113/report.md) | 80次新反演、3960次真实恢复、199次原生GS检测；95%校准点配对45–46/50，多模板SSCD48/50；配对打乱及实例混淆完整对照 |
| [实例编辑方向与密钥恢复](instance_edit_quotient_20261113/report.md) | 14760次真实恢复；八方向473–477/500、不同实例0/50；共同90%校准点、同变换连续SSCD、存储与来源bootstrap对照；196次恢复材料原生检测 |
| [编辑状态码恢复](instance_edit_code_20261113/report.md) | 14760次真实恢复；模板441–443/500，陪集无额外增益；195次恢复材料原生检测 |
| [实例密钥恢复接口](instance_coset_recovery_20261113/report.md) | 2110张图新特征、14760次真实128位恢复；编辑可靠选位466–467/500；固定/模板/陪集完整比较 |
| [载体与内容漂移](carrier_content_channel_20261113/report.md) | 130张图重新检测，BA全部重放一致；完整GS Imprint原生10/10、SSCD门控0/10；通用潜变量统计的漂移预测优于等维载体错误统计 |
| [处理误差方向](instance_nuisance_geometry_20261113/report.md) | 24/26/50来源拆分，普通子空间对照；连续特征426–430/450，二值半径187–208仍较大；无新攻击 |
| [最终发布实例表征](instance_representation_20261113/report.md) | 新执行1300图×两种提取链；SSCD最终图参考428/450、实例误匹配1/50；SemBind码420/450、42/50；开发数据按提示词族拆分 |
| [二值码纠错预算](sscd_binary_fe_20261113/report.md) | 12个投影/码长/选位配置；1024位可靠选位半径214–220；码参数分析已完成，未实现FE |
| [三个原生防御基线](three_baselines_20261113/report.md) | GS-S正常图100/100；SEAL公开83/100、论文salt91/100；PGID-R完整151更新10/10恢复。逐方法范围与差异单独报告 |
| [汉明恢复半径](phase1_hamming_radius_20261113/report.md) | 保留至少95%配对的描述性半径同时纳入82%/90%同提示词实例；非攻击/FE结果 |
| [100提示词轨迹](phase1_trajectory_native_gs_20261113/report.md) | 生成、300次DPM反演及400条决策完成；第35步P5为0.904639/0.892383，未通过轨迹门槛 |
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
