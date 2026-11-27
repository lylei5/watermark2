# 研究证据索引

运行状态见[机器状态](../project_state.json)。本页保存历史实验的实现范围、分母、工作点与逐样本证据。2026-11-18 的[语义读回研究记录](semantic_readback_20261118/receipt.json)列明已完成的评分文件和中止位置。旧统一基准的[部分产物清单](unified_null_stop_20261116/report.md)及[当时项目快照](https://github.com/lylei5/watermark2/blob/ac09574711f9ea2b056b5782e683d51e5eb2f87e/project_state.json)可用于回查。

## 前向承诺与中步流形引导（Commit-then-Steer 无库内容绑定）

[Commit-then-Steer Stage 6 完整实验报告](commit_steer_stage6_20261126/report.md)已完成标准千图质量评测与 `N=10` 实时伪造显著性扩展（全部 16 项任务 `rc = 0`，独立 0-mismatch 审计 `total_mismatches = 0`）：**Stage 6A MS-COCO-1K 无参考 FID/CLIP 与千图盲检**（1,000 条不重叠未见 `val2017` 提示词 vs 5,000 张真实 `test2017` 参考图，`sscd_64b` FID `52.97 → 52.17 (Δ=-0.79)`、CLIP `0.2614 → 0.2605`、Clean 认证 `1,000/1,000` / 盲联合 `973/1,000 = 97.3%`；`hyb_sscd32_clip32` FID `53.02 → 52.98 (Δ=-0.04)`、CLIP `0.2639 → 0.2634`、Clean 认证 `1,000/1,000` / 盲联合 `972/1,000 = 97.2%`；同 $z_T$ 轨迹扰动诊断 `hyb` LPIPS `0.089` / 配对 FID `14.83` vs `sscd_64b` `0.114` / `19.01`）；**Stage 6B `N=10` 实时 151 步 Imprint 全基线扩展**（跨提示词 6 方法 180 检查点下，Commit-then-Steer `sscd_64b` 与 `hyb` 均为 `0/10` [全 30/30 拦截，95% CP `0.0%–30.8%`]，精确双边 Fisher 检验显著优于 Native GS `10/10` [`p=1.08×10⁻⁵`]、SEAL `@4` `6/10` [`p=0.0108`] / `@2` `7/10` [`p=0.0031`] 与官方 PGID-F `≥167` `5/10` [`p=0.0325`]，与 SemBind `0/10` 持平；同提示词兄弟图 3 方法 90 检查点下，SemBind GS-S `10/10` 全破 [全 30/30 检查点 `BA=1.0000`]，Commit-then-Steer `sscd_64b` 压至 `3/10` [`p=0.0031`]、`hyb` 压至 `5/10` [`p=0.0325`]）。

[Commit-then-Steer Stage 5 完整实验报告](commit_steer_stage5_20261126/report.md)已完成全量四组实验与独立 0-mismatch 审计（`total_mismatches = 0`）：**Stage 5B/5C 同提示词实例级伪造压力测试**（未改写 CVPR 2025 151 步 Imprint 攻击同提示词独立兄弟图与 24 候选同提示词 Reprompt+，论文级 SemBind GS-S 在 Imprint `4/4` 全 `12/12` 检查点 `BA=1.0000` 与 Reprompt+ `24/24 = 100%` 全线失守，而 Commit-then-Steer `hyb_sscd32_clip32` 压至 Imprint `1/4`、`9/12` 检查点拦截与 Reprompt+ `2/24 = 8.3%`，`sscd_64b` 为 `2/4` 与 `9/24 = 37.5%`）；**Stage 5A SEAL 完整同协议正面对比**（40k 校准冻结工作点 `@12/@4/@2` 下 Matched-30 9-GS `45.6%/68.5%/81.5%`、Matched-20 WAVES-8 `49.4%/58.1%/63.8%`、Regen-4 `26.3%/36.3%/46.3%`、Live Imprint `0/4` vs `2/4` vs `2/4`、Live Proxy Reprompt `0/36` vs `16/36` vs `26/36`、官方目标再生 `regeneration.py` `0/18` vs `10/18` vs `15/18`，揭示鲁棒性与安全性双重坍塌）；**Stage 5A/5B/5C 官方 ICML 2024 WAVES-8 (`strength=0.5`) 与扩散重绘擦除（Matched-20 五方法 1,200 核验）**（Commit-then-Steer 192b 认证段零损，盲联合 `sscd_64b` WAVES-8 `104/160 = 65.0%` / Regen-4 `65/80 = 81.3%`，大幅超越 SEAL 并逼近 SemBind `110/160` / `80/80`）；**Stage 5D SDXL Base 1.0 (`1024×1024`) Live 151 步 Imprint**（原生 GS-SDXL `4/4` 全 `12/12` 检查点 `auth_ba=1.0000` vs Commit-then-Steer SDXL `0/4` 全拦截，`mean d_blind=0.9207`）。

[Commit-then-Steer Stage 4 完整实验报告](commit_steer_stage4_20261125/report.md)已完成全量四组实验、离线 COCO-50 FID 与独立 0-mismatch 审计（`total_mismatches = 0`）：**Stage 4A SDXL Base 1.0 (`1024×1024`) 跨骨干**（24 条不重叠统一集 Clean `24/24`、9-GS `184/216 = 85.2%`、`LPIPS=0.0705`、Live 36 候选 Reprompt+ `0/36` vs 原生 `36/36`、200 COCO 自然图 `0/200`）；**Stage 4B 100 图多维融合 `hyb_sscd32_clip32`**（Clean `96/100`、9-GS `613/900 = 68.1%`、`LPIPS=0.096`、COCO-50 FID `227.18→230.64` / 配对漂移 `43.08`、2,000 COCO 自然图默认与特征项均 `0/2,000`、Live 151 步 Imprint `0/4` vs `4/4`、Live 108 候选 Reprompt+ SD2.1 `1/54` & SD1.5 `1/54` vs `108/108`）；**Stage 4C Native GS 与官方 PGID-F 同协议对比**（Matched-30 下 PGID-F 验证 `21.59s`、9-GS `175/270 = 64.8%`、Null200 `0/200`、Imprint step150 `2/4`、Reprompt+ `104/108 = 96.3%` 失守，Commit-then-Steer 全面胜出）；**Stage 4D 论文级 SemBind GS-S 同协议对比**（Matched-20 下生成 `22.09s`、验证 `49.51s`、`LPIPS=0.723`、Clean `20/20`、9-GS `169/180 = 93.9%`、Null50 `0/50`、Imprint `0/4`、Reprompt+ `0/72`，与 Commit-then-Steer 构成清晰 Pareto 权衡并严格标注实现边界与 SEAL 参照；主表空白单元格已随 Stage 5 全部补齐）。

[Commit-then-Steer Stage 0–3 完整实验报告](commit_steer_20261123/report.md)完成了纯 RGB/直方图/通道比例证伪（Stage 0）、中步 Tweedie 流形引导与窗口消融（Stage 1 & 3E）、官方统一集 100 条未见 prompt 的原生 256-bit GS 盲联合验证（Clean `99/100`、载荷 BER `0.00%`、9 项 GS 失真 `694/900 = 77.1%`、7 项常规失真 `86.1%`、离线 COCO-50 FID `-4.54`）、2,000 张真实 COCO 自然图经验 Null FPR（默认 `0/2,000`，经验 1% auth 切点下由 `15/1,000` 压至 `0/1,000`）、全预算 Live 151 步 Imprint（原生 `4/4` vs 本文 `0/4`）与 108 候选 Reprompt+（SD2.1 `0/54`、SD1.5 `2/54`）、跨载体原生 512-bit PRC 16 条统一 prompt（Clean `15/16`、失真可解码中 `22/23`、Live Reprompt `1/8` vs `8/8`）、多维度 64-bit 融合（`32b SSCD + 32b CLIP` Stage 0 Reprompt+ `1/180`、端到端 Live Reprompt `0/8`）及 80 步白盒自适应 PGD 压力测试（全部独立复核 `mismatches = 0`）。

## 生成图的颜色指纹

[按登记原图色度丰富程度加权](color_richness_20261120/report.md)在 60 条随机提示词和 40 条定向提示词上测试原乘积与加权规则。近灰组自身合法通过对齐到 95% 时，误收从 577/928 降至 424/928；全局门限下另见近灰强模糊通过率下降。

[新提示词的 49 次误收与 9 张强对比度拒绝图](color_product_cases_20261119/report.md)逐组对照原图，其中两对近灰图贡献 42 次误收，9 张额外被拒真图的布局检索均唯一首位。

[单项否决与原乘积规则](color_veto_20261119/report.md)在新提示词上对齐 95% 合法通过，误收由 49/1,920 变为 65/1,920，强模糊多通过 8 张，强对比度少通过 9 张。

[逐组误收与强模糊诊断](color_joint_diagnosis_20261119/report.md)对回 8 对原图和 45 张被拒查询，给出各分项的直接证据、模糊强度配对与原图检索结果。

[分别计算颜色与布局证据的独立确认](color_joint_20261119/report.md)收录额外 60 对新图，冻结门限下合法通过 1,819/1,920，同提示词误收 114/1,920。1.8 倍对比度下合法通过 118/120，误收 4/120。

[288 字节等预算验证](color_budget_20261119/report.md)比较新提示词上的固定分区、相对分区、全局色度、明暗布局及颜色与布局组合，报告强对比度认回和 95% 合法通过工作点的同提示词误收。

[亮度分区与同提示词种子区分](color_identity_20261119/report.md)收录 100 对原图的校准和留出评测，分别报告图库检索、身份验证、逐编辑结果与配对区间。

## DPM 组件研究与基线重验

| 研究 | 已完成证据与边界 |
|---|---|
| [SemBind共同提取一致性](unified_extraction_reproducibility_20261116/report.md) | 8来源×3GPU完成24份产物，16组新旧入口逐值一致；PGID24份差异0；完整预算及容差保持，正式自然图队列已恢复，尚非SemBind/PGID主表准确率 |
| [SEAL共同工作点与500来源合法检出](unified_seal_results_20261116/report.md) | 40000自然图完成；目标1%/0.1%的独立误报169/20000、10/20000，干净合法500/500、496/500；40500份记录及1000图独立复核；原阈值12为453/500，攻击和共同失真待完成 |
| [统一对比500来源队列](unified_main_sources_20261116/report.md) | COCO250/SDP250；121份历史元数据、312750对文本核验，规定重叠0；500原文及有效CLIP输入核验完成，全部249 CPU；不是主表准确率结果 |
| [SEAL校准进度核验](unified_seal_progress_20261116/report.md) | 2026-11-16 03:57北京时间保存20152/40000；最新三产物完整检查通过；校准尚未完成，固定时间快照不代表当前进度 |
| [256位GS随图指纹机制验证](gs_fingerprint_fixedpoint_20261116/report.md) | 50提示词，1%/3%/6%三轮收敛均0/50；九类处理联合32/450、39/450、66/450；GS基准449/450，三配置448/450、449/450、449/450；500次生成、2000次完整反演，1.398 GPU小时，未通过并停止 |
| [公开SSCD二值参考＋连续查询](sscd_public_binary_20261116/report.md) | CPU1820配对、9100判定；128位合法425/450、同提示词5/50，256位431/450、4/50；Reprompt类均至多比连续SSCD多1来源；汉明误收14/50、10/50。未实现随图嵌入 |
| [历史PGID分母审计](unified_historical_denominators_20261116/report.md) | 440条原判定与220对同像素查询核验；100→60和64→29均为10来源的干净+编辑合计，干净图TR10→8/10、PRC10→10/10；非新模型结果 |
| [40000图共同自然负例队列](unified_natural_cohort_20261115/report.md) | 官方40670图完整获取且全部可解码；去重/开发关联排除后40642簇，冻结20000校准+20000独立验证；SEAL协议已冻结，准确率和主表尚未完成 |
| [PRC连续分数与校准接口](unified_calibration_interface_20261115/report.md) | 47组输入、5个名义水平，235次判定与原Detect完全一致；并列值和不可达FPR分支检查完成，尚非自然图FPR校准 |
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
