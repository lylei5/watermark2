# 研究接续入口

## 最新进展（2026-11-03 UTC）：防御基线已评估

- 30 对、30 步 Imprint：普通 GS 30/30（BA 0.8228，PSNR 24.04 dB）；共享身份，不外推。
- Geometry：G-Cos AUC 0.9958、L-SPD 0.638（无公开阈值，只报 AUC）。
- PGID：官方 151 步、1 图，防御后 0.578/0.621/0.664（阈值 0.707，差 4.3pp）；n=1 且模型设定不同。
- SemBind：单对伪造 0.484；30 对批量 28 例中 2 例通过（平均 0.5379）；151 步对照未完成。
- 负结果：无密钥“投票边际”代理不可行（参考逐坐标符号错误 24.3%，模板一致率 42.6%，Spearman −0.025）。
- 报告：`reports/defense_baselines_20261103/report.md`；run：`runs/confirmation100_baseline_20261102T204500Z`。


**当前状态（2026-10-25 UTC）：**100 对确认资产 v1.1 已完成并发布：100 张载体、100 条唯一文本及固定配对。恢复核查通过服务器 200 份原图/PNG 的存在性、大小与 SHA256，以及315项执行封存文件哈希。[资产报告](https://github.com/liylei666-hub/watermark2/blob/e5c86c769dbb39853e8994367e4210045d782a54/reports/confirmation100_assets_v11_completed/report.md) / [恢复核验](https://github.com/liylei666-hub/watermark2/blob/e5c86c769dbb39853e8994367e4210045d782a54/reports/confirmation100_assets_v11_completed/recovery_verification_20261025.json)，研究提交 `e5c86c769dbb39853e8994367e4210045d782a54`。
**模型实验边界：**100 对模型实验尚未启动。Stage31 已发布结果仅为4对共享身份原生校准及事后CPU LPIPS，不构成100对完整论文复现。[校准报告](https://github.com/liylei666-hub/watermark2/blob/e5c86c769dbb39853e8994367e4210045d782a54/reports/31_native_calibration/report.md) / [质量补充](https://github.com/liylei666-hub/watermark2/blob/e5c86c769dbb39853e8994367e4210045d782a54/reports/31_native_calibration_lpips_20261025/report.md)。确定性选择与像素检查采用历史服务器回执，本轮没有重跑。
**下一项：**审阅完整 corrected native Imprint 100 对基线协议、独立私有 GS 材料及实际资源预算；100 对模型实验尚未启动。Stage30、八攻击对照和 Jain 原设置保持 hold。本轮仅发布及审阅，不启动GPU作业，不下载新模型。

更新：2026-10-25 UTC。目标为扩散语义水印黑盒伪造的方法、充分实验、可复现代码和论文初稿。用户授权自主研究、核验后正常推送GitHub。定时研究心跳已按用户要求移除。每次先核对真实进程、Git与机器账本。

## 阶段27历史结果与当前接续指针

**[阶段27完整方法比较](reports/27_method_resume_20261019T173017Z/report.md)已完成并复核，固定交付路径Imprint候选关闭。** 12任务中，原生Imprint普通GS通过8/12、delivery为6/12、Jain为3/12次计划尝试（11张交付、1次工程失败保留）。delivery平均BA低1.822917个百分点，2对提高、10对下降；中位配对像素MSE比0.908971、核心时间比1.051660。12张参考均有效，失真和时间满足门槛，但未达到至少增加3个通过的预设条件。这个固定候选不调参、补样或追加微观诊断；结论不推广为所有同类方法无效。

35张候选先封存，59张实际PNG后评估，全部消息分数经服务器独立CPU解码核验；本机重新计算像素、时间与汇总统计。全部参考/载体/候选均作模型辅助视觉检查，正式独立人工评审为0。普通GS不等于完整几何/SemBind接受；12任务是开发筛选，不是充分统计确认。

原始任务2 Jain在加载模型前失败，未重跑。r4只接续32个从未启动的攻击任务，算法未改；资源等待修复及CPU收尾均完成。整组45次工作进程启动（44成功、1失败），累计进程墙钟时间14775.389390秒。2026-10-19T21:20:54.039208+00:00实查：控制器、模型与CPU后处理均已结束，没有活动比较进程。原始run、失败和冻结文件保留。

**当前下一项：**审阅完整 corrected native Imprint 100 对基线协议、独立私有 GS 材料及实际资源预算；100 对模型实验尚未启动。Stage30、八攻击对照和 Jain 原设置保持 hold。

## 已验证的研究依据

- [阶段17](reports/17_cover_pilot/report.md)：近似376秒核心预算，Imprint普通GS通过2/3、适配Jain0/3。三对共用一个加密身份；全部候选先封存。完整防御与正式内容标签unknown。
- [阶段18/19](reports/19_target_encoding/report.md)：六个固定端点的排序改变依赖任务、编码与反演，不能单因归属。该组三对六候选的微观诊断关闭。
- [阶段21](reports/21_directional_study/report.md)：4对新图、2身份，同像素失真下反演方向相对VAE方向的目标MSE改善差均为正，但仅3/4改善载体、20候选GS均未通过。8项中心差分均超25%门槛，一阶机制未建立；正反方向只能作有限步长描述。本组关闭。
- [阶段22](reports/22_decoder_contract/report.md)：MSE、取位、重复投票与最终阈值须分层。原论文已有bin resampling；逐图独立key/nonce与历史共享身份实验不同。
- [目标函数审查](docs/METHOD_HYPOTHESIS.md)：最近同符号集合距离等价平方hinge，零间隔不保证负符号；半正态随机目标MSE平均等价固定均值目标加常数。基础恒等式不构成新方法或迁移保证。
- [阶段10–15交付/成本与组件证据](docs/COVER_BASELINE_CONTRACT.md)以及[历史01–05](docs/PROJECT_STATE.md)保留原始结果和失败。[执行账本](reports/research_execution_ledger.json)阶段27历史账本累计105进程、22027.065950秒，包含4个非零退出；原失败版本不重跑。

## 完整评估与新颖性边界

- [几何评估契约](docs/EVALUATION_CONTRACT.md)：最终版理论AND拒绝区域已查清，公开完整实现、参考关联和固定工作点仍缺。独立余弦不等于完整接受。
- SemBind最终版E.1/Table17已有5组端到端代理masker测试Det.=0；公开库不是完整GS-S验证器。
- [SEAL](docs/SEAL_EVALUATION_CONTRACT.md)需未缓存caption/embedding与固定语义协议；[PAI](docs/PAI_EVALUATION_CONTRACT.md)仍缺PCA参考和接口；不自造补全。
- [CSI](docs/CSI_INFORMATION_CONTRACT.md)使用检测失败反馈，不能原样作为零目标反馈基线。[PnP](reports/pnp_source_availability.json)匿名源403未绕过，完整性unknown。
- Jain已有单图VAE匹配与DCT限制，AdvPC/EOT/FIA已有相关机制；简单改损失、变换平均或自重建不是独立贡献。核对范围见[文献矩阵](docs/LITERATURE.md)，不能声称穷尽检索。
- [内容规则](docs/CONTENT_ADJUDICATION.md)正式人工评审0份，助手视觉观察不是人工标签。普通GS、内容、完整防御、错误归属和报警分别报告。

## 历史执行范围（截至阶段27；当前范围见页首）

仅liyanlei@172.18.185.249、/data0/liyanlei/watermark_research，物理GPU0–2中核对空闲的一张，每子进程核对UUID与占用，UUID绑定、cuda:0、串行。禁止本机GPU及其他服务器。用户已取消GPU时间授权上限，仍记录真实用量；各冻结协议保留1200秒工程watchdog。新增文件最多10GiB，不下载新模型，不终止他人进程。

CPU明确CUDA_VISIBLE_DEVICES为空。阶段09 seed103禁止重试或补样。保护原始run、失败、配置、图片和张量；工程失败保留后先CPU检查修复，再单独冻结版本。不得扫描网络、攻击生产服务、绕过SSH/TLS验证、存凭据或联系他人。

攻击/代理只读取获准参考图、输出提示词、代理配置及自身随机状态。目标密钥/消息/nonce、原始噪声、RNG、目标反演和分数属于评估端；候选或代理表示先封存哈希后评估，禁止分数反馈选候选或调参。阶段23私有材料在服务器private_eval，未传本机或Git；共享OS用户不构成权限隔离。

## 接续入口

[研究计划](docs/RESEARCH_PLAN.md) / [方法假设](docs/METHOD_HYPOTHESIS.md) / [机器状态](project_state.json) / [比较契约](docs/BASELINE_COMPARISON_CONTRACT.md) / [导师Word](docs/advisor_briefing.docx) / [原始证据](docs/ARTIFACT_GUIDE.md)。只发布确定、有用且复核过的进展；精简更新当前说明，保留所有原始证据。
