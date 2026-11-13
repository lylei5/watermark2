# TR/PRC实际恢复的冻结迁移协议

本目录保存已完成恢复迁移的执行协议、派发记录与[数值结果](report.md)。280次反演和3360次恢复已完成；440条合法图/封面配对检测及最终审计均已完成。当前队列统一见[project_state.json](../../project_state.json)。

采用[GS开发实验](../instance_edit_quotient_20261113/report.md)冻结的四种方法、三组投影与90%/95%校准阈值。TR、PRC各10个来源，查询含干净图、九类官方合法处理、自然cover和已有完整151更新Imprint的三个检查点，每载体140个查询。

每个查询执行原生反演与检测。随后从内容特征恢复128位绑定密钥，解封原作者验证器状态，并再次调用原生检测函数；相同恢复材料复用一次检测。共280次新反演、3360次真实恢复。PRC使用作者DPM精确反演，TR使用作者DDIM设置。代码不包含攻击生成。

单参考SSCD、九模板SSCD、int8九模板SSCD和相同实例编辑方向的连续SSCD看到同一批像素。恢复失败保留在合法分母。原生判定的跨运行差异逐条保存；后处理与既有PGID合法图和攻击结果按来源、像素哈希匹配。

验证材料采用完整原作者对象状态的可信序列化，以验证恢复接口。CPU检查中TR状态149336字节，PRC状态10041440字节；本轮按实际记录开销报告。密钥、状态和辅助记录均留在249的private_eval，公开协议只包含哈希。

恢复工作进程经pidfd等到SSCD门控进程退出，在空闲GPU7上完成。两项合法图对照的补跑与读取修正见[执行记录](../pgid_carrier_genuine_20261113/execution.md)。版本化审计在它们退出后生成了audited_scores_v3.json；210组描述性比较保存在comparison_analysis.json。原始失败记录保留，不持续轮询。GPU6始终排除。

入口：[恢复与原生检测](../../scripts/defense/recovered_carrier_transfer.py)、[退出后配对审计](../../scripts/defense/score_recovered_carrier_transfer.py)。冻结参数见[协议](../../protocols/recovered_carrier_transfer_20261113.json)，派发见[launch.json](launch.json)。

## 配对不确定性与汇总图

[来源级重采样](source_uncertainty.json)已完成。代码固定GS校准阈值，以来源为抽样单位，每个来源的10张合法查询一起抽取；10000次bootstrap只描述当前开发来源的不确定性。[分析入口](../../scripts/defense/audit_transfer_source_uncertainty.py)保存种子、原始来源差值和输入哈希。三组投影各自报告，不将其当成三倍独立样本。

[汇总图入口](../../scripts/defense/plot_carrier_transfer.py)在完整配对审计与描述性比较进程退出后执行，派发记录见[figure_launch.json](figure_launch.json)。它从审计JSON生成同一数据表的PNG、PDF和CSV；合法图分母为100，完整Imprint终点分母为10。图中SSCD与恢复方法使用冻结GS90%校准点，原生检测和PGID使用作者设置。不可行的校准点显示文字，三组投影显示为三个观测点。

图表用于内部研究比较。绘图流程参考scientific-visualization；软件出处见[相关工作中的工具记录](../paired_channel_recovery_20261113/related_work.md)。PNG的标签、脚注、工作点与缺失值显示已完成可视检查，PDF字体嵌入已核验。
