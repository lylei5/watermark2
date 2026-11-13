# TR/PRC实际恢复的冻结迁移协议

本目录保存已启动的执行协议与派发记录，数值结果待任务完成。当前队列统一见[project_state.json](../../project_state.json)。

采用[GS开发实验](../instance_edit_quotient_20261113/report.md)冻结的四种方法、三组投影与90%/95%校准阈值。TR、PRC各10个来源，查询含干净图、九类官方合法处理、自然cover和已有完整151更新Imprint的三个检查点，每载体140个查询。

每个查询执行原生反演与检测。随后从内容特征恢复128位绑定密钥，解封原作者验证器状态，并再次调用原生检测函数；相同恢复材料复用一次检测。共280次新反演、3360次真实恢复。PRC使用作者DPM精确反演，TR使用作者DDIM设置。代码不包含攻击生成。

单参考SSCD、九模板SSCD、int8九模板SSCD和相同实例编辑方向的连续SSCD看到同一批像素。恢复失败保留在合法分母。原生判定的跨运行差异逐条保存；后处理与既有PGID合法图和攻击结果按来源、像素哈希匹配。

验证材料采用完整原作者对象状态的可信序列化，以验证恢复接口。CPU检查中TR状态149336字节，PRC状态10041440字节；本轮按实际记录开销报告。密钥、状态和辅助记录均留在249的private_eval，公开协议只包含哈希。

工作进程先通过pidfd等待现有SSCD门控进程退出，再检查GPU7空闲并绑定UUID。CPU审计进程等待恢复任务及两项合法图任务退出后生成audited_scores.json，不持续轮询。GPU6始终排除。

入口：[恢复与原生检测](../../scripts/defense/recovered_carrier_transfer.py)、[退出后配对审计](../../scripts/defense/score_recovered_carrier_transfer.py)。冻结参数见[协议](../../protocols/recovered_carrier_transfer_20261113.json)，派发见[launch.json](launch.json)。
