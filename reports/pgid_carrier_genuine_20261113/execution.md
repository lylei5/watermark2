# TR/PRC合法图对照的执行记录

实验固定每载体110张查询：10张干净图、90张GS官方处理图和10张自然封面；每张分别调用原生检测和PGID-F，共440条检测。全部像素、种子、模型、阈值及上游算法来自[冻结协议](../../protocols/pgid_carrier_genuine_20261113.json)。当前运行状态统一见[project_state.json](../../project_state.json)。

## 已定位的执行问题

最初两项任务在进入`running`状态前遇到`BrokenPipeError`，未产出合法图检测表。失败状态保存在[TR记录](failures/TR_initial_failure.json)和[PRC记录](failures/PRC_initial_failure.json)。第二次执行使用相同源码和逐字节相同协议，采用独立会话、关闭stdin、stdout/stderr直接写服务器日志；见[派发记录](genuine_relaunch_v2.json)。

TR第二次执行在首个来源读取CSV时发现`message_bits_str_initial`字段不存在。TR检测读取频域图案而不读取消息载荷，因此项目评测入口改为仅PRC读取该字段，TR传入`None`。修改仅一行；上游水印、反演、PGID与攻击源码均保持原样。该失败保存在[CSV读取记录](failures/TR_csv_failure.json)，修正派发见[TR记录](genuine_relaunch_v3.json)和[对应协议](../../protocols/pgid_carrier_genuine_20261113_v3.json)。协议相对原版只改变执行脚本哈希，所有数值字段相同。

TR使用v3执行，PRC继续使用v2原样执行。输出目录分别为249上的`runs/pgid_carrier_genuine_20261113_v3/TR`与`runs/pgid_carrier_genuine_20261113_v2/PRC`。原失败目录保留，不与完整分母混合；未完成记录不计为零接受。

## 配对审计

恢复迁移的v3审计已在两个合法图任务退出后完成，核验了完成文件、像素哈希和协议。数值协议逐字一致，源码只包含上述TR载荷字段读取差异；280张查询的原生跨运行判定差异为0。原审计缺少完成文件的失败记录保留。

## 完成结果

TR耗时2820.09秒，PRC耗时5772.67秒，各完成220条检测。每载体100张合法图与10张自然封面分别调用原生检测和PGID。TR原生合法100/100、PGID60/100；PRC原生合法64/100、PGID29/100。两种方法的自然封面均0/10。

完整记录：[TR完成文件](TR/complete.json)、[TR逐图结果](TR/rows.json)、[PRC完成文件](PRC/complete.json)、[PRC逐图结果](PRC/rows.json)。与攻击及真实密钥恢复的统一比较见[完整报告](../recovered_carrier_transfer_20261113/report.md)。

最终比较保留三个层次：内容/密钥恢复、原生水印检测、两者共同接受。冻结GS校准点的迁移与评估侧描述性相同合法通过数比较分别报告。后一项只用于查看观测到的性能关系，阈值不部署为新的方法。
