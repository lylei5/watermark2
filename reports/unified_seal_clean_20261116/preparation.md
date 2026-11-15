# SEAL共同500来源合法图评估

**SEAL后续合法图评估协议已冻结并排队：500个共同来源，1000次完整生成、1000次完整反演，尚无本批准确率结果。** 协调器通过现有校准进程退出事件接续；只有40000张自然图校准/验证正常结束，才复制两个固定阈值并检查空闲GPU。最多三张并发，排除GPU6，不抢占现有校准。

每个来源产生一张无水印预览图和一张最终水印图，分别测合法TPR和补充的生成图误报。沿用校准的同一私有salt、原PNDM流程、FP16、TF32关闭、BLIP2与句子模型。原作者50步对应每次51次UNet调用；生成CFG7.5、反演CFG1。公开seed42路径不混入这次论文salt检测器成绩。源队列包括COCO250、SDP250，所有来源保留在适用分母。

CPU249预检已完成：500条提示词在SEAL模型tokenizer下均与共同队列一致；原作者生成步数/CFG默认值核对通过；50步PNDM时间表实际有51项。另以原作者距离函数重算历史200份噪声对，形状和有限值检查通过；CPU FP32与历史GPU FP16最大距离差0.005953，不据此声称逐位或判定完全一致。未初始化CUDA，未运行新模型推理。

保存预览/最终caption、候选自身参考、原始嵌入噪声和反演结果。主判定始终使用候选自身内容派生的参考。额外对原始嵌入噪声打分只用于分析合法拒绝原因，不构成另一个检测器或方法变体。详见[逐攻击分析口径](../../docs/UNIFIED_FAILURE_ANALYSIS.md)。

| 项目 | 冻结值 |
|---|---|
| 运行目录 | `/data0/liyanlei/watermark_research/runs/unified_seal_clean_20261116` |
| 协议 | [unified_seal_clean_20261116.json](../../protocols/unified_seal_clean_20261116.json) |
| 协议SHA256 | `14d0ac04f12803865ae614ad7697200ed0f5bbd2a4c86c31ad68b4e0d25c4475` |
| 完成后产物 | `clean_results.json`、`clean_table.csv`、逐来源记录和原始特征 |
| 代码 | [run_unified_seal_clean.py](../../scripts/defense/run_unified_seal_clean.py) |
| 预检 | [seal_clean_preflight.json](seal_clean_preflight.json) |
| 入队记录 | [launch.json](launch.json)、[入队时状态](pipeline_state.json) |

这是完整系统面板的SEAL干净图部分。共同失真、已发表攻击、质量指标与其他防御仍需补齐；本准备记录不代表主表或逐攻击分析完成。
