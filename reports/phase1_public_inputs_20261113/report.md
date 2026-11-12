# 第一阶段固定公开输入准备

2026-11-13完成100/100张COCO公开图的获取、SHA256核验与249上的512×512预处理。图片按冻结ID选择，避开已检查历史清单和本次生成提示词对应的COCO图像；只使用元数据中license4（CC BY）对象，具体来源与许可随协议保存。未将图像提交Git。

249直接下载完成3/100，其余97张因连接错误失败。一次固定URL的本地文件传输补充完成97/97，随后传回249核验、解码与预处理；没有替换ID或使用本机做模型推理/图像预处理。初始失败记录保持不变，完整获取结果单独保存。

另冻结100条经过类别、文本和人工审查的I2P提示词作为可能的Reprompt输入。该过滤子集不是论文未过滤I2P的严格复现，未来实验必须披露此差异。这一单元只准备输入，未执行任何攻击。

- [冻结来源协议](protocol.json)
- [初始获取记录](acquisition.json)、[最终获取记录](acquisition_complete.json)
- [补充传输协议](transport_protocol.json)、[传输校验记录](transport_receipt.json)
- [来源准备](../../scripts/defense/prepare_phase1_attack_assets.py)、[文件传输](../../scripts/defense/relay_phase1_public_inputs.py)、[249导入核验](../../scripts/defense/import_phase1_public_inputs.py)
- 服务器：`/data0/liyanlei/watermark_research/runs/phase1_public_inputs_20261113`
