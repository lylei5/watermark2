# 扩散水印的内容绑定与抗伪造研究

目标是在合法图检出率、图像质量与计算预算可比的条件下，提高语义水印抵抗迁移伪造与自适应攻击的能力。

## 当前进展

**已完成 296 个合法开发查询、36 个既有标准 Reprompt 候选评分，以及独立的 DPM 全精度正常图组件检查。** [最新评测报告](reports/sembind_official_audit_20261110/report.md)记录新增 80 个扰动查询、18 次 DPM 反演和同等合法检出率比较。内容模型目前没有优于校准后 SemBind BA 的证据；独立确认集尚未评分。

评测限定为已发表论文的标准基线、官方代码与既定参数；研究创新集中在防御方法。

## 当前可核查结果

最新研究报告：[SemBind 官方组件检查与合法扰动扩展](reports/sembind_official_audit_20261110/report.md)，日期 2026-11-10。SemBind-DDIM 在九种非裁剪条件下各为 16/16 通过，中心裁剪为 0/16；校准 BA 与内容验证器在相同 144/160 合法通过数下均拒绝全部 16 个既有 Reprompt 候选。DPM 组件检查的 8 张正常图在两种盲反演设置下全部通过。此前的[内容认证评测](reports/content_verifier_20261110/report.md)与[解码感知研究](reports/codeword_binding_20261109/report.md)保留开发证据。当前证据不支持宣称已达到 SOTA。

研究问题：重复纠错应消除传输误差，但可能同时消除防伪所需的内容差异。原型将语义码对齐到重复码字，使内容差异保留在解码后的消息层。CoverLock 已有消息级内容包装的先例；编码顺序本身不作为首创贡献。

- 官方语义网络：100 张自然图的四种版本，加上 30 对历史图像，共 460 条 float32 特征记录；CPU/GPU 重叠 120 条语义码逐位一致。
- 真实语义码与模拟信道：在预声明 25% 符号翻转条件下，跨图接受率由 5.50% 降至 0.63%，同时中心裁剪正例通过率由 100% 降至 93.25%。这些是信道实验，不能代替真实攻击结果。
- 实际 SD2.1/DDIM：5 个固定提示词的配对试验，包含原图、JPEG Q50、跨类别与同类场景 Reprompt；完整结果见报告。使用可复现开发密钥，尚无严格进程隔离或最终确认集。
- 历史 GS 原始结果为 30 对、30/30 通过；历史 SemBind 完成 30 对、2/30 通过。具体适用条件和证据路径见报告审计表。

## 实现与运行

- `scripts/defense/codeword_binding.py`：码字对齐、密钥化稀疏奇偶、GS 多数译码。
- `scripts/defense/test_codeword_binding.py`：代数不变量、覆盖与平票边界。
- `scripts/defense/extract_semantic_codes.py`：官方语义特征与输入哈希。
- `scripts/defense/codeword_channel_pilot.py`：共享随机量的配对信道实验。
- `scripts/defense/codeword_diffusion_pilot.py`、`codeword_hard_reprompt_pilot.py`、`codeword_degree2_pilot.py`：实际生成与验证。
- `protocols/codeword*.json`：冻结工作点与开发性实验说明。

```bash
python scripts/defense/test_codeword_binding.py
python scripts/defense/codeword_diffusion_pilot.py --help
```

服务器资产位于 `/data0/liyanlei/watermark_research`，SSH 别名 `249`。只使用即时确认空闲且索引不为 6 的 GPU，并按 UUID 绑定；不得终止其他用户任务。本次 GPU 实验运行于本机 RTX 4070 Laptop。

## 研究入口

- [当前状态](GPT_CONTEXT.md)与[机器可读状态](project_state.json)。
- [GS 解码契约](reports/22_decoder_contract/report.md)。
- [SemBind 官方 masker 核验](reports/repro_sembind_masker_20261103/report.md)。
- [既有防御实现](scripts/defense/)及[基线源码契约](docs/EVALUATION_CONTRACT.md)。

历史报告保留原始记录；当前数值和证据等级以最新研究报告为准。后续验证须完成官方基线闭环、同等绝对合法检出率校准与独立确认；仅使用已发表攻击，不开发或改进攻击方法。
