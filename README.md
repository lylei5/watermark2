# 扩散模型水印防御研究

本仓库研究扩散模型水印的防御方法，重点是内容绑定、抗伪造验证、合法编辑鲁棒性与误报控制。目标是在合法图检出率、图像质量与计算预算可比的条件下提升防御效果。

攻击基线仅用于防御评测，采用顶会顶刊已发表方法的既定实现与完整预算。研究创新集中于防御方法。

后续 CPU/GPU 实验统一在 249 服务器执行，本机仅保留代码、文档和必要研究记录，不再作为实验算力后备。

## 当前进展

[配对九条件实验](reports/sembind_published_revalidation_20261112/report.md)已完成：SemBind/GS在两种反演设置下均36/36接受，144条记录独立复核通过。完整151次更新Imprint正在执行。新增[条件内容似然防御实验](reports/conditional_content_likelihood_20261112/report.md)已完成真实图留出测试：30/40接受，低于同策略校准BA的39/40，保留负结果。下一项[条件内容锚嵌入实验](reports/conditional_anchor_pilot_20261112/plan.md)已冻结并排入进程退出事件任务链：6个新来源、4组嵌入、240次完整验证，达到预设条件后自动进行完整已发表Imprint评测。尚未建立优于SemBind的结论。下列此前结果保留作历史研究记录，关键结论须经新执行验证。

[完整预算 Imprint 评测](reports/full_imprint_20261111/report.md)已完成：8 个案例各151次更新，全部144张图完成验证。最终候选 SemBind 项目适配器接受0/4、GS接受4/4；五种冻结判定规则在全部图上完全一致，几何验证器没有额外防伪收益。

[32来源校准](reports/geometry_calibration_20261111/completed/report.md)与[64来源独立确认](reports/geometry_confirmation_20261111/report.md)均已完成。确认集384张合法查询：默认BA通过319张，全组几何验证通过382张，分组版本通过381张；三者在24,192次错误身份判决中均为0次接受。收益来自中心裁剪后放大的恢复，不能据此宣称新增防伪收益或SOTA。

[Gaussian Shading官方九条件评测](reports/gs_published_conditions_20261111/report.md)也已完成：默认554/576、全组几何552/576、分组529/576，均0/36,288错误身份接受。本轮几何搜索没有额外收益；官方填黑裁剪与中心裁剪后放大分别报告，不能合并为普适鲁棒性提升。

[100张未加水印自然图评测](reports/unwatermarked_validation_20261111/report.md)已完成：完整DDIM50及6,400次身份判决，五种固定规则均0次误接受，按查询为0/100。降低BA门限的90%、95%、100%校准工作点分别误接受472、1,385、6,337次。该轮使用已有开发资产，不能据此证明极低FPR。该历史评测流水线已结束；当前重验状态见上方。

## 当前可核查结果

此前开发报告：[SemBind 官方组件检查与合法扰动扩展](reports/sembind_official_audit_20261110/report.md)，日期 2026-11-10。SemBind-DDIM 在九种非裁剪条件下各为 16/16 通过，中心裁剪为 0/16；校准 BA 与内容验证器在相同 144/160 合法通过数下均拒绝全部 16 个既有 Reprompt 候选。DPM 组件检查的 8 张正常图在两种盲反演设置下全部通过。此前的[内容认证评测](reports/content_verifier_20261110/report.md)与[解码感知研究](reports/codeword_binding_20261109/report.md)保留开发证据。当前证据不支持宣称已达到 SOTA。

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

服务器资产位于 `/data0/liyanlei/watermark_research`，SSH 别名 `249`。当前只使用 GPU7、5、4，最多三张；启动前确认空闲并按 UUID 绑定，不得终止其他用户任务。完整预算任务通过进程退出事件接续评测，不持续轮询。

## 研究入口

- [文献与历史结论审计](reports/repository_audit_20261111/report.md)、[核验后的文献](docs/LITERATURE.md)及[防御研究计划](docs/RESEARCH_PLAN.md)。

- [当前状态](GPT_CONTEXT.md)与[机器可读状态](project_state.json)。
- [GS 解码契约](reports/22_decoder_contract/report.md)。
- [SemBind 官方 masker 核验](reports/repro_sembind_masker_20261103/report.md)。
- [既有防御实现](scripts/defense/)及[基线源码契约](docs/EVALUATION_CONTRACT.md)。

历史报告保留原始记录；当前数值和证据等级以最新研究报告为准。后续验证须完成官方基线闭环、同等绝对合法检出率校准与独立确认；仅使用已发表攻击，不开发或改进攻击方法。
