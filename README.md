# 扩散模型水印防御研究

本仓库研究内容绑定扩散水印的防御机制。有效改进必须同时有合法检出率、误接受、图像质量和计算预算的可比证据。当前尚未建立优于原生 SemBind 的结果。

## 当前状态

**SemBind 完整原生复现尚未完成。** 当前代码使用作者发布的语义网络与 DPM 生成/反演组件，载体、绑定和评测编排属于项目实现。当前执行[第一阶段协议](docs/PHASE1_PROTOCOL.md)：论文规格基线、工作点及正常去噪轨迹语义测量；载体变体暂停。

[第一阶段规格与工作点核验](reports/phase1_contract_20261113/report.md)已完成，模型实验正在准备。此前完成的[连续语义区域筛选](reports/continuous_geometry_screen_20261113/report.md)有 24 条干净图验证：四组均接受 6/6，但主候选未通过绑定偏差和比特准确率门槛，没有执行攻击评测。它不支持鲁棒性或防伪优势。

| 已完成研究 | 主要结果 | 结论范围 |
|---|---|---|
| [共享语义证据](reports/shared_semantic_evidence_20261112/report.md) | 默认合法接受 107/120、最终 Imprint 接受 1/12；候选 85/120、0/12 | 未建立匹配合法检出率下的增益 |
| [固定语义反馈](reports/semantic_feedback_pilot_20261112/report.md) | full / feedback / consensus 为 58 / 56 / 57，分母均为 60 | 未通过冻结门槛 |
| [最小反射区域](reports/decision_region_pilot_20261112/report.md) | 主候选 44/60，full 59/60 | 未通过冻结门槛 |
| [完整码字条件传输](reports/semantic_transport_pilot_20261112/report.md) | 主候选 55/60，full 53/60，同符号重排 57/60 | 保真改善，绑定和合法接受门槛未通过 |
| [连续语义区域](reports/continuous_geometry_screen_20261113/report.md) | 主候选平均 BA 0.968750，量化对照 1.000000 | 仅六来源干净图筛选，未通过推进门槛 |

各实验使用不同来源，表格只作证据导航，不能跨行比较方法排名。完整失败结果和冻结协议见[报告索引](reports/README.md)。

## 当前实现

| 功能 | 代码与实际范围 |
|---|---|
| 1024 位语义码提取 | [semantic_code.py](scripts/defense/semantic_code.py)，仅语义提取，无水印绑定接口 |
| 官方生成与反演组件 | [sembind_official_audit.py](scripts/defense/sembind_official_audit.py) 的 `load_official()`，加载发布的 `inversion.exact_inversion` 与原始 diffusers 0.21.4 DPM scheduler |
| 完整预算实验工作进程 | [run_anchor_binding_worker.py](scripts/defense/run_anchor_binding_worker.py)：生成/反演各 50 步，`inv_order=0`、`decoder_inv=True`，VAE decoder 优化 100 步 |
| 项目 full 载体 | [anchor_binding.py](scripts/defense/anchor_binding.py)：私有 `binding_key` 派生 HMAC 置换、256 位消息、64 次重复 |
| 既有攻击基线调用 | [run_published_imprint_full.py](scripts/defense/run_published_imprint_full.py)，固定上游实现，完整 151 次更新 |

`inv_order=0` 关闭扩散反演中的固定点校正；VAE decoder 优化是另一项操作。论文默认 inverse order 也为 0，但调用该组件本身不证明完整原生复现。[官方论文 §4.1](https://arxiv.org/html/2601.20310v1)的理论 FPR 工作点为 1e-6；本项目的 `BA > 181/256` 尚未与其对齐。

现行代码与已执行版本分别保存：已完成结果对应服务器 `runs/<run>/` 中冻结源码及协议哈希；复核这些结果应使用该冻结版本。当前源码清理不会改写既有实验的来源记录。

## 工作约束与入口

实验仅在 SSH 主机 `249` 的 `/data0/liyanlei/watermark_research` 执行。仅使用空闲 GPU 7、5、4，最多三张，始终排除 GPU 6；不持续轮询。模型、图像与私钥留在服务器。

防御评测只使用顶会顶刊已经发表的攻击基线及其既定实现和完整预算，不创造或改进攻击。所有论文主张需要新的独立确认，并报告实际合法检出率与误接受。

- [机器可读状态](project_state.json)、[研究交接](GPT_CONTEXT.md)、[研究计划](docs/RESEARCH_PLAN.md)
- [基线复现清单](docs/DEFENSE_BASELINE_INVENTORY.md)、[文献](docs/LITERATURE.md)、[第三方来源](THIRD_PARTY.md)
- [仓库同步核验](reports/repository_sync_20261113/report.md)、[操作规则](AGENTS.md)
