# 连续语义解码区域：干净图筛选结果

六个新来源、四组方法的完整干净图筛选已完成。24 条反演记录和 120 条错误身份判决通过逐条审计。所有组均接受 6/6 干净图，但主候选 continuous_transport 未通过预先冻结的绑定偏差和比特准确率门槛，`advance=false`。没有执行攻击任务，不能报告防伪接受率或鲁棒性优势。

| 方法 | 干净图接受 | 平均 BA | 归一化 logits 登记偏差 | 与辅助图的 LPIPS | 初始潜变量 MSE | 错误身份接受 |
|---|---:|---:|---:|---:|---:|---:|
| full | 6/6 | 1.000000 | 0.140400 | 0.756393 | 1.995248 | 0/30 |
| quantized_geometry | 6/6 | 1.000000 | 0.274732 | 0.641091 | 0.832833 | 0/30 |
| continuous_independent | 6/6 | 0.969401 | 0.312098 | 0.731283 | 1.996159 | 0/30 |
| continuous_transport | 6/6 | 0.968750 | 0.322839 | 0.644119 | 0.836780 | 0/30 |

登记偏差是辅助图与最终图归一化 hash-head logits 的欧氏距离。LPIPS 是与辅助预览的配对差异，不是总体生成质量指标。错误身份比较共享六个来源，0/30 不能证明极低经验 FPR。

## 机制判断

连续传输降低了与辅助预览的 LPIPS，但登记偏差比 full 和连续独立采样更大；平均 BA 也低于量化区域对照。实际生成过程没有满足本轮所需的“平滑内容变化同时保留绑定与译码”的机制条件。单纯把二值绑定变成连续旋转不足以得到本轮设定中的改进。

[量化对照等价检查](quantized_control_equivalence.json)中，16 个身份的嵌入差异为 0，任意观测下的译码全部一致：该对照等价于固定秘密旋转坐标系中的二值语义掩码。它是实现等价检查，不是新的性能实验。[理想信道分析](geometry_tradeoff.json)和[冻结计划](plan.md)说明参数与假设，不替代上述真实生成结果。

## 执行与复核

运行目录为 `/data0/liyanlei/watermark_research/runs/continuous_geometry_screen_20261113_v2`，完成于 2026-11-13 00:41:17 CST。生成和反演各 50 步，`inv_order=0`、`decoder_inv=True`，VAE 优化 100 步；256 位身份、64 次重复，原始 `BA > 181/256` 不变。使用 GPU 7、5、4，未使用 GPU 6。

- [执行协议](../../protocols/continuous_geometry_screen_20261113_v2.json)，SHA256：`45787294dc25d21666d1378cbca0a4d5de25b0bea5f913f8dab07340cf587fc9`。
- [逐条核验](genuine_verified.json)：输入输出哈希、独立复数旋转和直接 CDF 嵌入重建、硬多数译码与阈值复算、CPU LPIPS；密钥相位辅助函数与工作进程共用。
- [推进门槛](pilot_decision.json)：clean / binding / continuous_vs_quantized / fidelity / wrong_identity 分别为通过 / 未通过 / 未通过 / 通过 / 通过。
- [最终执行状态](pipeline_state.json)，攻击任务数 0。

[首次启动记录](initial_launch_state.json)保留了 GPU 工作开始前的 `os.pidfd_open` 兼容性失败。v2 使用系统调用等待退出，保留原定方法、来源和门槛，并重新生成私有身份。没有将首次启动计为已完成实验。

本实验使用 SemBind 发布组件及项目载体，不是完整原生 SemBind 复现。六来源干净图筛选不构成匹配合法检出率的防伪比较，也不支持 SOTA 主张。
