# 三个防御基线的实现与验收

SemBind、PGID和SEAL已建立各自的原生入口，并保存源码、模型、协议和逐图产物哈希。以下是已完成的图像结果；各方法使用自己的载体与工作点，不构成同条件排名或全论文矩阵复现。执行状态只维护于[project_state.json](../../project_state.json)。

| 已完成链路 | 合法图 | 无水印对照 | 完整既有攻击 |
|---|---:|---:|---|
| 原GS空间OTP | 100/100 | 0/100 | 本链路未执行 |
| 论文补齐SemBind GS-S | 100/100 | 0/100 | 本链路未执行 |
| 论文补齐SemBind PRC-S | 6/6；512位消息均精确解码 | 0/6 | 本链路未执行 |
| SEAL公开代码 | 83/100；COCO48/50，SDP35/50 | 0/100 | 未执行 |
| SEAL论文私有salt实现 | 91/100；COCO50/50，SDP41/50 | 0/100 | 未执行 |
| PGID-R原生GS | 10/10 | 本轮无该对照 | Imprint去除后原检测0/10，防御后10/10；三个检查点一致 |

GS两条路径的166、167、182位工作点结果一致，详见[400条决策](../phase1_trajectory_native_gs_20261113/baseline_summary.json)。负例按提示词生成新身份，不代表固定用户自然图FPR。SEAL结果分别见[公开路径](seal_released_results.json)、[论文路径](seal_paper_results.json)。每图保存的1024块距离及FP16潜变量通过独立CPU重算，两条路径各核验400个图像/潜变量文件，计数判定完全一致。论文路径100张预览图与公开路径PNG哈希逐张一致。

[PGID-R结果](pgid_removal_results.json)来自原样调用官方run.py的10个案例，包含70条决策、40张图像和3份原始CSV的哈希检查。攻击执行完整151次更新，原源码索引50/100/150对应实际51/101/151次更新。三个检查点的原检测平均BA分别为0.2875、0.029296875、0.00234375，防御后分别为0.999609375、1、1。PGID-F的10例完整预算结果见[既有报告](../defense_revalidation_20261111/report.md)。公开fixture身份及已使用过的开发来源，不支持未知私钥安全或独立确认结论。

100例无水印零接受的双侧95% Clopper–Pearson上界约3.62%，不能写成经验FPR1e-6。SEAL两种派生方式对应不同水印图和密钥；91/100与83/100的差别不是私有salt改善性能的因果证据。

[SEAL干净图误差分解](seal_clean_diagnosis.json)用保存的生成时噪声作诊断参考，两种实现的100张图均可达到12块门槛；公开路径17次、论文路径9次实际漏检，都在重新从候选图提取语义参考后出现。该现象支持本批数据存在语义参考漂移问题。生成时参考对普通无数据库验证器不可用，这不是新的检测器、改进TPR或攻击结果。

[实现差异表](../../docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md)列明原代码、论文补齐、环境和未验收范围。SEAL七条件按原代码保留Noise0.1的共享偏移、Blur8的radius和Brightness6的实际区间；这些不能冒充不同定义的论文扰动。

## SEAL语义检测的七条件结果

[完整1400条决策](seal_benign_results.json)覆盖两个实现各100个来源、每图7个条件，2800个图像/潜变量文件通过哈希与独立分块距离核验。每格分母都是100，未剔除干净图漏检。200次Clean重测的caption、匹配块数和布尔判定与对应原结果相同。

| 条件 | 公开代码 | 论文salt实现 |
|---|---:|---:|
| Clean | 83/100 | 91/100 |
| Rot75 | 0/100 | 0/100 |
| JPEG25 | 44/100 | 48/100 |
| CropScale75 | 2/100 | 0/100 |
| Blur8 | 0/100 | 0/100 |
| Noise0.1（原代码定义） | 77/100 | 73/100 |
| Brightness6（原代码定义） | 49/100 | 53/100 |

本表使用候选自身caption和固定12块/2.3语义规则。官方另有image_transformations_exhaustive_tpr.py：逐块穷举128个码字，不使用候选语义，并按负例分数99百分位报告TPR@1%FPR；两个入口的接受集合和统计口径不同。本表不能冒充该穷举评测，也不能据此声称复现或推翻论文Figure6。当前采用固定开发来源和按task的set_random_seed约定；原扰动入口的外层NumPy seed为img_idx，本轮为task+3，随机样本序列不同，固定噪声强度与原函数保持不变。

## 固定来源与实现差异

| 方法 | 官方固定commit | 实现范围 |
|---|---|---|
| SemBind | 337c9f90a150157ac1d410208ae20e6530ca8cdd | 官方语义网络、DPM生成/精确反演；未发布的绑定由论文补齐 |
| PGID | dc4ad963ffefcb4cf84340084db617472983d3e0 | 官方F/R、四载体与原样实验入口 |
| SEAL | 92d31b31b93a6e373fe88a584c640beafb68c6fa | 官方生成/caption/SimHash/反演；最终布尔规则补自ICCV算法3 |

干净源码位于249的third_party/frozen。[来源核验](source_equivalence.json)显示PGID新旧22个Python文件可执行AST相同，SemBind使用的5个生成/反演/语义文件与官方归档字节一致。该检查不替代模型实验。

SemBind按论文使用平铺、比例截断、秘密置换和符号乘法；私有HMAC置换的具体派生是项目明确约定，未声称作者字节等价。DPM50、CFG7.5生成；空提示词CFG3、inv_order0、VAE decoder优化100步反演。sigma为GS/TR1、PRC/GS++0.5。[300次掩码检查](sembind_binding_checks.json)及6条原PRC私钥消息CPU往返通过。[PRC图像完整重放](sembind_prc_results.json)完成18次DPM反演、24条决策与48项文件哈希核验。原PRC和PRC-S均6/6接受、6/6精确恢复512位消息；两者无水印均0/6。其样本量仅能验证集成链路。

SEAL最终规则为1024块、每块7位、距离严格小于2.3、至少12块匹配。公开源码固定投影seed42，没有论文secret salt。论文路径由[专用实现](../../scripts/defense/seal_paper_simhash.py)补齐，明确SHA256序列化及设备内私有Gaussian RNG约定，不能声称作者实现等价。[私有salt组件检查](seal_paper_simhash_checks.json)验证确定性、密钥隔离和全局RNG不变。

[环境和权重](seal_preflight.json)、[配置](configuration_audit.json)验证BLIP2、作者caption embedding的上游LFS/Git blob哈希。补齐embedding原始元数据配置后，[3条CPU编码](seal_embedding_config_check.json)逐值相同。扩散模型为固定Manojb SD2.1镜像；与stabilityai原始fp16快照的同一性尚未认证，不能写成已经证实模型不同。

## 调度器和载体核验

[SEAL实际调度网格](scheduler_seal.json)为PNDMScheduler，请求50步产生51个时间步；保留原代码逆序DDIM代数反演。[SemBind与PGID检查](scheduler_sembind_pgid.json)记录两者不同的DPM反演时间网格，以及PGID原码完整控制流：F的GS/TR/T2SMark调用298次预测器，F的PRC为138次，R为200次。此处使用CPU替身预测器，不是图像性能测试。

[PGID四载体CPU检查](pgid_carrier_checks.json)各2/2通过；GS、PRC和T2SMark消息BA均为1，TR采用原生p-value。其他载体图像链路仍须逐项验收。[验收契约](../../docs/THREE_BASELINE_REPRODUCTION.md)列明完整矩阵范围。

## 失败与重放记录

PRC-S v2的GPU7分片在模型载入前被空闲检查拒绝：显存1MiB、利用率1%，完成决策数0；GPU5/4两个分片正常完成。保留失败目录及原协议。[v3协议](../../protocols/sembind_prc_native_20261113_v3.json)重放全部6例，数值代码、私钥、辅助图、载体和参数完全相同，不按结果筛选样本，也不放宽占用检查。重放三个分片均成功退出，6例均完整纳入汇总。

[重复运行对照](native_replay_audit.json)比较之前已完成的4个来源：8张生成图的哈希、查询语义码和16条最终判定/载荷结果一致，12个反演潜变量并非逐值相同，最大坐标差约0.14284。不能把源码相同写成运行逐位等价；另设[后端确定性检查](../../protocols/sembind_determinism_20261113.json)，保持50步DPM和100步VAE优化，比较默认与严格确定性设置。

TR-S初次GPU4分片同样在模型加载前因1MiB/1%未通过空闲检查，其余4个来源完成。[TR-S重放](../../protocols/sembind_tr_native_20261113_v2.json)保留全部数值代码和私有输入，只将协调器的固定设备释放等待延为15秒，空闲条件未放宽，未持续轮询实验。

已完成产物用冻结summarize_native_baseline.py逐条审计。后台等待器仅留下首项SEAL自动审计记录；其余已完成链路在本次恢复后显式审计。缺失的自动事件不能冒充已经验收；所有原始结果和未通过的启动记录保留。
