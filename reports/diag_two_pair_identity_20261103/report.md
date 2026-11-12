# 逐对身份诊断（第 1 对，2026-11-03）

## 0. 一句话

把 GS 身份从"全 30 对共用 offset=0"改成**逐对独立**后，第 1 对的候选 BA 从 **0.8711 降到 0.8086（−6.25 个百分点）**，但**仍过 0.707 阈值**；参考图在自己身份下 BA=1.0、在旧共享身份下只有 0.5078，证明身份确实换了。图像质量不变（PSNR 26.35 vs 26.41）。

## 1. 设置

- 任务：confirmation100 design v7 的 task 1，`reference_seed=33101`
- 身份：`offset = reference_seed % 10000 = 3101`（官方 MESSAGES/KEYS/NONCES 各 10000 条中的真实条目）
- 攻击：Imprint，30 次更新，代理 SD2.1 → 目标 SDXL
- 设备：GPU3，`GPU-e602f6b2-50a3-72ee-ae1b-a0ae80eca5da`
- 性质：**诊断，非冻结协议运行**

## 2. 结果

| 量 | 值 | 说明 |
|---|---:|---|
| `reference_ba_own` | **1.0000** | 参考图在自己身份下完全有效 |
| `reference_ba_shared` | **0.5078** | 同一张参考图在旧共享身份下等于随机 → 身份确实不同 |
| `candidate_ba_own` | **0.8086** | 逐对身份下的真实攻击结果 |
| `candidate_ba_shared` | 0.5234 | 同一张伪造图用旧身份判 → 随机（预期，参考是用新身份生成的） |
| `candidate_pass_own` | **true** | 仍通过 |
| PSNR | 26.35 dB | 基线记录 26.41 dB |
| 耗时 | 1257 秒 | |

## 3. 与已记录 30 对结果的对比

| | 共享身份（已记录） | 逐对身份（本次） | 差 |
|---|---:|---:|---:|
| 候选 BA | 0.8711 | 0.8086 | **−0.0625** |
| PSNR | 26.41 dB | 26.35 dB | −0.06 dB |
| 通过 | 是 | 是 | 不变 |

## 4. 判读

1. **共享身份确实带来乐观偏差**，量级约 6 个百分点（单对证据）。
2. **但通过/不通过的结论没变**：0.8086 仍明显高于 0.707。
3. **PSNR 基本不受影响**，说明偏差来自评估端身份，不是攻击变弱。
4. 只有 1 对，**不能据此断言全 30 对的偏差都是 6 个百分点**；身份间方差未知，这正是需要全量重跑的理由。

## 5. 已知缺陷（本次自查发现）

脚本里的 `fingerprint()` 最初读 `wm.messages[0]`，而 `GsProvider` 保留完整列表、在 `get_wm_latents` 内部才用 `offset` 索引，所以记录的 `identity_own` 字段与 `identity_shared` 相同、`identities_differ=false` 是**记录字段的 bug**。

**实质数据不受影响**，因为 `reference_ba_shared=0.5078` 已经从行为上证明两次评估用的是不同身份。脚本已修正为 `messages[wm.offset]`，后续运行会正确记录指纹。

## 6. 未完成

第 2 对（`offset=3102`）在跑到参考图生成后，按用户"先跑 SemBind 和 PGID"的指示停止，GPU3 让给 PGID。产物保留：`2_cover.png`、`2_own_noise.npy`、`2_reference.png`。

## 7. 文件

- 脚本：[执行版本源码](https://github.com/lylei5/watermark2/blob/ba6283f0e93f86a03f1636f5515fb91d10562bbf/scripts/confirmation100/diag_two_pair_identity.py)（已修正指纹）
- 结果：`runs/diag_two_pair_identity_20261103/1_result.json`（服务器）
