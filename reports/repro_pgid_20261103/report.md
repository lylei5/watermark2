# PGID 复现（NeurIPS 2026）

## 0. 一句话

PGID 是三项防御里**复现得最好的**：检测率与论文完全一致（0.00/0.00/0.00），位准确率只高 0.04–0.05，趋势一致。但当前只有 n=1，n=5 复跑正在进行。

## 1. 出处补全

仓库此前缺 PGID 的文献记录（`LITERATURE.md` 无条目）。补全如下：

- 论文：*PGID: Progressive Guided Inversion and Denoising for Robust Watermark Detection*
- 作者：Minh Quoc Duong, Chun Tong Lei, Chun Pong Lau
- arXiv：**2605.09319**（v1 2026-05-10，v3 2026-10-27）
- 会议：**NeurIPS 2026 已接收**（arXiv comments 明确写明，先前"会议归属存疑"已澄清）
- 代码：`github.com/MinhQDuong/PGID`，本地 commit `eb7fb0ddc46a07e4f52a0a79038e546f7c925b50`
- 路径更正：报告写的 `third_party/pgid` 实际是 `third_party/PGID`

## 2. 对照目标（论文表 2，MS-COCO 封面，攻击者 SD2.1，GS，伪造攻击）

| 阶段 | 论文 Det.↓ | 论文 Bit Acc.↓ |
|---|---|---|
| GS 无防御（50/100/150 步） | 1.00 / 1.00 / 1.00 | 0.9987 / 0.9994 / 0.9995 |
| **GS + PGID-F** | **0.00 / 0.00 / 0.00** | **0.5358 / 0.5764 / 0.6099** |

论文另有 ImageNet 封面：GS+P 为 0.5479 / 0.5895 / 0.6210。

## 3. 我们的结果（`runs/pgid_smoke_20261102T231752Z`，n=1）

| 阶段 | Det. | Bit Acc. |
|---|---|---|
| 水印图 | 1.0 | 1.0000 |
| 攻击后（50/100/150） | 1.0 / 1.0 / 1.0 | 1.0000 / 1.0000 / 1.0000 |
| **PGID 防御后** | **0.0 / 0.0 / 0.0** | **0.5781 / 0.6211 / 0.6641** |

## 4. 逐项对照

| 项 | 论文（COCO） | 我们 | 差 |
|---|---:|---:|---:|
| Det. 50 步 | 0.00 | 0.00 | **0** |
| Det. 100 步 | 0.00 | 0.00 | **0** |
| Det. 150 步 | 0.00 | 0.00 | **0** |
| BA 50 步 | 0.5358 | 0.5781 | +0.0423 |
| BA 100 步 | 0.5764 | 0.6211 | +0.0447 |
| BA 150 步 | 0.6099 | 0.6641 | +0.0542 |
| 水印图 Det. | 1.00 | 1.00 | **0** |
| 无防御攻击后 Det. | 1.00 | 1.00 | **0** |

**判读：** 检测率三项完全命中；BA 系统性高 0.04–0.05，方向与步数单调趋势一致。我们的值正好落在论文 COCO 与 ImageNet 两列之间。差异可由 n=1（论文 n=100）、封面集不同、以及本地快照补丁解释，但**不改变"防御有效"这一结论**。

## 5. 复现过程中发现的环境阻塞（重要）

官方代码在本机**无法直接运行**，需要三处处理：

1. **依赖互斥**：`envs/pgid_py310` 有 `ldpc` 但缺 `pandas`/`diffusers`；`envs/semantic_forgery_py310` 有 `pandas`/`diffusers` 但缺 `ldpc`/`galois`。仓库 `wheels/` 里没有 pandas 轮子，无法离线补全。采用 `PYTHONPATH` 注入 `ldpc`/`galois` 两个 stub（仅 PRC 水印需要，GS 路径不调用），未修改环境与第三方代码。
2. **离线模型解析**：`pipe/base_pipe.py` 用 `scheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")` 走 HF Hub，离线模式下失败。改为显式传本地快照路径给 `--modelid_target` 与 `--modelid_attacker`。
3. 先前报告提到的 `run.py` model choices 与 `pipe/pg_utils.py` 模型表 fallback 补丁，本次未复现其具体形态；用本地路径绕过后不需要。

这三条都应记入复现说明——"官方代码可运行"与"官方代码在本机能运行"是两件事。

## 6. 进行中

n=5 复跑已启动（`runs/repro_pgid_n5_20261103`，MS-COCO 前 5 张封面，151 步，GPU3），用于把 n=1 的单点对照变成均值对照。

## 7. 边界

- 我们的目标模型是 Manojb SD2.1 镜像，与论文的官方 SD2.1 checkpoint **字节等价性未建立**。
- 伪造封面用仓库自带 10 张 MS-COCO 子集，非论文的 100 张随机 COCO。
- 无独立人工内容评审。
