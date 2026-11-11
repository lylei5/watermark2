# 249 服务器目录清理记录

日期：2026-11-11。范围：`/data0/liyanlei`。

成功删除 13 个文件，释放已分配空间 13.613 GiB；删除失败 0 项。目录实测占用从 223.499 GiB 降至 209.887 GiB。目录差值与删除量的少量差异来自审计文件写入。

## 删除清单

| 相对路径 | 释放 GiB | 依据 |
|---|---:|---|
| `huggingface/hub/models--runwayml--stable-diffusion-v1-5/blobs/6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa.incomplete` | 1.4063 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `huggingface/hub/models--runwayml--stable-diffusion-v1-5/blobs/1a189f0be69d6106a48548e7626207dddd7042a418dbf372cefd05e0cdba61b6.incomplete` | 1.4648 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `huggingface/hub/models--runwayml--stable-diffusion-v1-5/blobs/cd1b6db09a81cb1d39fbd245a89c1e3db9da9fe8eba5e8f9098ea6c4994221d3.incomplete` | 1.7969 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `huggingface/hub/models--runwayml--stable-diffusion-v1-5/blobs/c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21.incomplete` | 1.3867 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `watermark_research/models/sembind/dinov2-giant/.cache/huggingface/download/Q1p2l2BzM1m6P5jKvr8WTq1TUio=.55755116234d40720deebc04dcf8b8c146e0e7dd5aace4a413eb203604c44762.incomplete` | 0.7153 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `watermark_research/models/sembind/dinov2-giant/.cache/huggingface/download/xGOKKLRSlIhH692hSVvI1-gpoa8=.917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f.incomplete` | 0.3747 | 旧的未完成下载，已检查文件未被当前用户进程打开 |
| `natural_image_512bit_watermark_poc/datasets/div2k/DIV2K_train_HR.zip` | 3.2881 | 全部压缩包成员 SHA-256 与保留的解压文件一致 |
| `FontGuard_repro/fontguard_ckpt.zip` | 1.8245 | 全部压缩包成员 SHA-256 与保留的解压文件一致 |
| `watermark_research/runs/geometry_calibration_payload_20261110.zip` | 0.8652 | 全部压缩包成员 SHA-256 与保留的解压文件一致 |
| `watermark_research/runs/benign_dev_assets.zip` | 0.1020 | 与保留副本 SHA-256 一致 |
| `FontGuard_repro/partial/exp_data/base_sty_feat_CH.pth` | 0.0001 | 与保留副本 SHA-256 一致 |
| `FontGuard_repro/partial/exp_data/clip_cls_CH.pt` | 0.3882 | 与保留副本 SHA-256 一致 |
| `FontGuard_repro/partial/exp_data/GB2312_CN6763.txt` | 0.0000 | 与保留副本 SHA-256 一致 |

三个解压目录分别核对 800、16、212 个文件；所有成员均一致。保留副本路径与逐文件 SHA-256 见 `archive_checks.json`、`deletion_results.json`。删除前复核文件所有者、路径范围、大小、修改时间、inode、硬链接数和当前用户进程打开文件。

## 清理后一级目录

| 目录 | GiB |
|---|---:|
| `watermark_research` | 48.651 |
| `genstego_sota_assets` | 44.410 |
| `ceramic_watermark` | 39.044 |
| `genstego_sota_results` | 23.060 |
| `envs` | 15.417 |
| `genstego_sota_lab` | 12.378 |
| `natural_image_512bit_watermark_poc` | 11.746 |
| `FontGuard_repro` | 9.050 |
| `huggingface` | 6.058 |
| `third_party` | 0.033 |
| `trajectory_orth_20261103` | 0.019 |
| `logs` | 0.014 |
| `official_hidden_compare` | 0.005 |
| `bin` | 0.000 |

## 保留资产与依赖

- 当前水印研究的模型、完整实验记录、私有验证资产、运行环境和正在安装的隔离环境均保留。
- `ceramic_watermark/envs/miniconda` 是水印研究 Python 解释器的基础依赖，不能作为闲置旧项目环境删除。
- `genstego_sota_assets/modelscope_models` 约 33.85 GiB，包含不同精度、格式和 EMA 版本；本次未将它们认定为无用副本。
- `ceramic_watermark/experiments` 约 23.74 GiB、`genstego_sota_results` 约 23.06 GiB 包含实验记录；用途与保留期限未明确的内容继续保留。
- `FontGuard_repro/partial` 剩余约 0.69 GiB，未找到完整一致的保留副本，继续保留。
- 未操作其他用户目录，未改变或终止正在运行的任务。

共享 `/data0` 文件系统清理后仍显示 97% 使用率，快照可用空间约 260.80 GiB；其占用包含其他用户数据。

服务器记录：`/data0/liyanlei/watermark_research/runs/storage_audit_20261111/`。目录明细见 `before.tsv`、`after.tsv`；文件系统快照与扫描错误输出一并保留。
