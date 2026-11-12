# 100 对确认资产：封存证据

100 张 COCO 载体、100 条唯一 caption 及固定配对已准备并封存。200 个来源身份互不重叠；本项是数据准备，没有启动 100 对模型实验。实际图片留在 249 的 `/data0/liyanlei/watermark_research/datasets/confirmation100_relay_v11/`。

[封存清单](manifest.json) SHA256：`a41bbad8a755418cbc06f83a6660f170dd0b48d5b9cb31d31d434d42657c8220`。本次仓库同步补齐服务器已有的清单与核验 JSON；不是重新执行图像生成或数据选择。

## 可复核记录

- [冻结资产协议](../../protocols/confirmation100_assets_v11.json)和[离线完成协议](../../protocols/confirmation100_v11_completion.json)。
- [确定性选择重放](selection_replay_verification.json)：857 个候选来源、100 个载体、100 条唯一 caption、200 个来源身份，100 张原图与 100 张 RGB512 PNG 的哈希核对通过。
- [数据完整性核验](verification.json)和[结果封存](result_seal.json)。
- [既有数据保留核验](prior_data_preservation.json)：302 个既有文件字节未变，仅补齐最终一张原图的规范路径及 PNG。
- [历史恢复核验收据](recovery_verification_20261025.json)：区分当时重新核验与仅引用的历史检查。

网络采集曾在最后一张就绪记录交接时失败；离线完成复用既有收据和图像，没有重新选择来源。原失败与补齐记录在服务器保留。仓库未找回的历史中转报告、差异文件及本地专项检查不作为当前可点击证据或重新通过的检查。

## 使用范围

来源元数据声明与当前图片再分发权利不是同一证明。100 张载体作者账户未知，重分发状态未获确认；仓库保留元数据与哈希，不发布原图或 PNG。caption 来源和许可归属在清单中保留。

历史排除记录不能证明完全未用于开发或基础模型训练。该批资产不自动构成新方法的独立确认集；当前研究状态见[机器可读记录](../../project_state.json)。本项没有生成水印秘密或认证凭据，也没有模型/GPU评分结果。
