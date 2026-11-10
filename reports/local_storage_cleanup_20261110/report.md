# 本地研究资产清理与服务器计算策略

日期：2026-11-10。

后续全部 CPU/GPU 实验只在 249 服务器执行。本机不再运行实验、充当显卡后备或下载研究模型权重；可继续编辑代码和文档。服务器限制保持 GPU7、5、4，最多三张，不持续轮询。

已删除 15 个文件，共 14.61 GB（13.61 GiB，按文件逻辑大小统计）。包括本地 SD2.1 半精度与全精度权重、DINOv2、SemBind 语义头、AlexNet、四份 CoverLock 权重，以及两份未完成的模型下载缓存。模型配置、小型源码和目录结构保留。

删除前，13 份模型权重都核验了服务器副本。除半精度 UNet 使用有效张量哈希与保留的全精度源文件建立一致性证据外，其余均逐字节 SHA-256 一致。四份 CoverLock 权重额外保存在服务器 `/data0/liyanlei/watermark_research/runs/local_model_archive_20261110/coverlock/`。半精度实验的有效模型位于服务器 `/data0/liyanlei/watermark_research/runs/geometry_calibration_20261110/model/`。

代码仓库、论文 PDF、实验报告、模型配置、密钥原始记录、实验图像和反演缓存保留。未清理其它项目或系统共享缓存。现有服务器实验及接续任务不受本地清理影响。

11 个临时 ZIP 包（合计 1.11 GB）有经哈希核验的服务器副本，但自动审批拒绝了删除命令，返回 `blocked by policy`，没有具体说明，因此仍留在本地研究 work 目录。首次包含递归缓存清理的批量命令也被拦截；最终采用明确文件路径的非递归操作完成权重与未完成下载清理。没有绕过审批执行 ZIP 删除。

逐文件结果见 [deletion_result.json](deletion_result.json)，服务器副本核验见 [server_verification.json](server_verification.json)。
