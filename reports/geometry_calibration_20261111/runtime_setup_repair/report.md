# 隔离运行环境安装修复

隔离环境已安装成功，11 个协议固定版本全部一致，实际导入检查及 `pip check` 均通过。完整 DDIM50 开发桥接通过原定 0.001 门限，GPU5 已启动校准缓存的第一个分片，工作进程 PID 为 42257；该状态是启动时快照，不代表校准全部完成。

| 开发查询 | 相对 L2 | 符号差异率 | 门限 |
|---|---:|---:|---:|
| clean | 0.000547554 | 0 | 0.001 |
| crop90 | 0.000569046 | 0.000183105 | 0.001 |

版本与依赖检查见 `installed_runtime.json`；桥接证据见 `bridge_runtime.json`。该结果支持恢复记录的软件环境后数值兼容性达到预定要求，但不单独归因于某一个软件包，也不表示防御性能已经提高。

2026-11-11，249 服务器。安装流程支持恢复已有隔离环境、保存前次日志，并使用进程锁防止重复启动。

安装源采用清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple/`。服务器实测该源能够安装 pip 25.0.1，并包含协议固定的 NumPy、Diffusers、SciPy、huggingface-hub、Pillow、cryptography、Accelerate 和 safetensors 版本。阿里云源在同一服务器多次发生连接重置。清华、中科大和腾讯镜像的 Triton 安装包均成功读取 256 KiB 片段，结果见 `mirror_probe.json`；该小样本只验证连接，不代表完整下载速度。

首次失败来自仅使用 PyTorch CUDA 专用源解析普通依赖：pip 23.0.1 拒绝 typing-extensions 的 wheel 元数据后转入源码构建，随后找不到构建依赖 flit_core。修复使用通用 PyPI 镜像、更新 pip 并限定二进制包，避免该构建路径。[pip 安装选项说明](https://pip.pypa.io/en/stable/cli/pip_install/)

PyTorch 2.1.2+cu121 和 torchvision 0.16.2+cu121 从服务器已完成的下载缓存恢复并安装，未重新下载大型安装包。恢复文件的 SHA-256 与字节数见 `recovered_wheels.json`。其原下载来源是 PyTorch 官方源；本次恢复是本地文件操作。

执行协议保留原有科学参数，仅更新安装脚本哈希并记录安装来源。服务器验证全部 13 个源文件哈希一致，新旧协议的科学参数完全一致；既有完整轨迹和冻结阈值的 6 项检查通过。实际启动第二个安装进程时，进程锁正确拒绝重复执行。

安装后的导入检查和 GPU5 空闲检查通过后，流程执行完整 DDIM50 开发桥接；相对 L2 门限仍为 0.001。通过后才运行 192 个校准查询、32 个身份的 6,144 次判决，三个分片在 GPU5 上顺序执行。独立确认集不参与本次流程。

安装状态、导入检查和桥接结果以服务器运行目录中的最新记录为准：`/data0/liyanlei/watermark_research/runs/geometry_calibration_replay_20261111/`。安装成功本身不构成防御效果证据。
