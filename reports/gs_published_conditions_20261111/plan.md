# Gaussian Shading 官方九条件评测

本轮直接调用 CVPR 2024 Gaussian Shading 官方 `image_utils.py` 的 `image_distortion`，采用官方 `scripts/run.sh` 中的全部九项失真设置。上游版本固定为 `09c678fadc7545acf7be12647ddf2a5e66f6a9dc`，文件哈希记录于 `upstream.json`；原函数、参数强度均保持不变。每个条件单独应用，不组合或优化攻击。

| 上游参数 | 固定值 |
|---|---:|
| random_crop_ratio | 0.6 |
| random_drop_ratio | 0.8 |
| brightness_factor | 6 |
| jpeg_ratio | 25 |
| gaussian_blur_r | 4 |
| gaussian_std | 0.05 |
| median_blur_k | 7 |
| sp_prob | 0.05 |
| resize_ratio | 0.25 |

使用独立确认的同一组 64 个 SemBind 项目参考图，每图九个查询，共 576 张。分别验证全部 64 个身份，得到 36,864 次判决，其中 36,288 次错误身份组合。每个任务的随机种子为 `2026111100 + task`，在调用每个上游条件前固定。默认 BA、几何检测器及四个校准 BA 阈值均沿用已冻结配置，不根据本轮结果修改。

官方 crop 保留随机窗口并将其余位置填黑，crop/drop 比例直接乘以边长；这与项目的中心裁剪后放大不同。按官方函数运行的输出和九项设置逐项报告，不能把当前校准的 crop90 收益直接外推至本轮。

六条件确认与九条件评测共享来源，应分别报告而非当作两组独立来源。两轮协议均在查看确认检测结果前固定。本轮复现的是官方图像失真组件及其参数，水印参考仍为项目 SemBind/DDIM50 适配器，不构成 Gaussian Shading 完整方法复现。

执行通过前一轮监督进程的退出事件接续，成功完成后才启动；只使用空闲 GPU7、GPU5，最多两张卡。每张查询使用完整 DDIM50，三个分片完成后自动 CPU 评分。任务和失败状态写入服务器运行目录，不进行定时轮询。

协议：[gs_published_conditions_20261111.json](../../protocols/gs_published_conditions_20261111.json)。官方来源：[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.html)、[源码](https://github.com/bsmhmmlf/Gaussian-Shading/blob/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/image_utils.py)、[参数](https://github.com/bsmhmmlf/Gaussian-Shading/blob/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/scripts/run.sh)。
