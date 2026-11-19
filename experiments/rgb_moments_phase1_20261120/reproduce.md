# 249 上的复核入口

完整运行目录为 `/data0/liyanlei/watermark_research/runs/rgb_moments_phase1_20261120`。三个子目录 `pilot`、`refine`、`test` 按样本 ID 分目录保存全分辨率 PNG 和 `record.json`。记录包含生成种子、共享潜变量 SHA-256、各方法每步 loss 和六项统计、选中的迭代步及运行时间。提示词、供体种子、目标向量、供体文件路径和哈希在对应的来源清单内。

生成使用项目的 `envs/seal_reproduction_py310/bin/python`，评价使用 `envs/semantic_forgery_py310/bin/python`。生成时启用已有离线 Hugging Face 缓存，调度器检查 4、5、7 号卡的进程、占用和利用率并按 UUID 绑定。评价设置 `CUDA_VISIBLE_DEVICES=''`。在项目根目录执行下列命令，可从现有冻结图像重新核算独立测试。

```bash
r=$PWD/runs/rgb_moments_phase1_20261120
p=$PWD/envs/semantic_forgery_py310/bin/python
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$p" "$r/quality.py" test
"$p" "$r/edits.py"
"$p" "$r/fid.py" test
"$p" "$r/analyze.py"
```

生成阶段先执行 `prepare.py`，将生成的 `pilot_sources.json` 原样复制为 `refine_sources.json`，依次执行 `generate.py pilot`、`generate.py refine`、`generate.py test`。也可以从本目录已冻结的来源清单开始。Pilot 两轮和测试分别读取 `pilot_config.json`、`refine_config.json`、`test_config.json`。每个生成进程在模型初始化后预热。另起生成试验时，将 `prepare.py` 的 `RUN` 指向新的独立目录，放入脚本与配置，并为评分准备上述模型和评价资源。

FID 特征取自独立目录 `assets/lib` 中的 `pytorch-fid==0.3.0`。作者发布的权重保存在 `assets/fid.pth`，完整 SHA-256 为 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`。`fid.py reference` 按哈希选取 256 张本地 COCO test2017 参考图并保存 pool3 特征。裁剪和缩放取自 Stable Signature 作者的 `utils_img.py`，Git blob 为 `a412156bd8368c3aa8cbeaa6b1fbd3ad2fb7e0a5`。评价脚本核实文件摘要后读取其中的两个函数。LPIPS 复用本地 AlexNet 权重，CLIP 复用项目 `phase1_backend_audit_20261113_v2` 登记的 ViT-L/14。[FID 实现与权重](https://github.com/mseitzer/pytorch-fid) · [Stable Signature 作者实现](https://github.com/facebookresearch/stable_signature)
