# SemBind masker 复现核验（2026-11-03）

## 0. 一句话

我们重写的 `SemanticMasker.code()` 与作者发布的官方入口 `AttnSupConHashingModel.forward(stage='binarize', s=12.0)` 在 20 张图上**位级完全相同（Hamming 恒为 0）**，含失真变体。这是 SemBind 唯一被公开的组件，现已确认忠实复现。

## 1. 为什么只核验 masker

SemBind 官方库（`third_party/SemBind` @ `337c9f90a150157ac1d410208ae20e6530ca8cdd`）发布的内容是：

- `model.py` + `sembind_masker.pth`（886,343,333 bytes）：语义哈希 masker
- `inversion.py`：含 `exact_inversion`（默认 `inv_order=1`，论文用 0）
- `semantimark.ipynb`：只演示**哈希**（生成 10 张原图 + 10 张失真图，打印 hex code）

**绑定/解绑代数（S ⊙ zT、置换 π、σ）没有发布。** notebook 里没有 π、没有 σ、没有嵌入或解绑代码。所以本项目的 `scripts/defense/sembind.py` 是按论文重写的绑定链路，不属于"复现"，无法与作者实现做数值对照。可对照的只有 masker。

## 2. 核验方法

`scripts/defense/repro/sembind_masker_equiv.py`，GPU3（`GPU-e602f6b2-50a3-72ee-ae1b-a0ae80eca5da`），20 张 confirmation100 v1.1 封面。

三条检查：

- **R1 位级一致**：同一图像分别走官方 `forward(..., stage='binarize', s=12.0)` 与我们的 `SemanticMasker.code()`，比 1024 位码的 Hamming 距离。
- **R2 失真稳定**：同内容 + 轻度失真（高斯模糊 r=1.0 + 0.8 缩放往返），看 Hamming 变化。
- **R3 跨图区分**：不同图像两两比较。

## 3. 结果

| 检查 | 结果 |
|---|---|
| R1 官方 vs 我们（原图） | **identical = true，max Hamming = 0** |
| R1 官方 vs 我们（失真图） | **max Hamming = 0** |
| R2 同内容+失真 | 均值 20.55/1024（2.0%），中位 9，min 1，max 78 |
| R3 跨图（190 对） | 均值 501.7/1024，中位 520，min **29**，max 571 |

参考：随机基线为 512/1024。

## 4. 两个结论

**（1）masker 复现成立。** 位级零差异，说明我们的预处理（DINOv2 CLS → `F.normalize`）、前向顺序（encoder → proj_head → hash_head）、饱和常数 s=12.0、取位规则（sign → {0,1}）全部与作者一致。R2 也复现了 notebook 展示的核心性质：轻度失真下哈希稳定。

**（2）跨图出现自然碰撞。** 190 对不同图像里，最相似的一对 Hamming 仅 **29/1024（一致率 97.2%）**。这说明 1024 位语义哈希不是密码学抗碰撞的——两个**内容不同**的封面可以自然撞到近乎相同的码。这条为"掩码碰撞"攻击路线提供了独立于优化实验的直接证据。

## 5. 文件

- 脚本：`scripts/defense/repro/sembind_masker_equiv.py`
- 结果：`runs/repro_sembind_masker_20261103/masker_equiv.json`（服务器）
- 设备：GPU3，UUID `GPU-e602f6b2-50a3-72ee-ae1b-a0ae80eca5da`，`cuda:0`，单卡串行

## 6. 仍未复现

- 绑定/解绑链路（作者未发布）
- E.1/Table 17 的 `Det.=0.00`：需 5 个独立 proxy masker，本地只有 1 个 checkpoint
- 精确反演对齐（`inv_order=0`）与 150 步预算：下一步

---

## 7. 补充（2026-11-03）：官方精确反演无法运行

尝试用官方 `inversion.py::exact_inversion` 复现自检行为，遇到两个层面的阻塞。

### 7.1 结构性：官方反演只支持 SD2.1

`exact_inversion` 依赖 `InversableStableDiffusionPipeline` + `DPMSolverMultistepScheduler`，只对 SD2.1 成立。我们的基线目标是 SDXL，**无法套用**。所以真正的 SemBind 复现必须把目标模型换回 SD2.1（这也是论文设定）。

### 7.2 技术性：diffusers 版本漂移导致官方代码崩溃

本机 diffusers **0.31.0** 的签名已变：

```
convert_model_output(self, model_output, *args, sample=None, **kwargs)
```

而官方 `src/inverse_stable_diffusion.py:193` 仍按位置传参：

```python
model_s = self.scheduler.convert_model_output(noise_pred, t, latents)
```

第三个位置参数被 `*args` 吞掉，`sample` 保持 `None`，内部 sigma 退化为整个 51 元数组，最终在 64 维潜变量上广播失败：

```
RuntimeError: The size of tensor a (51) must match the size of tensor b (64) at non-singleton dimension 3
```

### 7.3 结论

**SemBind 的官方反演在当前 pinned 环境下不可运行。** 要跑通必须改第三方代码（把调用改成 `convert_model_output(noise_pred, sample=latents, timestep=t)`），改完就不再是"未修改的官方代码"，只能标为"打过补丁的官方代码"。

因此 SemBind 的复现现状：

| 组件 | 状态 |
|---|---|
| masker（1024 位语义哈希） | ✅ **位级复现**（Hamming 恒为 0） |
| 官方 exact_inversion | ❌ 环境不兼容，需打补丁 |
| 绑定/解绑代数（S、π、σ） | ❌ 作者未发布，是我们的重写 |
| E.1/Table 17 的 Det.=0.00 | ❌ 需 5 个独立 proxy masker，本地只有 1 个 |

脚本保留在 `scripts/defense/repro/sembind_exact_inversion.py`（含 `datasets` stub 处理），一旦决定打补丁即可继续。
