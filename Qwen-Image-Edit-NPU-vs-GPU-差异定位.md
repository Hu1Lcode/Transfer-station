# Qwen-Image-Edit-2511 NPU/GPU 数值差异定位与修复

## 背景

Qwen-Image-Edit-2511（图像编辑，diffusers 实现）在 **910B NPU** 与 **N 卡** 上，同一份权重、同一 prompt/image、同一 seed 推理，输出图存在差异。目标是定位差异来源并给出修复方案。

排查工具：[forward_probe.py](https://github.com/Hu1Lcode/Transfer-station/blob/main/forward_probe.py)

- probe 模式：把 diffusers `QwenImageTransformer2DModel` 的一次完整前向摘出来（输入由 seed 固定生成，与 train_v1.py 推理路径同形状），在 ±900 个子模块输出上打点（max/min/std/张量），两机各存一份 dump
- `--compare`：逐层输出 `max_abs / rel / cos / nan`，定位第一个超阈值层

## 一、定位过程

### 1. 输入层排除

```
img_in d_absmax = 0        pos_embed 1.2e-7        img_mod/txt_mod rel ≈ 1e-6
```

两机输入逐位相同，误差在模型内部产生。

### 2. 逐层趋势：非算子损坏型

| 层范围 | rel 误差 | 状态 |
|---|---|---|
| 0 ~ 28 | 0.4% ~ 2.3% | 干净（bf16 正常噪声） |
| 29 ~ 38 | 2.7% → 爬升 | 开始放大 |
| 39 ~ 44 | ~30% | 明显放大 |
| 45 ~ 59 | 稳定 ~74%，cos 0.975 | 饱和 |

特征：cos 始终 ≥ 0.975、误差"饱和"而非"发散" → 典型的数值噪声逐层累积，不是算子损坏（损坏会 rel 爆表 + cos 崩塌）。

### 3. 分布特征：文本路径 + added_kv attention 重灾

被标记的 576 行模块分布：

| 模块 | 行数 |
|---|---|
| txt_mlp.net | 155 |
| attn.to_add_out | 53 |
| attn.add_q/k/v_proj | 143 |
| attn.norm_added_q/k | 96 |
| txt_norm1/2 | 87 |

图像路径（img_mlp）几乎为零，偏差集中在文本路径与 added_kv 交叉注意力。

### 4. 澄清"norm 之后偏差大"的假象

norm 是线性归一化（只传递不放大），proj 是线性传播（误差水平保持），真正的放大点是 **attention softmax**（指数非线性）。norm 行 rel 高是因为归一化把输出幅值压到 ~1、rel 分母变小（观测偏差），不是误差在 norm 处变大。

### 5. 实验验证：`--force-rmsnorm-fp32`

跑 probe 时强制 RMSNorm 走 fp32 实现（绕开 npu_rms_norm），GPU vs NPU 对比：

| 层 | 默认（bf16 rmsnorm） | fp32 rmsnorm | 改善 |
|---|---|---|---|
| 29 | 2.72e-2 | 1.36e-2 | ↓50% |
| 39 | 2.98e-1 | 1.79e-1 | ↓40% |
| 45 | 7.79e-1 | 5.58e-2 | ↓93% |
| 59 | 7.40e-1 | 6.08e-2 | ↓92% |
| FINAL | 2.96e-2 / cos 0.99996 | 2.41e-2 / cos 0.99998 | 改善 |

→ 深层 ~90% 的偏差来自 `npu_rms_norm`。

### 6. 根因实锤

- 读 diffusers `models/normalization.py`：`RMSNorm.forward` 按 `is_torch_npu_available()` 分路，NPU 分支**先把输入降成 bf16**，再调 `torch_npu.npu_rms_norm`
- 实验：`npu_rms_norm` 传 fp32 输入**直接报错**（`aclnnRmsNorm failed`）→ 算子只支持 fp16/bf16
- 实验：单次 bf16 RMSNorm 相对误差实测 ≈ 0.4%（正好是 bf16 尾数精度 2⁻⁸）

## 二、问题本质

| 层面 | 内容 |
|---|---|
| **直接原因** | `torch_npu.npu_rms_norm` 融合算子只支持半精度输入，diffusers 调用前主动降 bf16 → **方差在 bf16 8 位尾数下计算**（`pow(2).mean(-1)` 平方累加丢精度） |
| **放大机制** | 单次误差 0.4% × 60 层 × 每层 6 个 norm（≈360 次）→ 残差流逐层累积 → attention softmax 指数放大 → 深层门控 scale 小、rel 分母小 → 深层 rel 74% |
| **N 卡无此问题** | N 卡走 diffusers 官方 else 分支：fp32 算方差 + rsqrt，单次误差 ~1e-6 |
| **残留差异** | 修复后仍有 ~6%：torch 版本（GPU 2.5.1 vs NPU 2.6.0）、matmul/attention kernel 累加顺序等次要来源 |

## 三、解决方案

### 核心思路

绕开 `npu_rms_norm`，让 NPU 也走 N 卡的纯 PyTorch fp32 实现（fp32 算方差）。

### 实现（一行 patch）

```python
import diffusers.models.normalization as N

# 让 diffusers 的 RMSNorm 走官方 else 分支（纯 PyTorch fp32 实现）
N.is_torch_npu_available = lambda: False
```

- 已验证与手写复制实现的方案**逐位等价**（final rel=0、1692 层全部干净）
- 不手写实现 → diffusers 升级不漂移
- 影响面仅限 `diffusers.models.normalization` 模块（text_encoder 的 transformers RMSNorm、VAE 的 GroupNorm 不受影响）

### 效果

- 深层偏差 74% → 6%
- FINAL noise_pred rel 2.96% → 2.41%、cos 0.99998
- 实际出图与 N 卡结果对齐（批量对比验证中）

### 落地

- `forward_probe.py` 新增 `--force-rmsnorm-fp32` 参数，probe 与 `--infer` 模式均已接入
- 批量推理脚本 `run_infer.sh` 加同一参数，输出 `batch_outputs_rmsnorm_fp32/`
- 代价：norm 环节非融合、略慢，但占比小、整体无感

## 四、结论

910B 部署 Qwen-Image 系模型（乃至所有在 diffusers NPU 分支下使用 RMSNorm 的模型）时，`npu_rms_norm` 的 bf16 方差计算是两机数值差异的主要来源。通过一行 patch 强制走纯 PyTorch fp32 实现，可将深层误差从 74% 压到 6%，使 NPU 推理精度对齐 N 卡。
