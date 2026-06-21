# flash_api.cpp 解析

## 1. 文件定位

`flash_api.cpp` 是 `flash_attn_npu_2` 这个 PyTorch C++ 扩展模块的 Host 侧入口文件。

它本身不是主要的 NPU 计算 kernel，而是负责把 Python 层传入的 `torch.Tensor` 参数转换成底层 AscendC kernel 可以直接使用的参数：

```text
Python API
  │
  │ flash_attn_npu_2.fwd / bwd / fwd_kvcache / varlen_fwd / varlen_bwd
  ▼
flash_api.cpp
  │
  ├── 参数校验
  ├── shape 解析
  ├── tiling 构造
  ├── workspace 分配
  ├── mask / seqlens / lse / 输出张量准备
  └── kernel launch
        │
        ├── 前向: SplitFuse::FAInfer
        └── 反向: FAG::FAG
```

`setup.py` 中将该文件编译为扩展模块：

```cpp
name="flash_attn_npu_2",
sources=source_files
```

文件末尾通过 `PYBIND11_MODULE(flash_attn_npu_2, m)` 注册 Python 可调用接口。

---

## 2. 和 Python 层的关系

Python 层文件：

```text
flash_attn_npu/flash_attn_interface.py
```

会调用本文件注册出的 C++ 函数：

| Python 调用 | C++ 注册名 | C++ 函数 | 作用 |
|---|---|---|---|
| `flash_attn_npu_2.fwd(...)` | `fwd` | `mha_fwd` | 标准 BSND 前向 |
| `flash_attn_npu_2.bwd(...)` | `bwd` | `mha_bwd` | 标准 BSND 反向 |
| `flash_attn_npu_2.fwd_kvcache(...)` | `fwd_kvcache` | `mha_fwd_kvcache` | KV-cache 推理前向 |
| `flash_attn_npu_2.varlen_fwd(...)` | `varlen_fwd` | `mha_varlen_fwd` | 变长 TND 前向 |
| `flash_attn_npu_2.varlen_bwd(...)` | `varlen_bwd` | `mha_varlen_bwd` | 变长 TND 反向 |

整体关系：

```text
flash_attn_interface.py
  │
  ├── _flash_attn_forward
  │      └── flash_attn_npu_2.fwd
  │              └── mha_fwd
  │
  ├── _flash_attn_varlen_forward
  │      └── flash_attn_npu_2.varlen_fwd
  │              └── mha_varlen_fwd
  │
  ├── _flash_attn_backward
  │      └── flash_attn_npu_2.bwd
  │              └── mha_bwd
  │
  ├── _flash_attn_varlen_backward
  │      └── flash_attn_npu_2.varlen_bwd
  │              └── mha_varlen_bwd
  │
  └── flash_attn_with_kvcache
         └── flash_attn_npu_2.fwd_kvcache
                 └── mha_fwd_kvcache
```

---

## 3. 关键 include

```cpp
#include "mha_fwd_kvcache.cpp"
#include "tilingdata.h"
#include "mha_varlen_bwd.cpp"
#include "fag_tiling.cpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "runtime/rt_ffts.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
```

含义：

| 文件 | 作用 |
|---|---|
| `mha_fwd_kvcache.cpp` | 前向 `SplitFuse::FAInfer` kernel 主要实现 |
| `tilingdata.h` | 前向 `FAInferTilingData` tiling 结构体 |
| `mha_varlen_bwd.cpp` | FAG 反向 kernel 入口和阶段调度 |
| `fag_tiling.cpp` | FAG 反向 tiling 生成逻辑 |
| `NPUStream.h` | 获取当前 NPU stream |
| `rt_ffts.h` | 获取 FFTS 调度控制地址 |
| `kernel_common.hpp` | 前向通用常量、mask 类型、layout 类型 |
| `kernel_operator.h` | AscendC kernel 编程基础接口 |

---

## 4. 两个前向 tiling 辅助函数

### 4.1 `GetQNBlockTile`

```cpp
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
```

用于计算前向 kernel 在 head/group 维度上每个 tile 同时处理多少个 query head。

关键输入：

```text
groupSize = num_heads / num_heads_k
```

这正是 GQA/MQA 的分组数。

逻辑：

```text
qNBlockTile = floor(Q_TILE_CEIL / qSeqlen)
qNBlockTile 按 N_SPLIT_HELPER 对齐
qNBlockTile 不超过 groupSize
qNBlockTile 至少为 1
```

也就是说，当 query 序列较短时，一个 tile 可以塞入更多 query head；当 query 序列较长时，至少保证处理 1 个 query head。

### 4.2 `GetQSBlockTile`

```cpp
uint32_t GetQSBlockTile(int64_t kvSeqlen)
```

当前固定返回：

```text
Q_TILE_CEIL = 128
```

表示前向 kernel 在 query 序列维度每个 tile 固定处理 128 行 query。

---

## 5. 前向路径

### 5.1 `mha_fwd_kvcache`

这是 KV-cache 推理前向入口。

典型场景：

```text
q      : 当前 step 的 query
kcache : 历史 key cache
vcache : 历史 value cache
k_/v_  : 可选的新 token key/value
```

核心流程：

```text
mha_fwd_kvcache
  │
  ├── 获取当前 NPU stream
  ├── 检查 dtype: fp16 / bf16
  ├── 检查 q/kcache/vcache 最后一维连续
  ├── 解析 paged_KV、seqlens_k、block_table
  ├── 拦截当前不支持的 alibi / rotary / leftpad / window
  ├── 填充 FAInferTilingData
  ├── 分配 workspace 和 softmaxlse
  ├── 逐 batch 计算 totalTaskNum
  ├── 如果 causal，生成 2048 x 2048 上三角 mask
  ├── 拷贝 tiling 到 NPU
  ├── 获取 FFTS 地址
  └── 按 dtype × paged_KV × causal 选择 SplitFuse::FAInfer 模板
```

KV-cache 前向支持 paged KV，因此 kernel 模板参数中有：

```cpp
PAGED_CACHE_FLAG = true / false
```

并通过 `blockTableDevice` 告诉 kernel 每个 batch 的 KV cache page 映射。

### 5.2 `mha_fwd`

这是标准 BSND 前向入口。

输入布局：

```text
q: batch_size x seqlen_q x num_heads   x head_size
k: batch_size x seqlen_k x num_heads_k x head_size
v: batch_size x seqlen_k x num_heads_k x head_size
```

当前限制：

```text
dropout      不支持，要求 p_dropout == 0
alibi_slopes 不支持
softcap      不支持，要求 softcap == 0
return_softmax 不支持
window_size_left/right 不支持
```

核心流程：

```text
mha_fwd
  │
  ├── 输入校验
  ├── 创建 out / p / rng_state
  ├── causal 时创建 mask
  ├── 创建 softmaxlse
  ├── 获取 FFTS 地址
  ├── 计算前向 workspace 大小
  ├── 填充 FAInferTilingData
  ├── 计算 totalTaskNum
  ├── 构造 q/k/v/out/seqlen/tiling/workspace 指针
  └── 启动 SplitFuse::FAInfer<..., inputLayout::BSND>
```

### 5.3 `mha_varlen_fwd`

这是变长 TND 前向入口。

输入布局：

```text
q: total_q x num_heads   x head_size
k: total_k x num_heads_k x head_size
v: total_k x num_heads_k x head_size
```

每个 batch 的真实长度由：

```text
cu_seqlens_q: shape = b + 1
cu_seqlens_k: shape = b + 1
```

表示。

例如：

```text
cu_seqlens_q = [0, 3, 5, 9]
```

表示 batch=3：

```text
样本0: q token [0, 3)
样本1: q token [3, 5)
样本2: q token [5, 9)
```

该路径启动：

```cpp
SplitFuse::FAInfer<..., FaiKenel::inputLayout::TND, ...>
```

即 kernel 侧按 TND 变长布局解释输入。

---

## 6. 前向 `FAInferTilingData`

前向路径使用 `tilingdata.h` 中的：

```cpp
struct FAInferTilingData
```

Host 侧写入字段，Device 侧 kernel 读取字段。

主要字段：

| 字段 | 含义 |
|---|---|
| `numHeads` | Q head 数 |
| `kvHeads` | K/V head 数 |
| `embeddingSize` | Q/K head_dim |
| `embeddingSizeV` | V head_dim |
| `batch` | batch size |
| `maskType` | 是否 causal mask |
| `numBlocks` | paged KV cache 的 block 数 |
| `blockSize` | paged KV cache 每块 token 数 |
| `maxNumBlocksPerBatch` | 每个 batch 最大 block 数 |
| `firstBatchTaskNum` | 第一个 batch 的任务数 |
| `totalTaskNum` | 总任务数 |
| `scaleValue` | softmax scale |
| `softcapValue` | softcap 参数 |
| `mm1OutSize` | QK^T 阶段 workspace 大小 |
| `smOnlineOutSize` | online softmax workspace 大小 |
| `mm2OutSize` | PV 阶段 workspace 大小 |
| `UpdateSize` | 更新 LSE / max 等状态的 workspace 大小 |
| `workSpaceSize` | 总 workspace 大小 |

前向 workspace 大致分区：

```text
workspace base
  │
  ├── mm1OutSize
  │      QK^T / score 中间结果
  │
  ├── smOnlineOutSize
  │      online softmax 中间状态
  │
  ├── mm2OutSize
  │      P * V 中间结果
  │
  └── UpdateSize
         LSE / max / 更新缓冲
```

---

## 7. 反向路径

反向路径分两种：

| 函数 | 输入布局 | kernel layout |
|---|---|---|
| `mha_bwd` | BSND | `InputLayout::BSND` |
| `mha_varlen_bwd` | TND | `InputLayout::TND` |

它们都启动同一个 FAG backward kernel：

```cpp
FAG::FAG<..., MaskType::..., InputLayout::...>
```

### 7.1 `mha_varlen_bwd`

输入：

```text
dout: total_q x num_heads   x head_size
q   : total_q x num_heads   x head_size
k   : total_k x num_heads_k x head_size
v   : total_k x num_heads_k x head_size
out : total_q x num_heads   x head_size
```

核心流程：

```text
mha_varlen_bwd
  │
  ├── 获取 NPU stream 和 AIC blockDim
  ├── 准备 dq/dk/dv 输出张量
  ├── 解析 nheads / nheads_k / headdim
  ├── 构造 FAGTiling::FAGInfo
  ├── 调用 FAGTiling::GetFATilingParam
  ├── tiling_cpu_tensor 拷贝到 tiling_gpu_tensor
  ├── 分配 FAG workspace
  ├── causal 时创建 mask
  ├── cu_seqlens_q/k 拷贝到 NPU
  ├── 构造 q/k/v/out/dout/dq/dk/dv/workspace/tiling 指针
  └── 启动 FAG::FAG<..., InputLayout::TND>
```

### 7.2 `mha_bwd`

标准 BSND 反向入口。

输入：

```text
q   : batch_size x seqlen_q x num_heads   x head_size
k   : batch_size x seqlen_k x num_heads_k x head_size
v   : batch_size x seqlen_k x num_heads_k x head_size
out : batch_size x seqlen_q x num_heads   x head_size
dout: batch_size x seqlen_q x num_heads   x head_size
```

Host 侧会把 batch 和 seqlen 展平成 FAG tiling 中的 total token 数：

```cpp
fagInfo.queryShape_0 = qsizes[0] * qsizes[1];
fagInfo.keyShape_0 = ksizes[0] * ksizes[1];
```

但 kernel 模板使用：

```cpp
InputLayout::BSND
```

因此 Device 侧仍按 BSND 布局解释指针。

---

## 8. FAG 反向 tiling 和 workspace

反向路径不使用 `FAInferTilingData`，而是使用 `fag_tiling.cpp` 中的：

```cpp
FAGTiling::GetFATilingParam
```

生成一块 `int64_t` 为主体的 tiling buffer。

Host 侧填充：

```cpp
FAGTiling::FAGInfo fagInfo;
fagInfo.seqQShapeSize = ...;
fagInfo.queryShape_0 = ...;
fagInfo.keyShape_0 = ...;
fagInfo.queryShape_1 = nheads;
fagInfo.keyShape_1 = nheads_k;
fagInfo.queryShape_2 = headdim;
fagInfo.scaleValue = 1.0 / sqrt(headdim);
FAGTiling::GetFATilingParam(fagInfo, blockDim, ...);
```

`GetFATilingParam` 负责写入：

```text
shape / nheads_k / g / headdim
softmax tiling
softmaxGrad tiling
AIV coreNum
workspace byte offsets
```

反向 workspace 在 FAG kernel 内部用于：

```text
workspace base
  │
  ├── dq fp32 workspace
  ├── dk fp32 workspace
  ├── dv fp32 workspace
  ├── sfmg workspace
  ├── mm1 workspace = dOut * V^T
  ├── mm2 workspace = Q * K^T
  ├── p workspace   = softmax P
  └── ds workspace  = dS
```

---

## 9. Kernel 模板分发

### 9.1 前向模板分发

前向根据三类条件组合选择模板：

```text
dtype       : fp16 / bf16
paged_KV    : true / false
is_causal   : true / false
inputLayout : BSND / TND
```

例如标准 BSND fp16 causal：

```cpp
SplitFuse::FAInfer<half, half, float,
                   false,
                   FaiKenel::MaskType::MASK_CAUSAL,
                   FaiKenel::inputLayout::BSND,
                   Catlass::Epilogue::LseModeT::OUT_ONLY>
```

变长 TND bf16 non-causal：

```cpp
SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float,
                   false,
                   FaiKenel::MaskType::NO_MASK,
                   FaiKenel::inputLayout::TND,
                   Catlass::Epilogue::LseModeT::OUT_ONLY>
```

### 9.2 反向模板分发

反向根据三类条件组合选择模板：

```text
dtype       : fp16 / bf16
is_causal   : true / false
inputLayout : BSND / TND
```

例如变长 TND fp16 causal：

```cpp
FAG::FAG<half, MaskType::MASK_CAUSAL, InputLayout::TND>
```

标准 BSND bf16 non-causal：

```cpp
FAG::FAG<bfloat16_t, MaskType::NO_MASK, InputLayout::BSND>
```

这种模板分发方式的好处是：

```text
Host 侧用 if 选择一次
Device 侧模板参数固定
kernel 内少走运行时分支
```

---

## 10. 简单例子 1：标准 BSND 前向

假设：

```text
q shape = [2, 128, 8, 64]
k shape = [2, 128, 8, 64]
v shape = [2, 128, 8, 64]
causal = true
dtype = fp16
```

Host 侧主要动作：

```text
batch_size = 2
seqlen_q   = 128
seqlen_k   = 128
num_heads  = 8
num_heads_k = 8
head_size  = 64
```

填充 `FAInferTilingData`：

```text
batch = 2
numHeads = 8
kvHeads = 8
embeddingSize = 64
maskType = 1
scaleValue = softmax_scale
maxQSeqlen = 128
```

任务数概念图：

```text
batch 0:
  qN blocks = num_heads_k * ceil(groupSize / qNBlockTile)
  qS blocks = ceil(128 / 128) = 1

batch 1:
  同上

最终 totalTaskNum = batch0 task + batch1 task
```

最后启动：

```text
SplitFuse::FAInfer<half, ..., MASK_CAUSAL, BSND>
```

---

## 11. 简单例子 2：变长 TND 反向

假设：

```text
batch = 3
cu_seqlens_q = [0, 3, 5, 9]
total_q = 9
num_heads = 4
num_heads_k = 2
headdim = 64
```

则：

```text
g = num_heads / num_heads_k = 2
```

`mha_varlen_bwd` 中填充：

```cpp
fagInfo.seqQShapeSize = 3;
fagInfo.queryShape_0 = 9;
fagInfo.keyShape_0 = 9;
fagInfo.queryShape_1 = 4;
fagInfo.keyShape_1 = 2;
fagInfo.queryShape_2 = 64;
fagInfo.scaleValue = 1.0 / sqrt(64);
```

整体流程：

```text
Python varlen backward
  │
  ▼
mha_varlen_bwd
  │
  ├── 生成 FAG tiling
  ├── 分配 workspace
  ├── 准备 cu_seqlens_q/k
  └── 启动 FAG::FAG<..., InputLayout::TND>
        │
        ├── FAGPre 清零 dq/dk/dv fp32 workspace
        ├── FAGSfmg 计算 sum(dout * out)
        ├── Cube1 计算 QK^T 和 dOutV^T
        ├── Vector epilogue 计算 P / dS
        ├── Cube2 / Cube3 计算 dq/dk/dv
        └── FAGPost cast/scale 输出
```

图示：

```text
q/k/v/out/dout + cu_seqlens
          │
          ▼
   flash_api.cpp
          │
          ├── FAGInfo
          ├── tiling_gpu_tensor
          ├── workspace_tensor
          └── mask / ptrs
          │
          ▼
       FAG::FAG
          │
          ▼
      dq / dk / dv
```

---

## 12. 注意点

### 12.1 前向和反向使用不同 tiling 结构

```text
前向: FAInferTilingData
反向: FAGTiling::GetFATilingParam 生成的 int64_t tiling buffer
```

这是因为前向和反向 kernel 的阶段、workspace 分区、softmax 重算需求不同。

### 12.2 `softmax_lse` 形状不完全相同

标准前向中：

```text
softmaxlse shape = [batch_size, seqlen_q, num_heads]
```

变长前向中：

```text
softmaxlse shape = [T, num_heads]
```

反向路径直接接收 Python 层传回来的 `softmax_lse`，并作为 `softMaxLseDevice` 传给 FAG kernel。

### 12.3 Causal mask 固定生成 2048 x 2048

多个路径中 causal mask 都通过：

```cpp
at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, ...);
mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
```

生成上三角 mask，然后拷贝到 NPU。

### 12.4 `rtGetC2cCtrlAddr`

多个 kernel launch 前都会调用：

```cpp
rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
```

用于获取 Ascend FFTS 多核调度需要的控制地址，随后作为 kernel 第一个参数传入。

### 12.5 `ENABLE_ASCENDC_DUMP`

反向路径支持 dump 调试分支：

```cpp
#if defined(ENABLE_ASCENDC_DUMP)
```

开启后会分配 dump buffer，并在 kernel 同步后打印 device 工作区内容，便于调试 FAG kernel 内部状态。

---

## 13. 总结

`flash_api.cpp` 是连接 Python 层和 AscendC kernel 层的桥梁。

它的核心价值不在于实现 FlashAttention 数学公式，而在于完成以下 Host 侧编排：

1. 接收 Python/PyTorch 参数；
2. 校验 dtype、layout、stride 和当前支持范围；
3. 解析 BSND / TND / KV-cache 的 shape；
4. 生成前向 `FAInferTilingData` 或反向 FAG tiling；
5. 分配 workspace、输出、LSE、mask 等张量；
6. 获取 NPU stream 和 FFTS 调度地址；
7. 根据 dtype、mask、layout、paged KV 选择具体模板；
8. 启动 `SplitFuse::FAInfer` 或 `FAG::FAG` kernel；
9. 通过 pybind11 暴露给 Python。

一句话概括：

```text
flash_api.cpp = Python FlashAttention API 到 Ascend NPU kernel 的 Host 侧调度与参数装配层。
```
