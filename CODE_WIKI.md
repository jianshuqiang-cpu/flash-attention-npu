# FlashAttention for Ascend NPU — Code Wiki

> **项目版本**: 0.1.1  
> **许可证**: BSD 3-Clause  
> **作者**: Minghua Shen (shenmh6@mail.sysu.edu.cn)  
> **仓库**: https://github.com/MinghuasLab/flash-attention-npu

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [模块职责详解](#4-模块职责详解)
   - 4.1 [Python 接口层 (flash_attn_npu)](#41-python-接口层-flash_attn_npu)
   - 4.2 [Python 接口层 (flash_attn_npu_v3)](#42-python-接口层-flash_attn_npu_v3)
   - 4.3 [C++/AscendC 内核层 (csrc/flash_attn_npu)](#43-cascendc-内核层-csrcflash_attn_npu)
   - 4.4 [C++/AscendC 内核层 (csrc/flash_attn_npu_v3)](#44-cascendc-内核层-csrcflash_attn_npu_v3)
   - 4.5 [CATLASS 框架依赖 (csrc/catlass)](#45-catlass-框架依赖-csrccatlass)
5. [关键类与函数说明](#5-关键类与函数说明)
   - 5.1 [v2 Python API](#51-v2-python-api)
   - 5.2 [v3 Python API](#52-v3-python-api)
   - 5.3 [C++ 内核入口函数](#53-c-内核入口函数)
   - 5.4 [Tiling 数据结构](#54-tiling-数据结构)
   - 5.5 [AscendC Kernel 模板](#55-ascendc-kernel-模板)
6. [数据流与计算流程](#6-数据流与计算流程)
7. [依赖关系](#7-依赖关系)
8. [构建与安装](#8-构建与安装)
9. [测试](#9-测试)
10. [功能特性矩阵](#10-功能特性矩阵)
11. [架构设计要点](#11-架构设计要点)

---

## 1. 项目概述

本项目是 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的 Ascend NPU 适配实现。FlashAttention 通过 tiling 和内存感知算法显著提升大语言模型的训练与推理效率。原始实现面向 NVIDIA GPU 架构，本项目基于华为 [CANN/CATLASS](https://gitcode.com/cann/catlass) 框架及其示例代码，在 Ascend NPU 上实现了 API 兼容的 FlashAttention 算法，降低模型迁移的适配复杂度。

**核心目标**：
- 提供与 Dao-AILab/flash-attention API 一致的接口
- 基于 Ascend NPU 硬件特性进行算子优化
- 支持 FlashAttention v2 和 v3 两代算法

**目标硬件**：Ascend 910B / 910C NPU

---

## 2. 整体架构

项目采用 **Python 接口层 + C++/AscendC 内核层** 的分层架构：

```
┌─────────────────────────────────────────────────────────┐
│                    用户代码 (PyTorch)                      │
├─────────────────────────────────────────────────────────┤
│              Python 接口层 (flash_attn_npu)              │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   v2 接口模块          │  │   v3 接口模块              │ │
│  │ flash_attn_npu/       │  │ flash_attn_npu_v3/       │ │
│  │  - flash_attn_func   │  │  - flash_attn_func       │ │
│  │  - flash_attn_with_  │  │  - flash_attn_with_     │ │
│  │    kvcache           │  │    kvcache               │ │
│  │  - flash_attn_varlen │  │  - flash_attn_varlen_   │ │
│  │    _func             │  │    func                  │ │
│  └──────────┬───────────┘  └──────────┬───────────────┘ │
│             │ pybind11                │ pybind11         │
├─────────────┼────────────────────────┼─────────────────┤
│             ▼                        ▼                   │
│           C++/AscendC 内核层                               │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │ csrc/flash_attn_npu/ │  │ csrc/flash_attn_npu_v3/ │ │
│  │  - flash_api.cpp     │  │  - flash_api.cpp        │ │
│  │  - mha_fwd_kvcache   │  │  - mha_fwd_kvcache      │ │
│  │  - mha_varlen_bwd    │  │  - tilingdata.h         │ │
│  │  - fag_tiling.cpp    │  │                          │ │
│  │  - online_softmax    │  │                          │ │
│  │  - qk_matmul/pv_mat  │  │                          │ │
│  └──────────┬───────────┘  └──────────┬───────────────┘ │
│             │                         │                  │
│             ▼                         ▼                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           CATLASS 框架 (csrc/catlass submodule)       │ │
│  │  - AscendC 算子基础设施                                 │ │
│  │  - 矩阵乘法 / Epilogue / Tiling 抽象                   │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│              华为 CANN 软件栈 (>= 8.5.0)                   │
│  - AscendC 编译器 (bisheng)                               │
│  - ACL 运行时 / Tiling API / Platform API                │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
flash-attention-npu/
├── flash_attn_npu/                  # v2 Python 包
│   ├── __init__.py                  #   包初始化，导出公共 API
│   └── flash_attn_interface.py     #   v2 核心接口实现
├── flash_attn_npu_v3/              # v3 Python 包
│   ├── __init__.py                  #   包初始化，导出公共 API
│   └── flash_attn_interface.py     #   v3 核心接口实现
├── csrc/                            # C++/AscendC 内核源码
│   ├── flash_attn_npu/              #   v2 内核实现
│   │   ├── fag_common/             #     公共头文件 (地址抽象等)
│   │   ├── flash_api.cpp           #     ★ pybind11 入口，注册 fwd/bwd/fwd_kvcache/varlen_fwd/varlen_bwd
│   │   ├── mha_fwd_kvcache.cpp     #     KV-cache 前向内核实现
│   │   ├── mha_varlen_bwd.cpp      #     变长序列反向内核实现
│   │   ├── fag_tiling.cpp          #     FAG (Flash Attention Graph) Tiling 参数计算
│   │   ├── softmax_tiling.cpp      #     Softmax Tiling 参数计算
│   │   ├── tilingdata.h            #     ★ FAInferTilingData 结构体定义
│   │   ├── fa_block.h              #     FA Block 定义
│   │   ├── fag_block.h            #     FAG Block 定义
│   │   ├── fag_sfmg.h             #     Softmax 相关 AscendC 类型别名
│   │   ├── kernel_common.hpp      #     ★ 内核常量与工具函数
│   │   ├── online_softmax.hpp     #     Online Softmax 实现 (高精度)
│   │   ├── online_softmax_low_prec.hpp # Online Softmax 实现 (低精度)
│   │   ├── qk_matmul.hpp          #     QK 矩阵乘法内核
│   │   ├── pv_matmul.hpp          #     PV 矩阵乘法内核
│   │   ├── rescale_o.hpp          #     输出重缩放 (高精度)
│   │   ├── rescale_o_low_prec.hpp #     输出重缩放 (低精度)
│   │   ├── fag_epilogue_*.hpp    #     Epilogue 阶段 (pre/op/post/sfmg)
│   │   ├── fag_mmad_cube*.hpp    #     MMAD Cube 矩阵乘加速
│   │   ├── init_outputs.hpp      #     输出初始化
│   │   ├── cube_addr.h / vector_addr.h # Cube/Vector 地址抽象
│   │   └── common_header.h       #     公共头文件
│   ├── flash_attn_npu_v3/          #   v3 内核实现
│   │   ├── flash_api.cpp          #     ★ pybind11 入口，注册 fwd
│   │   ├── mha_fwd_kvcache.cpp    #     KV-cache 前向内核
│   │   ├── tilingdata.h           #     ★ Tiling 数据结构
│   │   ├── kernel_common.hpp      #     内核常量
│   │   ├── online_softmax.hpp     #     Online Softmax
│   │   ├── qk_matmul.hpp / pv_matmul.hpp
│   │   └── rescale_o.hpp / rescale_o_low_prec.hpp
│   └── catlass/                    #   ★ Git 子模块 - CATLASS 框架
├── csrc_AscendC950/                # AscendC 950 变体 (chunk prefill)
│   └── flash_attention_chunk_prefill/
│       ├── CMakeLists.txt
│       ├── fai.cpp / fai_kernel.cpp / fai_tiling.cpp
│       └── fai_tilingdata.h
├── tests/                           # 测试文件
│   ├── test_flash_attn_npu.py      #   v2 前向/反向/KV-cache/Varlen 测试
│   ├── test_flash_attn_npu_bwd.py  #   v2 反向传播测试
│   ├── test_flash_attn_npu_v3.py   #   v3 前向/KV-cache 测试
│   └── precision_compare.py        #   精度对比工具
├── setup.py                         # ★ 构建与安装脚本
├── Makefile                         # 辅助构建目标 (clean/create_dist/upload)
├── MANIFEST.in                      # sdist 包含规则
├── README.md / README.zh.md         # 项目文档
├── LICENSE                          # BSD 3-Clause 许可证
├── AUTHORS                          # 作者信息
└── .gitmodules                      # Git 子模块配置
```

---

## 4. 模块职责详解

### 4.1 Python 接口层 (flash_attn_npu)

**路径**: `flash_attn_npu/`

v2 版本的 Python 接口层，提供与 Dao-AILab/flash-attention 兼容的 API。

**核心职责**：
- 封装 C++ 扩展模块 `flash_attn_npu_2` 的调用
- 实现 `torch.autograd.Function` 子类以支持自动微分
- 处理输入张量的 contiguity 和 head dimension 对齐（8 的倍数）
- 注册 `torch.library.custom_op` 以支持 `torch.compile()`

**导出的公共 API**（定义于 `__init__.py`）：
- `flash_attn_func` — 标准 FlashAttention 前向+反向
- `flash_attn_kvpacked_func` — KV 打包模式
- `flash_attn_qkvpacked_func` — QKV 打包模式
- `flash_attn_varlen_func` — 变长序列模式
- `flash_attn_varlen_kvpacked_func` — 变长序列 KV 打包模式
- `flash_attn_varlen_qkvpacked_func` — 变长序列 QKV 打包模式
- `flash_attn_with_kvcache` — KV-cache 推理模式（不支持反向）

### 4.2 Python 接口层 (flash_attn_npu_v3)

**路径**: `flash_attn_npu_v3/`

v3 版本的 Python 接口层，相比 v2 提供更多特性（如 Paged KV Cache 的 `page_table`、FP8 反量化、attention chunk、softcap 等）。

**核心职责**：
- 封装 C++ 扩展模块 `flash_attn_npu_3` 的调用
- 实现 `torch.autograd.Function` 子类
- 支持 `num_splits` 并行拆分和 `pack_gqa` GQA 优化
- 提供 `get_scheduler_metadata` 辅助函数

**导出的公共 API**（定义于 `__init__.py`）：
- `flash_attn_with_kvcache` — v3 KV-cache 推理模式

**内部函数**（定义于 `flash_attn_interface.py`）：
- `flash_attn_func` — 标准前向+反向
- `flash_attn_qkvpacked_func` — QKV 打包模式
- `flash_attn_varlen_func` — 变长序列模式
- `flash_attn_combine` — 分块结果合并

### 4.3 C++/AscendC 内核层 (csrc/flash_attn_npu)

**路径**: `csrc/flash_attn_npu/`

v2 版本的 NPU 内核实现，基于 CATLASS 框架的 `SplitFuse::FAInfer` 和 `FAG::FAG` 模板。

**核心文件**：

| 文件 | 职责 |
|------|------|
| `flash_api.cpp` | pybind11 模块入口，注册 `fwd`/`bwd`/`fwd_kvcache`/`varlen_fwd`/`varlen_bwd` 五个函数 |
| `mha_fwd_kvcache.cpp` | KV-cache 前向内核的 AscendC 实现 |
| `mha_varlen_bwd.cpp` | 变长序列反向传播的 FAG 内核实现 |
| `fag_tiling.cpp` | FAG 模式的 Tiling 参数计算 |
| `softmax_tiling.cpp` | Softmax Tiling 参数计算 |
| `tilingdata.h` | `FAInferTilingData` 结构体定义 |
| `kernel_common.hpp` | 内核常量（Q_TILE_CEIL=128, MAX_KV_STACK_LEN=512 等）、枚举类型、工具函数 |
| `online_softmax.hpp` | Online Softmax 高精度实现 |
| `qk_matmul.hpp` / `pv_matmul.hpp` | QK/PV 矩阵乘法内核 |
| `rescale_o.hpp` | 输出重缩放 |

**内核模板实例化策略**：

`flash_api.cpp` 中根据数据类型、是否 Paged KV、是否 Causal、是否 Varlen 四个维度组合实例化不同的内核模板：

```
SplitFuse::FAInfer<数据类型, 数据类型, float, 是否PagedKV, MaskType, InputLayout, LseModeT>
```

- 数据类型：`bfloat16_t` / `half`
- 是否 Paged KV：`true` / `false`
- MaskType：`MASK_CAUSAL` / `NO_MASK`
- InputLayout：`BSND`（标准批处理）/ `TND`（变长序列）
- LseModeT：`OUT_ONLY`

### 4.4 C++/AscendC 内核层 (csrc/flash_attn_npu_v3)

**路径**: `csrc/flash_attn_npu_v3/`

v3 版本的 NPU 内核实现，结构与 v2 类似但更精简，当前仅注册了 `fwd` 函数。

**与 v2 的主要差异**：
- 统一了标准前向和 KV-cache 前向为单一 `mha_fwd` 函数
- 支持 `is_varlen_q` / `is_varlen_kv` 判断变长模式
- 支持 `page_table`（Paged KV Cache）
- 尚未实现反向传播

### 4.5 CATLASS 框架依赖 (csrc/catlass)

**路径**: `csrc/catlass`（Git 子模块）

CATLASS (CANN Template Library for Accelerated System Solutions) 是华为 CANN 生态下的矩阵运算框架，类似于 NVIDIA 的 CUTLASS。本项目通过 Git 子模块引入，版本为 `v1.3.1-notla`。

**提供的核心能力**：
- AscendC 算子基础设施（矩阵乘、Epilogue、Tiling 抽象）
- `SplitFuse::FAInfer` 模板 — FlashAttention 前向推理内核
- `FAG::FAG` 模板 — FlashAttention Graph（支持反向传播）
- `Catlass::Epilogue::LseModeT` — LogSumExp 输出模式
- `FaiKenel::MaskType` — 注意力掩码类型枚举
- `FaiKenel::inputLayout` — 输入布局枚举（BSND / TND）

---

## 5. 关键类与函数说明

### 5.1 v2 Python API

#### `flash_attn_func(q, k, v, ...)`

标准 FlashAttention 前向+反向接口。

```python
def flash_attn_func(
    q, k, v,
    dropout_p=0.0, softmax_scale=None,
    causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None,
    deterministic=False, return_attn_probs=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

**输入形状**：
- `q`: `(batch_size, seqlen, nheads, headdim)`
- `k`: `(batch_size, seqlen, nheads_k, headdim)`
- `v`: `(batch_size, seqlen, nheads_k, headdim)`

**输出**：
- `out`: `(batch_size, seqlen, nheads, headdim)`
- 可选 `softmax_lse`, `S_dmask`

**内部实现**：通过 `FlashAttnFunc(torch.autograd.Function)` 子类实现，前向调用 `_flash_attn_forward` custom op，反向调用 `_flash_attn_backward` custom op。

#### `flash_attn_with_kvcache(q, k_cache, v_cache, ...)`

KV-cache 推理接口，用于增量解码。**不支持反向传播**。

```python
def flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=None, v=None,
    rotary_cos=None, rotary_sin=None,
    cache_seqlens=None, cache_batch_idx=None,
    cache_leftpad=None, block_table=None,
    softmax_scale=None, causal=False,
    window_size=(-1, -1), softcap=0.0,
    rotary_interleaved=True, alibi_slopes=None,
    num_splits=0, return_softmax_lse=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**关键特性**：
- 若提供 `k`/`v`，`k_cache`/`v_cache` 将被**原地更新**
- 支持 Paged KV Cache（通过 `block_table`）
- 支持 MQA/GQA（`nheads_k < nheads`）

#### `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`

变长序列模式，适用于同一 batch 中序列长度不同的场景。

**输入形状**：
- `q`: `(total_q, nheads, headdim)` — total_q 为所有序列 token 总数
- `cu_seqlens_q`: `(batch_size + 1,)` — 累积序列长度

#### `FlashAttnQKVPackedFunc` / `FlashAttnKVPackedFunc`

打包模式的 autograd.Function 子类，分别处理 QKV 打包和 KV 打包的输入格式，减少反向传播中的显式梯度拼接开销。

### 5.2 v3 Python API

#### `flash_attn_with_kvcache(q, k_cache, v_cache, ...)`

v3 版 KV-cache 接口，相比 v2 增加了以下参数：

| 新增参数 | 说明 |
|---------|------|
| `qv` | 可选的 QV 张量，用于不同 head_dim_v |
| `page_table` | Paged KV Cache 页表（替代 v2 的 `block_table`） |
| `cu_seqlens_q` / `cu_seqlens_k_new` | Ragged 模式累积序列长度 |
| `max_seqlen_q` | 最大查询序列长度 |
| `rotary_seqlens` | RoPE 位置序列长度 |
| `q_descale` / `k_descale` / `v_descale` | FP8 反量化缩放因子 |
| `attention_chunk` | Attention 分块大小 |
| `softcap` | Softcapping 阈值 |
| `scheduler_metadata` | 调度器元数据 |
| `num_splits` | KV 序列拆分数 |
| `pack_gqa` | GQA 打包优化 |
| `sm_margin` | SM 余量调优参数 |

#### `get_scheduler_metadata(...)`

辅助函数，调用 `flash_attn_npu_3.get_scheduler_metadata` C++ 扩展，生成调度器所需的元数据张量。

### 5.3 C++ 内核入口函数

#### v2 入口 (`flash_api.cpp` → `PYBIND11_MODULE(flash_attn_npu_2, ...)`)

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `mha_fwd` | `(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, ...)` → `[out, softmax_lse, p, rng_state]` | 标准前向 |
| `mha_bwd` | `(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, ...)` → `[dq, dk, dv, softmax_d]` | 标准反向 |
| `mha_fwd_kvcache` | `(q, kcache, vcache, k_, v_, seqlens_k_, ...)` → `[out, softmax_lse]` | KV-cache 前向 |
| `mha_varlen_fwd` | `(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` → `[out, softmax_lse, p, rng_state]` | 变长前向 |
| `mha_varlen_bwd` | `(dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, ...)` → `[dq, dk, dv, softmax_d]` | 变长反向 |

#### v3 入口 (`flash_api.cpp` → `PYBIND11_MODULE(flash_attn_npu_3, ...)`)

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `mha_fwd` | `(q, k, v, k_new_, v_new_, q_v_, out_, cu_seqlens_q_, ...)` → `[out, softmax_lse, out_accum, softmax_lse_accum]` | 统一前向（含 KV-cache） |

### 5.4 Tiling 数据结构

#### `FAInferTilingData`（定义于 `tilingdata.h`）

Tiling 是 Ascend NPU 算子中的关键概念，用于将计算任务分解到多个 AI Core 上并行执行。

```cpp
struct FAInferTilingData {
    uint32_t numHeads;           // Q 头数
    uint32_t embeddingSize;      // Q/K 头维度
    uint32_t embeddingSizeV;     // V 头维度
    uint32_t numBlocks;          // Paged KV 块数
    uint32_t blockSize;          // Paged KV 块大小
    uint32_t maxQSeqlen;         // 最大 Q 序列长度
    uint32_t maxKvSeqlen;        // 最大 KV 序列长度
    uint32_t kvHeads;            // KV 头数
    uint32_t batch;              // 批大小
    uint32_t maxNumBlocksPerBatch; // 每序列最大块数
    uint32_t firstBatchTaskNum;  // 首批任务数
    uint32_t totalTaskNum;       // 总任务数
    uint32_t maskType;           // 掩码类型 (0=NO_MASK, 1=CAUSAL)
    uint64_t mm1OutSize;         // QK 矩阵乘输出空间大小
    uint64_t smOnlineOutSize;    // Online Softmax 输出空间大小
    uint64_t mm2OutSize;         // PV 矩阵乘输出空间大小
    uint64_t UpdateSize;         // 更新空间大小
    uint64_t workSpaceSize;      // 总工作空间大小
    float scaleValue;            // Softmax 缩放因子
    float softcapValue;          // Softcap 值
};
```

### 5.5 AscendC Kernel 模板

#### `SplitFuse::FAInfer`（前向推理内核）

```cpp
template <
    typename input_t,      // 输入数据类型 (bfloat16_t / half)
    typename weight_t,     // 权重数据类型
    typename output_t,     // 输出数据类型 (float)
    bool pagedKV,          // 是否使用 Paged KV Cache
    typename MaskType,     // 掩码类型 (MASK_CAUSAL / NO_MASK)
    typename InputLayout,  // 输入布局 (BSND / TND)
    typename LseMode       // LogSumExp 输出模式
>
void FAInfer(uint64_t fftsAddr, ...);
```

#### `FAG::FAG`（Flash Attention Graph，支持反向传播）

```cpp
template <
    typename input_t,      // 输入数据类型
    typename MaskType = MaskType::NO_MASK,  // 掩码类型
    typename InputLayout = InputLayout::BSND // 输入布局
>
void FAG(uint64_t fftsAddr, ...);
```

---

## 6. 数据流与计算流程

### 6.1 前向推理流程 (SplitFuse::FAInfer)

```
输入: Q, K, V, [Mask], [BlockTable]
          │
          ▼
    ┌─────────────┐
    │ Tiling 计算   │ ← CPU 端计算任务分配参数
    │ (FAInferTilingData)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ QK MatMul    │ ← Cube 矩阵乘 (mm1OutSize workspace)
    │ (Q × K^T)    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Online       │ ← Vector Softmax (smOnlineOutSize workspace)
    │ Softmax      │   分块计算 max → exp → sum
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ PV MatMul    │ ← Cube 矩阵乘 (mm2OutSize workspace)
    │ (Softmax×V)  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Rescale O    │ ← Vector 重缩放输出
    └──────┬──────┘
           │
           ▼
输出: Out, SoftmaxLSE
```

### 6.2 反向传播流程 (FAG::FAG)

```
输入: dOut, Q, K, V, Out, SoftmaxLSE
          │
          ▼
    ┌─────────────┐
    │ Tiling 计算   │ ← FAGTiling::GetFATilingParam
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │ FAG 内核 (Flash Attention Graph) │
    │  - dV = Softmax^T × dOut        │
    │  - Softmax_grad                  │
    │  - dK = Q^T × dSoftmax          │
    │  - dQ = dSoftmax × K            │
    └──────┬──────────────────────────┘
           │
           ▼
输出: dQ, dK, dV, Softmax_d
```

### 6.3 工作空间分配

每个 NPU AI Core 的工作空间布局：

```
Workspace = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize

其中:
  mm1OutSize      = blockDim × (128 × 512) × 4 × 3   (QK 矩阵乘)
  smOnlineOutSize = blockDim × (128 × 512) × 2 × 3   (Online Softmax)
  mm2OutSize      = blockDim × (128 × 512) × 4 × 3   (PV 矩阵乘)
  UpdateSize      = blockDim × (128 × 512) × 4 × 3   (输出更新)
```

`blockDim` 由 `PlatformAscendCManager::GetCoreNumAic()` 获取，`3` 为预取缓冲数（`PRELANCH_NUM`）。

---

## 7. 依赖关系

### 7.1 系统依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Ascend NPU 硬件 | 910B / 910C | 目标硬件 |
| CANN | >= 8.5.0 | 华为计算框架 |
| PyTorch | >= 2.1.0 | 深度学习框架 |
| torch_npu | >= 2.1.0 | PyTorch NPU 适配层 |
| Python | >= 3.9 | 运行时 |
| Linux | - | 操作系统 |

### 7.2 Python 包依赖

**运行时依赖** (`install_requires`)：
- `torch`
- `torch_npu`
- `einops`

**构建时依赖** (`setup_requires`)：
- `packaging`
- `psutil`
- `ninja`

### 7.3 C++ 依赖

| 依赖 | 来源 | 说明 |
|------|------|------|
| CATLASS | Git 子模块 (`csrc/catlass`) | AscendC 矩阵运算框架 |
| ACL | CANN 运行时 | Ascend Computing Language |
| torch_npu | PyTorch 扩展 | NPU Stream / 设备管理 |
| tiling_api | CANN 编译器 | Tiling 平台信息 API |

### 7.4 构建工具链

- **编译器**: `bisheng`（华为 AscendC 编译器）
- **编译目标**: `dav-2201`（Ascend 910B/910C 的 NPU 架构）
- **C++ 标准**: C++17
- **链接库**: `ascendcl`, `torch_npu`, `tiling_api`, `platform`

---

## 8. 构建与安装

### 8.1 环境准备

```bash
# 1. 设置 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 2. 安装 Python 依赖
pip install packaging psutil ninja
```

### 8.2 获取源码

```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive
```

### 8.3 构建安装

```bash
# 构建全部版本 (v2 + v3)
python setup.py install

# 仅构建 v2
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install

# 仅构建 v3
FLASH_ATTN_BUILD_VERSION=v3 python setup.py install
```

### 8.4 构建控制环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `FLASH_ATTENTION_FORCE_BUILD` | `FALSE` | 强制本地编译，不下载预编译 wheel |
| `FLASH_ATTENTION_SKIP_NPU_BUILD` | `FALSE` | 跳过 NPU 编译（用于 sdist） |
| `FLASH_ATTN_BUILD_VERSION` | `all` | 构建版本：`v2` / `v3` / `all` |
| `FLASH_ATTN_LOCAL_VERSION` | - | 附加本地版本号 |

### 8.5 构建流程

`setup.py` 中的 `BishengBuildExt` 自定义构建扩展：

1. 检测 CANN 安装路径（`ASCEND_TOOLKIT_HOME`）
2. 收集 Ascend C++ 头文件路径和库路径
3. 收集 PyTorch / torch_npu 头文件和库路径
4. 检测 C++11 ABI 标志
5. 调用 `bisheng` 编译器编译 `.cpp` 源码为共享库
6. 输出 `.so` 文件作为 Python 扩展模块

### 8.6 预编译 Wheel

`setup.py` 支持从 GitHub Releases 下载预编译 wheel，命名格式为：

```
flash_attn_npu-{version}+npu{ver}torch{torch_ver}cxx11abi{abi}-{python}-{python}-linux_{arch}.whl
```

---

## 9. 测试

### 9.1 运行测试

```bash
# v2 测试
pytest -q -s tests/test_flash_attn_npu.py

# v2 反向传播测试
pytest -q -s tests/test_flash_attn_npu_bwd.py

# v3 测试
pytest -q -s tests/test_flash_attn_npu_v3.py
```

### 9.2 测试覆盖

**v2 测试** (`test_flash_attn_npu.py`)：

| 测试函数 | 覆盖功能 |
|---------|---------|
| `test_fa_custom_ops` | KV-cache 前向推理（含 Paged KV、Causal、Softcap） |
| `test_fa_fwd_custom_ops` | 标准前向推理（含 Causal、Return Attn Probs） |
| `test_fa_varlen_ops` | 变长序列前向推理（含不同 head_size、GQA） |

**测试参数化维度**：
- 数据类型：`torch.float16` / `torch.bfloat16`
- 批大小：1, 2, 5, 7
- Q/KV 头数：1×1, 4×4, 5×1 (GQA), 2×1 (GQA)
- 序列长度：512, 1024, 513, 777, 888, 1777, 1888, 7777, 8192
- 头维度：64, 111, 128, 192, 256
- 是否 Causal：True / False
- Cache 模式：标准 / Paged KV
- Softcap：0.0 / 30.0 / 50.0

**精度验证**：使用 `torch.testing.assert_close` 与参考实现对比，容差 `rtol=1e-2, atol=1e-2`。

### 9.3 参考实现

测试中的参考实现 `ref_flash_attention` 使用纯 PyTorch 实现，采用分块计算模拟 FlashAttention 的 online softmax 逻辑：
1. 将 KV 序列按 512 长度分块
2. 每块计算 QK^T → Online Softmax → PV 乘法
3. 累积更新全局最大值和输出

---

## 10. 功能特性矩阵

### flash_attn_with_kvcache

| 特性 | v2 | v3 |
|------|----|----|
| FP16 (float16) | ✅ | ✅ |
| BF16 (bfloat16) | ✅ | ✅ |
| Causal Attention | ✅ | ✅ |
| Sliding Window Attention | ❌ | ❌ |
| MQA/GQA | ✅ | ✅ |
| Paged KV Cache | ✅ | ✅ |
| Rotary Positional Embedding (RoPE) | ❌ | ❌ |
| ALiBi | ❌ | ❌ |
| Softcapping | ✅ (仅 KV-cache) | ❌ |
| FP8 Quantization | ❌ | ❌ |
| Variable-length Sequences | ✅ | ✅ |
| Attention Chunk | ❌ | ❌ |
| 反向传播 | ✅ (非 KV-cache) | ❌ |

### v2 当前不支持的功能（C++ 层 TORCH_CHECK 拦截）

- `alibi_slopes`
- `leftpad_k`
- `rotary_cos` / `rotary_sin` (RoPE)
- `dropout` (p_dropout != 0)
- `window_size_left` / `window_size_right`
- `return_softmax` (仅 v2 fwd)
- `softcap` (仅 v2 标准前向不支持，KV-cache 支持)

### v3 当前不支持的功能

- `leftpad_k`
- `rotary_cos` / `rotary_sin` / `seqlens_rotary`
- `q_descale` / `k_descale` / `v_descale` (FP8)
- `softcap`
- `window_size_left` / `window_size_right`
- `attention_chunk`
- `scheduler_metadata`
- `pack_gqa`
- 反向传播

---

## 11. 架构设计要点

### 11.1 Online Softmax

FlashAttention 的核心优化是 **Online Softmax**，避免将完整的 QK^T 注意力矩阵物化到 HBM。实现分块计算：

1. 对每个 KV 分块计算局部 Softmax
2. 维护运行最大值 `m` 和累积和 `l`
3. 使用 `rescale_o` 修正已有输出

项目中提供两个版本：
- `online_softmax.hpp` — 高精度版本（float32 累积）
- `online_softmax_low_prec.hpp` — 低精度版本

### 11.2 Tiling 策略

Tiling 参数由 `GetQNBlockTile` 和 `GetQSBlockTile` 函数计算：

```cpp
Q_TILE_CEIL = 128
N_SPLIT_HELPER = 2

qNBlockTile = min(Q_TILE_CEIL / qSeqlen / N_SPLIT_HELPER * N_SPLIT_HELPER, groupSize)
qSBlockTile = Q_TILE_CEIL
```

任务总数 = Σ_batch (qNBlockNum × qSBlockNum)，其中：
- `qNBlockNum` = (groupSize + qNBlockTile - 1) / qNBlockTile × kvHeads
- `qSBlockNum` = (qSeqlen + qSBlockTile - 1) / qSBlockTile

### 11.3 双缓冲与流水线

内核使用 `PRELANCH_NUM = 3` 的预取缓冲，配合 AscendC 的 Cube/Vector 双引擎实现计算-搬运流水线：

- **Cube 引擎**：执行 QK MatMul / PV MatMul
- **Vector 引擎**：执行 Online Softmax / Rescale

通过 `HardEvent` 信号量（`QK_READY_ID`, `SOFTMAX_READY_ID`, `PV_READY_ID`）同步两个引擎。

### 11.4 掩码处理

Causal 掩码在 CPU 端预计算为 2048×2048 的上三角矩阵，传输到 NPU 设备端供内核使用：

```cpp
mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
```

### 11.5 FFTS 地址

内核启动前通过 `rtGetC2cCtrlAddr` 获取 FFTS (Fast Task Schedule) 地址，用于 Ascend NPU 的多核任务调度。

### 11.6 自定义算子注册

Python 层通过 `torch.library.custom_op` 注册自定义算子（PyTorch >= 2.4），并使用 `register_fake` 提供符号形状推导，以支持 `torch.compile()` 和 `torch.export()`：

```python
@_torch_custom_op_wrapper("flash_attn_npu::_flash_attn_forward", mutates_args=(), device_types="npu")
def _flash_attn_forward(q, k, v, ...) -> Tuple[torch.Tensor, ...]:
    ...
```

对于 PyTorch < 2.4，使用 no-op wrapper 降级处理。

### 11.7 Head Dimension 对齐

Python 层自动将 head dimension 对齐到 8 的倍数：

```python
if head_size_og % 8 != 0:
    q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
    k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
    v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
```

计算完成后截断回原始维度：`out = out_padded[..., :head_size_og]`
