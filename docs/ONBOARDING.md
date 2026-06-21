# FlashAttention for Ascend NPU — 新成员入职指南

> **生成时间**: 2026-06-17
> **项目版本**: 0.1.1
> **许可证**: BSD 3-Clause

---

## 目录

1. [项目概述](#一项目概述)
2. [架构分层](#二架构分层)
3. [关键概念](#三关键概念)
4. [引导式学习路径](#四引导式学习路径)
5. [文件地图](#五文件地图)
6. [复杂度热点](#六复杂度热点)
7. [环境搭建与首次运行](#七环境搭建与首次运行)
8. [功能特性矩阵](#八功能特性矩阵)
9. [推荐学习资源](#九推荐学习资源)

---

## 一、项目概述

| 项目属性 | 详情 |
|---------|------|
| **项目名称** | flash-attention-npu |
| **版本** | 0.1.1 |
| **许可证** | BSD 3-Clause |
| **作者** | Minghua Shen (shenmh6@mail.sysu.edu.cn) |
| **仓库** | https://github.com/MinghuasLab/flash-attention-npu |

**项目定位**：本项目是 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的 **Ascend NPU 适配实现**。FlashAttention 通过分块计算（Tiling）和内存感知算法显著提升大语言模型的训练与推理效率。原始实现面向 NVIDIA GPU，本项目基于华为 [CANN/CATLASS](https://gitcode.com/cann/catlass) 框架，在 Ascend NPU 上实现了 API 兼容的 FlashAttention 算法。

**核心目标**：
- 提供与 Dao-AILab/flash-attention API 一致的接口，降低模型迁移成本
- 基于 Ascend NPU 硬件特性（Cube/Vector 双引擎）进行算子优化
- 支持 FlashAttention v2 和 v3 两代算法

**技术栈**：
- **编程语言**：Python（接口层）、C++/AscendC（内核层）
- **框架依赖**：PyTorch >= 2.1.0、torch_npu >= 2.1.0、CANN >= 8.5.0
- **目标硬件**：Ascend 910B / 910C NPU
- **编译器**：bisheng（华为 AscendC 编译器），目标架构 `dav-2201`

---

## 二、架构分层

项目采用 **Python 接口层 + C++/AscendC 内核层** 的分层架构，自上而下共 6 层：

```
┌─────────────────────────────────────────────────────────┐
│  第1层：用户代码（PyTorch 模型）                           │
├─────────────────────────────────────────────────────────┤
│  第2层：Python 接口层                                     │
│  ┌────────────────────┐  ┌────────────────────────┐    │
│  │ v2: flash_attn_npu/ │  │ v3: flash_attn_npu_v3/ │    │
│  │  (7个公共API)        │  │  (1个公共API)           │    │
│  └─────────┬──────────┘  └──────────┬─────────────┘    │
│            │ pybind11               │ pybind11          │
├────────────┼────────────────────────┼──────────────────┤
│  第3层：C++/AscendC 内核层                                │
│  ┌────────────────────┐  ┌────────────────────────┐    │
│  │ v2: csrc/          │  │ v3: csrc/              │    │
│  │  flash_attn_npu/   │  │  flash_attn_npu_v3/    │    │
│  │  (fwd/bwd/kvcache) │  │  (仅fwd)               │    │
│  └─────────┬──────────┘  └──────────┬─────────────┘    │
│            │                        │                   │
│  第4层：CATLASS 框架 (csrc/catlass submodule)             │
│  ┌──────────────────────────────────────────────────┐   │
│  │ AscendC 算子基础设施、矩阵乘/Epilogue/Tiling抽象  │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  第5层：华为 CANN 软件栈 (>= 8.5.0)                       │
│  bisheng 编译器 / ACL 运行时 / Tiling API / Platform API │
├─────────────────────────────────────────────────────────┤
│  第6层：Ascend NPU 硬件 (910B/910C)                       │
│  Cube 引擎（矩阵乘）+ Vector 引擎（Softmax/Rescale）       │
└─────────────────────────────────────────────────────────┘
```

### 各层职责

| 层级 | 名称 | 关键目录 | 职责 |
|------|------|---------|------|
| L2 | Python 接口层 (v2) | `flash_attn_npu/` | 封装 C++ 扩展、实现 autograd.Function、注册 custom_op |
| L2 | Python 接口层 (v3) | `flash_attn_npu_v3/` | v3 接口，支持更多特性（page_table、FP8、chunk 等） |
| L3 | C++ 内核层 (v2) | `csrc/flash_attn_npu/` | pybind11 入口、Tiling 计算、AscendC Kernel 实现 |
| L3 | C++ 内核层 (v3) | `csrc/flash_attn_npu_v3/` | v3 内核，统一前向（含 KV-cache） |
| L4 | CATLASS 框架 | `csrc/catlass` (子模块) | AscendC 矩阵运算框架（类似 CUTLASS） |
| — | AscendC950 变体 | `csrc_AscendC950/` | chunk prefill 独立示例（CMake 构建） |

---

## 三、关键概念

新成员需要理解以下核心概念才能有效参与开发：

### 3.1 Online Softmax（在线 Softmax）

FlashAttention 的核心优化，避免将完整 QK^T 注意力矩阵物化到 HBM。分块计算每个 KV 块的局部 Softmax，维护运行最大值 `m` 和累积和 `l`，通过 `rescale_o` 修正已有输出。

项目提供两个版本：
- `online_softmax.hpp` — 高精度版本（float32 累积）
- `online_softmax_low_prec.hpp` — 低精度版本

### 3.2 Tiling 策略

Tiling 是 Ascend NPU 算子的关键概念，将计算任务分解到多个 AI Core 上并行执行。核心常量定义在 `kernel_common.hpp`：

- `Q_TILE_CEIL = 128` — Q 分块大小
- `MAX_KV_STACK_LEN = 512` — KV 最大堆叠长度
- `N_SPLIT_HELPER = 2` — N 维分割辅助因子

任务总数 = Σ_batch (qNBlockNum × qSBlockNum)，由 `GetQNBlockTile` / `GetQSBlockTile` 计算。

### 3.3 Cube/Vector 双引擎流水线

Ascend NPU 的计算架构：
- **Cube 引擎**：执行 QK MatMul / PV MatMul（矩阵乘加速）
- **Vector 引擎**：执行 Online Softmax / Rescale（逐元素运算）

通过 `HardEvent` 信号量（`QK_READY_ID=1`, `SOFTMAX_READY_ID=2`, `PV_READY_ID=3`）同步两个引擎，配合 `PRE_LAUNCH=2` 预取缓冲实现计算-搬运流水线。

### 3.4 自定义算子注册（torch.compile 兼容）

Python 层通过 `torch.library.custom_op` 注册自定义算子（PyTorch >= 2.4），并使用 `register_fake` 提供符号形状推导，以支持 `torch.compile()` 和 `torch.export()`。对 PyTorch < 2.4 使用 no-op wrapper 降级。

### 3.5 autograd.Function

通过 `torch.autograd.Function` 子类实现自动微分，前向调用 `_flash_attn_forward` custom op，反向调用 `_flash_attn_backward` custom op。

### 3.6 MQA/GQA

多查询注意力 / 分组查询注意力，通过 `nheads_k < nheads` 实现。Q 的头数必须能被 KV 的头数整除。

### 3.7 Paged KV Cache

内存高效的 KV 缓存管理，通过 `block_table`（v2）或 `page_table`（v3）实现分页。块大小通常为 128 或 256。

### 3.8 Head Dimension 对齐

Python 层自动将 head dimension 对齐到 8 的倍数（NPU kernel 要求），计算完成后截断回原始维度。

### 3.9 FFTS（Fast Task Schedule）

内核启动前通过 `rtGetC2cCtrlAddr` 获取 FFTS 地址，用于 Ascend NPU 的多核任务调度。

---

## 四、引导式学习路径

建议新成员按以下顺序逐步了解项目：

### 第 1 步：理解项目目标
阅读 `README.zh.md`，理解项目是 Dao-AILab/flash-attention 的 NPU 适配，以及 v2/v3 的功能特性矩阵。

### 第 2 步：熟悉 Python 接口层
从 `flash_attn_npu/__init__.py` 开始，了解 7 个导出的公共 API。然后阅读 `flash_attn_interface.py` 中的工具函数 `maybe_contiguous`、`round_multiple`，理解输入预处理逻辑。

### 第 3 步：理解 autograd.Function 模式
阅读 `FlashAttnFunc`、`FlashAttnQKVPackedFunc` 等 6 个 autograd.Function 子类，理解前向/反向如何调用 custom op。

### 第 4 步：深入 C++ 入口
阅读 `csrc/flash_attn_npu/flash_api.cpp`，理解 pybind11 模块注册的 5 个函数（`mha_fwd`、`mha_bwd`、`mha_fwd_kvcache`、`mha_varlen_fwd`、`mha_varlen_bwd`），以及模板实例化策略。

### 第 5 步：理解 Tiling 数据结构
阅读 `tilingdata.h`，理解 `FAInferTilingData` 结构体的 22 个字段（注意力维度、Paged KV、序列长度、任务调度、Workspace 大小等）。

### 第 6 步：探索内核实现
按数据流顺序阅读内核文件：
1. `qk_matmul.hpp` — QK 矩阵乘
2. `online_softmax.hpp` — Online Softmax
3. `pv_matmul.hpp` — PV 矩阵乘
4. `rescale_o.hpp` — 输出重缩放

### 第 7 步：运行测试
```bash
pytest -q -s tests/test_flash_attn_npu.py      # v2 测试
pytest -q -s tests/test_flash_attn_npu_bwd.py   # v2 反向测试
pytest -q -s tests/test_flash_attn_npu_v3.py     # v3 测试
```
阅读 `tests/test_flash_attn_npu.py` 中的参考实现 `ref_flash_attention`，理解分块计算模拟 online softmax 的逻辑。

---

## 五、文件地图

### 5.1 Python 接口层

| 文件 | 职责 | 复杂度 |
|------|------|--------|
| `flash_attn_npu/__init__.py` | v2 包初始化，导出 7 个公共 API | 低 |
| `flash_attn_npu/flash_attn_interface.py` | v2 核心接口实现（~1808 行），含工具函数、custom op、autograd.Function、公共 API | **高** |
| `flash_attn_npu_v3/__init__.py` | v3 包初始化，导出 `flash_attn_with_kvcache` | 低 |
| `flash_attn_npu_v3/flash_attn_interface.py` | v3 核心接口实现，支持 page_table、FP8、chunk、softcap 等 | 中 |

### 5.2 C++/AscendC 内核层 (v2)

| 文件 | 职责 | 复杂度 |
|------|------|--------|
| `flash_api.cpp` | pybind11 入口，注册 fwd/bwd/fwd_kvcache/varlen_fwd/varlen_bwd | **高** |
| `mha_fwd_kvcache.cpp` | KV-cache 前向内核的 AscendC 实现 | **高** |
| `mha_varlen_bwd.cpp` | 变长序列反向传播的 FAG 内核实现 | **高** |
| `fag_tiling.cpp` | FAG 模式的 Tiling 参数计算 | 中 |
| `softmax_tiling.cpp` | Softmax Tiling 参数计算 | 中 |
| `tilingdata.h` | `FAInferTilingData` 结构体定义（22 字段 + 44 方法） | 中 |
| `kernel_common.hpp` | 内核常量、枚举类型（MaskType/InputLayout）、工具函数 | 中 |
| `online_softmax.hpp` | Online Softmax 高精度实现 | 中 |
| `qk_matmul.hpp` | QK 矩阵乘法内核 | 中 |
| `pv_matmul.hpp` | PV 矩阵乘法内核 | 中 |
| `rescale_o.hpp` | 输出重缩放 | 中 |

### 5.3 构建与测试

| 文件 | 职责 |
|------|------|
| `setup.py` | 构建脚本，`BishengBuildExt` 调用 bisheng 编译器，支持预编译 wheel 下载 |
| `Makefile` | 辅助构建目标（clean_dist / create_dist / upload_package） |
| `tests/test_flash_attn_npu.py` | v2 前向/反向/KV-cache/Varlen 测试 |
| `tests/test_flash_attn_npu_bwd.py` | v2 反向传播专项测试 |
| `tests/test_flash_attn_npu_v3.py` | v3 前向/KV-cache 测试 |
| `tests/precision_compare.py` | 精度对比工具 |

---

## 六、复杂度热点

以下区域代码量大、逻辑复杂，新成员应谨慎对待：

### 6.1 `flash_api.cpp`（v2，~967 行）— 最高复杂度

- **难点**：根据数据类型（bf16/fp16）、是否 Paged KV、是否 Causal、是否 Varlen 四个维度组合实例化不同的内核模板
- **模板实例化**：`SplitFuse::FAInfer<数据类型, 数据类型, float, 是否PagedKV, MaskType, InputLayout, LseModeT>`
- **建议**：先理解 `mha_fwd_kvcache` 函数的参数校验和 Tiling 计算流程，再逐步理解模板分发逻辑

### 6.2 `flash_attn_interface.py`（v2，~1808 行）— 高复杂度

- **难点**：包含 26 个函数/类，分为 6 类（工具函数、custom op、autograd.Function、公共 API）
- **建议**：按「工具函数 → custom op 注册 → autograd.Function → 公共 API」的顺序阅读

### 6.3 `mha_fwd_kvcache.cpp` / `mha_varlen_bwd.cpp` — 高复杂度

- **难点**：AscendC Kernel 实现，涉及 Cube/Vector 双引擎流水线、Workspace 分配、FFTS 多核调度
- **Workspace 布局**：`mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize`，每个含 3 个预取缓冲

### 6.4 `fag_tiling.cpp` — 中高复杂度

- **难点**：Tiling 参数计算涉及多维度任务分解，需理解 `GetQNBlockTile` / `GetQSBlockTile` 的分块逻辑

---

## 七、环境搭建与首次运行

### 7.1 环境要求

- **硬件**：Ascend 910B / 910C NPU
- **系统**：Linux
- **软件**：CANN >= 8.5.0、PyTorch >= 2.1.0、torch_npu >= 2.1.0

### 7.2 安装步骤

```bash
# 1. 设置 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 2. 安装 Python 依赖
pip install packaging psutil ninja

# 3. 拉取源码（含子模块）
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive

# 4. 编译安装
python setup.py install

# 或仅编译特定版本
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install  # 仅 v2
FLASH_ATTN_BUILD_VERSION=v3 python setup.py install  # 仅 v3
```

### 7.3 构建控制环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `FLASH_ATTENTION_FORCE_BUILD` | FALSE | 强制本地编译，不下载预编译 wheel |
| `FLASH_ATTENTION_SKIP_NPU_BUILD` | FALSE | 跳过 NPU 编译（用于 sdist） |
| `FLASH_ATTN_BUILD_VERSION` | all | 构建版本：v2 / v3 / all |
| `FLASH_ATTN_LOCAL_VERSION` | - | 附加本地版本号 |

### 7.4 验证安装

```bash
pytest -q -s tests/test_flash_attn_npu.py
```

---

## 八、功能特性矩阵

| 特性 | v2 | v3 |
|------|----|----|
| FP16 (float16) | 支持 | 支持 |
| BF16 (bfloat16) | 支持 | 支持 |
| 因果注意力 (Causal) | 支持 | 支持 |
| MQA/GQA | 支持 | 支持 |
| 分页 KV 缓存 | 支持 | 支持 |
| 变长序列 | 支持 | 支持 |
| Softcapping | 支持 (仅 KV-cache) | 不支持 |
| 反向传播 | 支持 (非 KV-cache) | 不支持 |
| 滑动窗口注意力 | 不支持 | 不支持 |
| 旋转位置编码 (RoPE) | 不支持 | 不支持 |
| ALiBi | 不支持 | 不支持 |
| FP8 量化 | 不支持 | 不支持 |

---

## 九、推荐学习资源

1. **项目内文档**：`CODE_WIKI.md` — 详细的代码 Wiki（11 章节）
2. **原始论文**：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al.)
3. **CATLASS 框架**：https://gitcode.com/cann/catlass — 类似 CUTLASS 的 AscendC 矩阵运算框架
4. **CANN 文档**：AscendC 算子开发指南
