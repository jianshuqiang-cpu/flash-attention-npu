# flash-attention-npu 长期发展规划

> 对标 Dao-AILab/flash-attention，基于昇腾 NPU (Ascend) 平台的 Flash Attention 实现演进路线

---

## 📊 一、功能差距全景分析

基于对 Dao-AILab/flash-attention（v2.0→v2.7→v3→v4）和当前项目的深入对比，梳理出以下差距矩阵：

| 功能特性 | Dao-AILab 状态 | flash-attention-npu 状态 | 差距等级 |
|---------|:---:|:---:|:---:|
| 标准前向/反向 (BSND) | ✅ | ✅ | 无 |
| Varlen 前向/反向 (TND) | ✅ | ✅ | 无 |
| KV Cache 推理 | ✅ | ✅ | 无 |
| Paged KV Cache | ✅ | ✅ | 无 |
| Causal Mask | ✅ | ✅ | 无 |
| GQA/MQA | ✅ | ✅ | 无 |
| BF16/FP16 | ✅ | ✅ | 无 |
| FlashDecode (Split-KV) | ✅ | ✅ | 无 |
| **Sliding Window Attention** | ✅ v2.3+ | ⚠️ v3 bwd 部分支持 | 🔴 高 |
| **ALiBi 位置编码** | ✅ v2.4+ | ❌ | 🔴 高 |
| **Softcapping** | ✅ v2.6+ | ❌ | 🔴 高 |
| **Rotary Embedding (融合)** | ✅ | ❌ | 🟡 中 |
| **Dropout** | ✅ | ❌ | 🟡 中 |
| **FP8 量化** | ✅ v3+ | ❌ | 🔴 高 |
| **Attention Chunk** | ✅ v3+ | ❌ | 🟡 中 |
| **Pack GQA** | ✅ v3+ | ❌ | 🟡 中 |
| **Deterministic 反向** | ✅ v2.4+ | ⚠️ 仅 v3 | 🟡 中 |
| **torch.compile 兼容** | ✅ v2.7+ | ⚠️ 基础支持 | 🟡 中 |
| **QKV/KV Packed 接口** | ✅ | ⚠️ 仅 Python 层 | 🟢 低 |
| **Split-KV 前向** | ✅ v3+ | ❌ | 🟡 中 |
| **Incoherent Processing (FP8)** | ✅ v3+ | ❌ | 🔴 高 |
| **Warp-specialization 重叠** | ✅ v3+ | ❌ (NPU 架构不同) | 🟠 需适配 |
| **FA4 新型 Online Softmax** | ✅ v4 | ❌ | 🔴 高 |

---

## 🗺️ 二、分阶段发展路线图

### 🟢 Phase 1：功能补齐期（0-6 个月）

**目标**：补齐 Dao-AILab flash-attention v2.3-v2.7 的核心特性，达到 LLM 训练/推理的基本可用性。

#### 1.1 Sliding Window Attention（优先级：🔴 最高）

**对标**：Dao-AILab v2.3，适配 Mistral/Qwen2 等模型

- v3 反向已有 band mask 基础，需完善前向的 sliding window 路径
- 在 `online_softmax.hpp` 中增加 window 边界检查，跳过窗口外的 KV 块
- 修改 tiling 逻辑，使 S2 维度的分块仅覆盖 `[max(0, s1-window_left), min(s2, s1+window_right+1)]` 范围
- Python API 层面对齐 `window_size=(left, right)` 参数

**关键修改点**：
- `fag_tiling.cpp` — Tiling 增加 window 范围计算
- `online_softmax.hpp` — Softmax 增加 window mask
- `flash_api.cpp` — 移除 `window_size_left/right == -1` 的 TORCH_CHECK 限制

#### 1.2 Softcapping（优先级：🔴 最高）

**对标**：Dao-AILab v2.6，适配 Gemma-2/Grok 模型

- 在 Online Softmax 的 `CalcExp()` 步骤前，插入 `tanh(S / softcap)` 操作
- 需在 `online_softmax.hpp` 的 UB tensor 布局中预留 softcap 中间结果空间
- 反向传播需计算 `d(tanh(x/softcap))/dx = (1 - tanh²(x/softcap)) / softcap`
- Python API 对齐 `softcap` 参数

#### 1.3 ALiBi 位置编码（优先级：🔴 高）

**对标**：Dao-AILab v2.4，适配 MPT/Bloom 等模型

- ALiBi 本质是在 S 矩阵上叠加一个线性偏置：`S += alibi_bias`
- 需新增 `alibi_slopes` 参数传入，在 UB 上计算 `slopes * (j - i)` 偏置矩阵
- 在 `online_softmax.hpp` 的 `CalcExp()` 之前叠加偏置
- 需考虑偏置矩阵在 UB 上的空间占用

#### 1.4 Rotary Embedding 融合（优先级：🟡 中）

**对标**：Dao-AILab KV Cache 接口中的 `rotary_cos/sin`

- 当前 KV Cache 前向接口已预留 `rotary_cos_`/`rotary_sin_` 参数但未实现
- 需在 K/V 加载到 L1 之前，在 Vector 侧执行 rotary embedding 计算
- 实现 `apply_rotary_emb()` 在 AIV 上的 kernel

#### 1.5 Dropout 支持（优先级：🟡 中）

**对标**：Dao-AILab v2.0 基础功能

- 训练场景必需，但 NPU 上随机数生成需使用 AscendC 的 `Random` 接口
- 在 Online Softmax 的 P 矩阵上应用 mask：`P = P * mask / (1 - p)`
- 反向传播需保存 mask 或使用确定性种子重生成

---

### 🔵 Phase 2：性能与精度提升期（6-12 个月）

**目标**：对标 FlashAttention-3，引入低精度计算和 NPU 架构深度优化。

#### 2.1 FP8 量化支持（优先级：🔴 最高）

**对标**：FlashAttention-3 FP8 路径

- 昇腾 950 (dav-3510) 已支持 FP8 数据类型
- 需实现 `q_descale`/`k_descale`/`v_descale` 的反量化路径
- 在 QK/PV 矩阵乘前将 FP8 反量化为 FP16/BF16
- 参考 `csrc/catlass/examples/29_a2_fp8_e4m3_matmul` 的 FP8 GEMM 实现

#### 2.2 Incoherent Processing（优先级：🔴 高）

**对标**：FlashAttention-3 的 FP8 精度优化

- 通过随机正交矩阵乘法分散异常值，减少量化误差
- 需在 NPU 上实现 Hadamard 变换的快速计算
- 这是 FP8 路径精度保障的关键技术

#### 2.3 Cube+Vector 流水线深度优化（优先级：🔴 高）

**对标**：FlashAttention-3 的 GEMM-Softmax 重叠

- NPU 的 AIC (Cube) 和 AIV (Vector) 天然分离，可实现比 GPU 更彻底的计算-计算重叠
- 当 Cube 在计算 PV 矩阵乘时，Vector 可并行执行下一块的 Softmax
- 需重构 `fag_general_host.hpp` 的流水线调度逻辑
- 利用 AscendC 的 `SetFlag<PIPE_VS>()` / `WaitFlag<PIPE_VS>()` 实现跨引擎同步

#### 2.4 Attention Chunk Prefill（优先级：🟡 中）

**对标**：FlashAttention-3 `attention_chunk` 参数

- 长序列 prefill 场景下，将 Q 序列分 chunk 处理，减少峰值内存
- 950 版本已有初步实现：`csrc_AscendC950/flash_attention_chunk_prefill`
- 需移植到 910B/C 并统一 API

#### 2.5 Pack GQA 优化（优先级：🟡 中）

**对标**：FlashAttention-3 `pack_gqa` 参数

- 优化 GQA 的内存布局，将同一 KV 头对应的多个 Q 头连续排列
- 减少 KV 的重复加载，提升 L1 命中率
- 需修改 tiling 逻辑中的 `groupSize` 处理方式

---

### 🟣 Phase 3：生态融合期（12-18 个月）

**目标**：融入主流 NPU 推理/训练框架，成为昇腾生态的注意力算子标准。

#### 3.1 vLLM-NPU 适配（优先级：🔴 最高）

- vLLM 是当前最流行的 LLM 推理框架，其 PagedAttention 核心依赖 FlashAttention
- 需确保 `flash_attn_with_kvcache` 的 `page_table`/`block_table` 接口与 vLLM-NPU 版本兼容
- 需实现 `flash_attn_varlen_func` 的 continuous batching 支持
- 提供 vLLM 的 NPU attention backend 适配层

#### 3.2 Megatron-LM / DeepSpeed NPU 适配（优先级：🔴 高）

- 训练框架的分布式注意力（TP/SP/CP）依赖 FlashAttention
- 需实现序列并行 (Sequence Parallelism) 的 all-gather/reduce-scatter 与 FA 的融合
- 提供与 Megatron-Core 兼容的注意力接口

#### 3.3 HuggingFace Transformers 集成（优先级：🟡 中）

- Transformers 已原生支持 `attn_implementation="flash_attention_2"` / `"flash_attention_3"`
- 需注册 `flash_attention_npu` 作为新的 attention implementation
- 或通过 monkey-patch 让 `flash_attention_2` 在 NPU 上自动路由到本项目

#### 3.4 torch.compile 深度兼容（优先级：🟡 中）

- 当前已有基础 `torch.library.custom_op` 注册，需完善：
  - 所有变体的 `register_fake` 实现（正确的 shape/dtype 推导）
  - 支持动态 shape（`SymInt`）的编译
  - 与 `torch.export` 的兼容性
  - AOTAutograd 的正确梯度图构建

---

### 🟠 Phase 4：前沿对齐期（18-24 个月+）

**目标**：对标 FlashAttention-4，探索 NPU 原生的新型注意力算法。

#### 4.1 新型 Online Softmax 算法（优先级：🔴 最高）

**对标**：FlashAttention-4 的 "skip 90% rescaling" 算法

- FA4 提出了一种新的在线 Softmax 算法，可跳过 90% 的输出 rescaling 操作
- 需研究该算法在 NPU Cube+Vector 架构上的适配方案
- 核心思路：利用指数的软件模拟（FA4 使用 `MUFU.EX2`）来提高吞吐
- NPU 上需寻找等价的快速 exp 近似计算方法

#### 4.2 稀疏注意力支持（优先级：🟡 中）

**对标**：FlashAttention 的 block-sparse attention

- 支持 MoE 等稀疏模型架构的注意力计算
- 需设计稀疏 mask 的高效编码和分发策略
- 在 tiling 层面跳过全零块的计算

#### 4.3 多模态注意力（优先级：🟢 低）

- 支持跨模态（文本-图像-视频）的注意力计算
- 处理不同模态不同序列长度的情况
- 需扩展 varlen 接口支持多模态 batch

#### 4.4 新一代昇腾芯片适配（优先级：🟡 中）

- 持续跟进昇腾新架构（dav-xxx）的硬件特性
- 类似 FA3 适配 Hopper 新特性（WGMMA/TMA/FP8），需适配昇腾新指令集
- 可能包括：更大的 L1/L0 缓存、新的 Cube 指令、更高的 AIV 吞吐

---

## 🏗️ 三、技术架构演进规划

### 3.1 代码架构统一

当前 v2 和 v3 两套代码并存，长期需统一：

```
当前架构:
├── csrc/flash_attn_npu/          # v2 (5个API, 独立实现)
├── csrc/flash_attn_npu_v3/       # v3 910 (2个API, FAG kernel)
├── csrc_AscendC950/              # v3 950 (独立代码树)
└── catlass/                      # 共享模板库

目标架构:
├── csrc/
│   ├── core/                     # 统一核心
│   │   ├── tiling/               # 统一 Tiling 框架
│   │   ├── kernel/               # 统一 Kernel 入口
│   │   └── dispatch/             # 运行时后端分发
│   ├── kernels/
│   │   ├── fwd/                  # 前向 kernel 族
│   │   ├── bwd/                  # 反向 kernel 族
│   │   └── kvcache/              # KV Cache kernel 族
│   ├── features/                 # 可选特性模块
│   │   ├── sliding_window/
│   │   ├── alibi/
│   │   ├── softcap/
│   │   ├── rotary/
│   │   ├── dropout/
│   │   └── fp8/
│   └── backends/
│       ├── dav_2201/             # 910B/C 专用
│       └── dav_3510/             # 950 专用
├── catlass/                      # 持续演进
└── python/                       # 统一 Python 包
    └── flash_attn_npu/           # 单一入口，运行时分发
```

### 3.2 Catlass 模板库演进

Catlass 是本项目的核心竞争力，需持续对标 CUTLASS：

| CUTLASS 特性 | Catlass 当前状态 | 演进方向 |
|-------------|:---:|------|
| 三级分块 (Device/Block/Tile) | ✅ | 保持 |
| Epilogue 融合 | ✅ 基础 | 扩展更多融合模式 |
| Swizzle 布局 | ✅ | 优化 bank conflict |
| Stream-K/Split-K | ✅ | 统一调度框架 |
| FP8 支持 | ⚠️ 示例级 | 产品级支持 |
| Conv 融合 | ✅ 基础 | 深度融合 |
| Python DSL (CuTe) | ❌ | 评估是否需要 |
| 自动调优 (Tuner) | ✅ 基础 | 强化搜索空间 |

### 3.3 测试与基准体系

**当前不足**：测试覆盖有限，缺乏系统性基准

**目标建立**：

1. **功能正确性测试**
   - 每个 API 变体的前向/反向数值正确性（与 PyTorch 参考实现对比）
   - 边界条件测试（head_dim=64/128/192/256, seqlen=1/128/2048/8192/32768）
   - 混合精度测试（FP16/BF16/FP8）

2. **性能基准测试**
   - 对标 `torch.nn.functional.scaled_dot_product_attention` (NPU 版)
   - 不同 batch/seq/head 组合下的吞吐量和延迟
   - 与 CUDA 版 FlashAttention 的性能比率追踪

3. **回归测试 CI**
   - GitHub Actions 自动化测试
   - 每次提交运行核心测试集
   - 每日/每周运行完整基准

---

## 📈 四、版本发布节奏建议

| 版本 | 时间 | 核心交付 | 对标 |
|------|------|---------|------|
| **v2.1** | +2月 | Sliding Window + Softcap | FA v2.3+v2.6 |
| **v2.2** | +4月 | ALiBi + Rotary + Dropout | FA v2.4 |
| **v2.3** | +6月 | torch.compile 完善 + 确定性反向 | FA v2.7 |
| **v3.1** | +9月 | FP8 + Cube/Vector 重叠优化 | FA v3 |
| **v3.2** | +12月 | Attention Chunk + Pack GQA + Incoherent | FA v3 |
| **v4.0** | +18月 | vLLM/Megatron 集成 + 架构统一 | 生态融合 |
| **v5.0** | +24月 | 新型 Online Softmax + 稀疏注意力 | FA v4 |

---

## 🎯 五、核心竞争力建设

### 5.1 差异化优势方向

本项目不应只是 CUDA 版的简单移植，而应发挥 NPU 架构的独特优势：

1. **Cube+Vector 双引擎天然重叠**：GPU 需要 warp-specialization 才能实现 GEMM-Softmax 重叠，而 NPU 的 AIC/AIV 天然分离，理论上可实现更彻底的流水线重叠
2. **统一内存架构**：昇腾 NPU 的 L1/UB 统一编址，比 GPU 的 Shared Memory 更灵活
3. **Catlass 模板库**：作为 CUTLASS 的 NPU 对标，本身就有独立价值，可服务于更多算子开发

### 5.2 风险与挑战

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| 昇腾新架构指令集变化 | 代码需大幅适配 | Catlass 抽象层隔离硬件差异 |
| Dao-AILab 迭代速度极快 | 持续追赶压力 | 聚焦核心特性，不追求 100% 对齐 |
| NPU 生态碎片化 | 多版本维护成本高 | 运行时分发 + 编译时模板特化 |
| 人才稀缺 | AscendC 开发者少 | 完善 Catlass 文档和示例降低门槛 |
| FP8 精度问题 | 影响模型质量 | Incoherent Processing + 严格数值测试 |

---

## 📋 六、近期行动清单（前 3 个月）

| 周次 | 任务 | 产出 |
|------|------|------|
| W1-W2 | Sliding Window 前向实现 | `window_size` 参数在 v2/v3 前向生效 |
| W3-W4 | Sliding Window 反向实现 | FAG kernel 支持 band mask |
| W5-W6 | Softcap 前向实现 | `softcap` 参数在 Online Softmax 中生效 |
| W7-W8 | Softcap 反向实现 | 反向传播正确计算 softcap 梯度 |
| W9-W10 | ALiBi 前向实现 | `alibi_slopes` 参数生效 |
| W11-W12 | 基准测试框架搭建 | 自动化 benchmark 脚本 + 基线数据 |

---

## 📌 总结

**核心差距**：当前项目与 Dao-AILab flash-attention 最大的功能缺口是 **Sliding Window、ALiBi、Softcap、FP8** 四项，这些是主流 LLM（Mistral、Gemma-2、MPT 等）的刚需特性。

**最大差异化机会**：NPU 的 **Cube+Vector 双引擎天然分离**架构，使得 GEMM-Softmax 流水线重叠比 GPU 更容易实现——GPU 需要 FlashAttention-3 才通过 warp-specialization 实现的部分重叠，在 NPU 上可以做得更彻底。这是本项目超越"简单移植"的关键突破口。

**最紧迫行动**：建议先从 **Sliding Window + Softcap** 入手，因为这两个特性实现复杂度相对较低（主要修改 Online Softmax epilogue），但对模型覆盖面的提升最大（Mistral + Gemma-2）。
