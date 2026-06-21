# fag_mmad_cube1.hpp 解析

## 文件定位

`fag_mmad_cube1.hpp` 实现了 FAG 反向传播中的 Cube1 矩阵乘模块，核心类型是：

```cpp
BlockMmad<MmadAtlasA2FAGCube1, ...>
```

它运行在 Cube/AIC 侧，消费 `CubeAddrInfo` 中生成好的 GM 地址和块形状，把输入矩阵从 GM 搬到 L1，再搬到 L0A/L0B，调用 `tileMmad` 做矩阵乘，最后把 fp32 的 `128x128` 结果块写入 GM workspace。

在 `mha_varlen_bwd.cpp` 中，Cube1 被复用为两类矩阵乘：

```text
1. scores = Q * K^T
2. dP     = dOut * V^T
```

这两类计算都符合“左矩阵不转置、右矩阵按 token 方向形成转置视图”的模式：

```text
A: (Q_block, headDim)
B: (K_or_V_block, headDim)
C: (Q_block, K_block)

实际数学含义：
C = A * B^T
```

## 核心输入输出

`operator()` 的签名是：

```cpp
void operator()(const CubeAddrInfo &addrs,
                __gm__ ElementA *left,
                __gm__ ElementB *right,
                __gm__ float *out,
                uint32_t &pingpongFlagL1A,
                uint32_t &pingpongFlagL0A,
                uint32_t &pingpongFlagL1B,
                uint32_t &pingpongFlagL0B,
                uint32_t &pingpongFlagC)
```

含义如下：

- `addrs`：地址生成器输出的任务描述，最多包含 16 个 `AddrInfo`。
- `left`：左矩阵基地址，可能是 `q` 或 `dout`。
- `right`：右矩阵基地址，可能是 `k` 或 `v`。
- `out`：Cube1 输出 workspace，可能是 `mm2Workspace` 或 `mm1Workspace`。
- `pingpongFlag*`：L1/L0/L0C 各级缓冲的乒乓状态。

`AddrInfo` 中关键字段：

```text
left/right/out : GM 线性偏移
ky             : 当前 Q 方向真实长度，也就是 M
kx             : 当前 K 方向真实长度，也就是 N
upperRight     : causal 边界标志
lowerLeft      : causal 边界标志
```

## 整体数据流

```text
CubeAddrInfo
    │
    │ 取 AddrInfo.left/right/out/kx/ky
    ▼
GM left/right
    │
    │ copyGmToL1A / copyGmToL1B
    ▼
L1A / L1B ping-pong
    │
    │ copyL1ToL0A / copyL1ToL0B
    ▼
L0A / L0B ping-pong
    │
    │ tileMmad
    ▼
L0C fp32 ping-pong
    │
    │ copyL0CToGm
    ▼
GM workspace
```

Cube1 的输出不是最终的 `dq/dk/dv`，而是后续 Vector epilogue 会读取的中间结果：

```text
Q*K^T      -> scores workspace，供重新计算 P/softmax 使用
dOut*V^T  -> dP workspace，供计算 dS 使用
```

## workspace 乒乓布局

文件中定义：

```cpp
static const uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;
```

含义是每个 Cube task 最多写 16 个 `128x128` fp32 块。

每个 AIC 使用两份 task slot 做乒乓：

```cpp
pingPongIdx = addrs.taskId % 2;
globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;
```

可以画成：

```text
workspace for one AIC
┌──────────────────────────────┬──────────────────────────────┐
│ task slot 0                  │ task slot 1                  │
│ 16 个 128x128 block          │ 16 个 128x128 block          │
└──────────────────────────────┴──────────────────────────────┘
        ▲ taskId % 2 = 0               ▲ taskId % 2 = 1
```

这样 Cube 写当前 slot 时，Vector 可以消费另一个 slot，减少读写冲突。

## `cube1_base_matmul` 的作用

`cube1_base_matmul` 假设 A/B 已经在 L1：

```text
A in L1: (M, D)
B in L1: (D, N)
```

然后它做三件事：

1. 按 N 方向每 128 列把 B 从 L1 搬到 L0B。
2. 按 M 方向每 128 行把 A 从 L1 搬到 L0A。
3. 调用 `tileMmad` 得到 C 子块，并写入 workspace。

循环结构可以理解为：

```text
for n_offset in N step 128:
    L1B -> L0B
    for m_offset in M step 128:
        if causal 右上无效块:
            skip
        else:
            L1A -> L0A
            L0C = L0A * L0B
            L0C -> GM workspace[cube1Cnt]
            cube1Cnt++
```

其中 `cube1Cnt` 是当前 `AddrInfo` 内已经写出的有效输出块数量。被 causal 跳过的块不会占用 workspace，因此后续 Vector 阶段读取的是压缩后的有效块序列。

## causal 右上角跳过

在 causal attention 中，理论上只需要下三角：

```text
K/token →
Q/token ↓

有效区域：q >= k

┌───────────────┐
│ 有效  无效     │
│ 有效  有效     │
└───────────────┘
```

代码中：

```cpp
bool upperRight = !shapeInfo.upperRight;
bool upper_right_flag = (upperRight && n_index == n_loop - 1);
bool l0_skip_flag = (upper_right_flag && m_offset == 0);
```

含义是：如果当前 `AddrInfo` 触碰 causal 右上边界，那么最后一个 N 子块里的第一个 M 子块可能完全无效，可以不搬 A、不做 MMAD、不写 workspace。

## 例子 1：普通 128x128 块

假设：

```text
km = 128
kn = 128
headdim = 128
upper_right_flag = false
```

则 Cube1 执行：

```text
A: 128x128
B: 128x128
C: 128x128
```

流程：

```text
GM A(128x128) -> L1A -> L0A
GM B(128x128) -> L1B -> L0B
L0C = L0A * L0B
L0C -> workspace 第 0 个 128x128 块
cube1Cnt: 0 -> 1
```

不会发生跳过，workspace 中保留一个完整结果块。

## 例子 2：causal 边界块

假设某个块跨过 causal 右上边界：

```text
km = 256
kn = 128
upper_right_flag = true
```

M 方向会被切成两个 128 行子块：

```text
n_offset = 0
    m_offset = 0    -> 右上无效块，skip
    m_offset = 128  -> 有效块，计算并写 workspace[0]
```

图示：

```text
当前 attention 子矩阵

K/token →
┌────────────────┐
│ skip           │  m_offset = 0，右上无效
├────────────────┤
│ compute        │  m_offset = 128，有效
└────────────────┘
Q/token ↓
```

这里虽然逻辑上存在两个 `128x128` 子块，但只有有效的第二块写入 workspace，因此 `cube1Cnt` 最终只增加一次。

## L1/L0 ping-pong 的意义

文件中维护多组 ping-pong buffer：

```text
L1A: ping / pong
L1B: ping / pong
L0A: ping / pong
L0B: ping / pong
L0C: ping / pong
```

配合 `AscendC::WaitFlag` / `AscendC::SetFlag`，大致形成这样的流水：

```text
MTE2: GM -> L1
MTE1: L1 -> L0
M:    MMAD compute
MTE1: L0C -> GM
```

ping-pong 的核心目的不是改变数学结果，而是让数据搬运和计算尽量重叠，提高 AIC 利用率。

## 与后续阶段的关系

Cube1 的结果会被 Vector epilogue 使用：

```text
Cube1(Q, K)       -> scores workspace -> Vector 重新计算 softmax P
Cube1(dOut, V)    -> dP workspace     -> Vector 计算 dS
Vector/FAGOp      -> 生成后续 Cube2/Cube3 需要的中间量
FAGPost           -> 输出 dq/dk/dv
```

因此 `fag_mmad_cube1.hpp` 是 FAG 反向主循环中连接原始输入张量和 softmax-gradient 向量阶段的第一类 Cube 矩阵乘实现。

## 注意点

- 本文件主要操作 GM/L1/L0/L0C，不直接处理最终梯度输出。
- `CubeAddrInfo` 已经完成 batch、head、GQA/MQA、TND/BSND 布局下的地址折算。
- `BLOCK_WORKSPACE` 和 `taskId % 2` 是 Cube/Vector 流水同步的重要约定。
- causal 跳过会压缩 workspace 中的有效块序列，`cube1Cnt` 必须只对实际写出的块递增。
- `m_mad_ == 1` 时会改成 2 行参与 MMAD，这是硬件形状限制相关处理，额外行来自 padding，不改变有效输出。
