# fag_mmad_cube2.hpp 解析

## 文件定位

`fag_mmad_cube2.hpp` 实现 FAG 反向传播中的 Cube2 矩阵乘模块，核心类型是：

```cpp
BlockMmad<MmadAtlasA2FAGCube2, ...>
```

它运行在 Cube/AIC 侧，用来计算：

```text
dQ = dS * K
```

其中：

```text
dS: Vector epilogue 计算出的 softmax score 梯度，来自 ds workspace
K : 原始 key tensor
dQ: fp32 dq workspace 中的中间累加结果
```

在 `mha_varlen_bwd.cpp` 中调用位置是：

```cpp
blockMmadFAGCube2(
    cubeAddrInfo[(taskId - 1) % 2],
    (__gm__ ElementA2*)(params.workspace + dsWorkSpaceOffset),
    (__gm__ ElementB2*)(params.k),
    (__gm__ float*)(params.workspace + dqWorkSpaceOffset),
    pingpongFlagL1A,
    pingpongFlagL0A,
    pingpongFlagL1B,
    pingpongFlagL0B);
```

这说明 Cube2 消费的是 Vector 阶段已经写好的上一拍 `dS`，并把结果累加到 `dqWorkSpace`。

## 数学含义

对一个 attention 子块，形状可以理解为：

```text
dS: (Q_block, K_block)
K : (K_block, headDim)
dQ: (Q_block, headDim)
```

所以：

```text
dQ[Q, D] += dS[Q, K] * K[K, D]
```

与 Cube1 不同：

- Cube1 是 `A * B^T`，用于 `Q*K^T` 和 `dOut*V^T`。
- Cube2 是 `A * B`，用于 `dS*K`。
- Cube3 是 `A^T * B`，用于 `dK=dS^T*Q` 和 `dV=P^T*dOut`。

## 核心输入输出

`operator()` 的核心参数是：

```cpp
void operator()(const CubeAddrInfo &addrs,
                __gm__ ElementA *left,
                __gm__ ElementB *right,
                __gm__ float *out,
                uint32_t &pingpongFlagL1A,
                uint32_t &pingpongFlagL0A,
                uint32_t &pingpongFlagL1B,
                uint32_t &pingpongFlagL0B)
```

含义如下：

- `addrs`：Cube 地址生成器输出的块任务。
- `left`：`dS workspace` 基地址。
- `right`：原始 `K` 基地址。
- `out`：`dq workspace` 基地址。
- `pingpongFlagL1A/L1B/L0A/L0B`：L1/L0 多级缓冲的乒乓状态。

`AddrInfo` 中的偏移在 Cube2 中对应：

```text
shapeInfo.out   -> dS workspace 中当前 attention block 的起点
shapeInfo.right -> K tensor 中当前 K block 的起点
shapeInfo.left  -> dq workspace 中当前 Q block 的起点
```

## 整体数据流

```text
Vector epilogue
    │
    │ 写 dS 到 ds workspace
    ▼
Cube2 读取 dS workspace + 原始 K
    │
    ├── dS: GM -> L1A -> L0A
    └── K : GM -> L1B -> L0B
             │
             │ tileMmad
             ▼
          L0C fp32
             │
             │ atomic add
             ▼
        dq workspace
```

更细一点的流水：

```text
当前 K 子块 n_loop_index
    │
    ├─ K(n_remain, D) GM -> L1B -> L0B
    │       └─ L0B 常驻，复用给多个 Q/M 子块
    │
    ├─ dS(M,N) GM -> L1A -> L0A
    │
    ├─ MMAD: dS(M,N) * K(N,D)
    │
    └─ 如果这是最后一个有效 K 子块：
            L0C -> dq workspace，使用 atomic add
```

## K 维累加和 `last_k`

Cube2 的外层循环是 `n_loop_index`，它对应 K/token 方向，也就是矩阵乘归约维：

```cpp
for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++)
```

每个 `n_loop_index` 会贡献一部分：

```text
dQ_partial = dS[:, K_block] * K[K_block, :]
```

这些 partial 需要在 L0C 中累加，直到最后一个有效 K 子块才写回 GM。

代码中：

```cpp
bool l0_c_init_flag = (n_loop_index == 0);
last_k = (m_loop_index == 0 && upperRight)
    ? n_loop_index == n_loop - 2
    : n_loop_index == n_loop - 1;
```

含义是：

- 第一个 K 子块初始化 L0C。
- 中间 K 子块继续累加。
- 最后一个有效 K 子块完成后写回。
- 如果 causal 右上角最后一个 K 子块对某些 M 子块是无效块，那么倒数第二个 K 子块就是最后一次有效累加。

## causal 跳过和 workspace 压缩

Vector 阶段写 `dS workspace` 时，causal 无效块不会写入。Cube2 读取 `dS` 时必须使用相同规则跳过，否则会读错块。

代码中：

```cpp
if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
    skip_num++;
    is_skip = true;
}
```

读取 dS 时：

```cpp
gLeft[(m_loop * n_loop_index + m_loop_index - skip_num) * 128 * 128]
```

`skip_num` 的作用是把逻辑块编号映射到压缩后的 workspace 块编号。

图示：

```text
逻辑 dS blocks

n_loop_index = last
┌───────────────┐
│ skip          │  m_loop_index = 0，causal 右上无效
├───────────────┤
│ valid         │
└───────────────┘

压缩后的 ds workspace
┌───────────────┐
│ valid         │
└───────────────┘
```

因此读取有效块时要减去已经跳过的块数。

## 为什么写 dq workspace 要用 atomic

Cube2 最终写回：

```cpp
AscendC::SetAtomicType<float>();
copyL0CToGm(...);
AscendC::SetAtomicNone();
```

原因是 `dQ` 是对所有 K block 的累加结果：

```text
dQ_i = Σ_j dS_{i,j} * K_j
```

不同 Cube task 或不同 KV block 可能写到同一个 Q/head 的 `dq workspace` 区域，因此需要 fp32 atomic add，避免覆盖已有 partial sum。

最终 `FAGPost` 会再把 fp32 `dq workspace` 缩放并 cast 成最终 `dq` 输出。

## 例子 1：单个 128x128 block

假设：

```text
km = 128
kn = 128
headdim = 128
无 causal 跳过
```

形状是：

```text
dS: 128 x 128
K : 128 x 128
dQ: 128 x 128
```

流程：

```text
K  -> L1B -> L0B
DS -> L1A -> L0A
L0C = dS * K
L0C atomic add -> dq workspace
```

此时：

- `n_loop = 1`
- `m_loop = 1`
- `l0_c_init_flag = true`
- `last_k = true`
- 计算后立即写回 `dq workspace`

## 例子 2：两个 K 子块累加

假设：

```text
km = 128
kn = 256
headdim = 128
无 causal 跳过
```

则：

```text
dS 被切成两个 K 子块：

dS0: 128 x 128
K0 : 128 x 128

dS1: 128 x 128
K1 : 128 x 128
```

计算为：

```text
dQ = dS0 * K0 + dS1 * K1
```

流程图：

```text
n_loop_index = 0
    L0C = dS0 * K0       init_c = true,  last_k = false

n_loop_index = 1
    L0C += dS1 * K1      init_c = false, last_k = true
    L0C atomic add -> dq workspace
```

也就是说，Cube2 不是每处理一个 K 子块就写回，而是等当前 Q/M 子块的 K 维全部累加完成后才写回。

## 与 FAG 反向流水的关系

Cube2 位于 Vector epilogue 之后：

```text
Cube1(Q, K) 和 Cube1(dOut, V)
    ↓
Vector epilogue 重新计算 P，并生成 dS
    ↓
Cube2: dQ = dS * K
    ↓
FAGPost: fp32 dq workspace -> scale/cast -> dq
```

完整反向中的梯度关系可概括为：

```text
dS -> Cube2 -> dQ
dS -> Cube3 -> dK
P  -> Cube3 -> dV
```

## 注意点

- Cube2 输出的是 fp32 `dq workspace`，不是最终 `dq` tensor。
- `dq` 最终还需要在 `FAGPost` 中乘 `softmax_scale` 并 cast 到目标 dtype。
- `dS workspace` 是压缩布局，causal 跳过块不占位置，因此 Cube2 必须维护 `skip_num`。
- `K` 子块在 L0B 中常驻，供多个 M/Q 子块复用。
- `tileMmad` 的 `unit flag` 为 2 时表示继续 K 维累加，为 3 时表示当前 K 维结束并准备输出。
- 写回 `dq workspace` 使用 fp32 atomic add，是为了累加多个 partial result。
