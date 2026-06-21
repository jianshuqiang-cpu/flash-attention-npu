# fag_mmad_cube3.hpp 解析

## 1. 文件定位

`fag_mmad_cube3.hpp` 实现 FAG backward 中第三类 Cube MMAD 计算，位于：

```text
csrc/flash_attn_npu/fag_mmad_cube3.hpp
```

它在主 backward kernel `mha_varlen_bwd.cpp` 中被调用两次：

```cpp
blockMmadFAGCube3(..., params.workspace + pWorkSpaceOffset, params.dout, params.workspace + dvWorkSpaceOffset, ...);
blockMmadFAGCube3(..., params.workspace + dsWorkSpaceOffset, params.q, params.workspace + dkWorkSpaceOffset, ...);
```

对应两条公式：

```text
dV = P^T  * dOut
dK = dS^T * Q
```

所以 Cube3 是 FAG backward 中生成 `dV` 和 `dK` 的 Cube 侧矩阵乘模块。

## 2. 和 Cube2 的区别

Cube2 计算：

```text
dQ = dS * K
```

Cube3 计算：

```text
dV = P^T  * dOut
dK = dS^T * Q
```

关键区别是左矩阵是否转置：

```text
Cube2: A 不转置，B 不转置
       dS(ky, kx) * K(kx, d) -> dQ(ky, d)

Cube3: A 转置，B 不转置
       P^T / dS^T(kx, ky) * dOut / Q(ky, d) -> dV / dK(kx, d)
```

其中：

- `ky` 是 Q 方向 token 数。
- `kx` 是 K/V 方向 token 数。
- `d` 是 `headdim`。

## 3. 输入输出映射

Cube3 的 `operator()` 签名为：

```cpp
void operator()(const CubeAddrInfo &addrs,
                __gm__ ElementA *left,
                __gm__ ElementB *right,
                __gm__ float *out,
                ...)
```

两次调用时参数含义不同：

| 公式 | left | right | out |
| --- | --- | --- | --- |
| `dV = P^T * dOut` | `P workspace` | `dOut` | `dv workspace` |
| `dK = dS^T * Q` | `dS workspace` | `Q` | `dk workspace` |

在代码中地址映射为：

```cpp
__gm__ ElementA* gm_a = left + (shapeInfo.out + globalBlockOffset);
__gm__ ElementB* gm_b = right + shapeInfo.left;
__gm__ float* gm_out = out + shapeInfo.right;
```

含义是：

```text
shapeInfo.out   -> P/dS workspace 中当前 attention block 的偏移
shapeInfo.left  -> Q/dOut 的输入偏移
shapeInfo.right -> dK/dV workspace 的输出偏移
```

注意 `gm_out` 写到 `shapeInfo.right`，因为 `dK/dV` 都属于 KV 方向梯度。

## 4. 数据流图

```text
Vector epilogue
    │
    ├── 写 P 到 p workspace
    └── 写 dS 到 ds workspace
            │
            ▼
Cube3 读取 P/dS workspace + dOut/Q
            │
            ├── A: P/dS workspace -> L1A -> L0A
            └── B: dOut/Q         -> L1B -> L0B
                         │
                         │ tileMmad
                         ▼
                      L0C fp32
                         │
                         │ atomic add
                         ▼
                 dv/dk fp32 workspace
```

最终 `FAGPost` 会从 fp32 workspace 中读取 `dk/dv`，转换成最终输出 dtype。

## 5. 核心循环结构

Cube3 的主要循环是：

```text
for each AddrInfo in CubeAddrInfo:
    读取 P/dS workspace 基址
    读取 Q/dOut 基址
    读取 dk/dv workspace 基址

    for n_loop_index in K/V token 方向:
        for m_loop_index in Q token 归约方向:
            搬运 P/dS 子块到 L1A/L0A
            复用 Q/dOut 子块到 L0B
            tileMmad 累加到 L0C
        写回当前 K/V token 子块的 dK/dV
```

数学上可以理解为：

```text
对每个 K/V token 块 n：
    dK_or_dV[n, :] = sum_over_q( left[q, n] * right[q, :] )
```

即沿 Q 方向 `ky` 做归约。

## 6. 为什么 right 只搬一次

代码中 `Q/dOut` 对同一个 `CubeAddrInfo` block 是公共右矩阵：

```cpp
copyGmToL1B(*l1_b_buf_tensor, gRight, layoutBInL1, layoutTileB);
```

随后在 `n_loop_index == 0` 时，将 B 从 L1B 搬到 L0B：

```cpp
if (n_loop_index == 0) {
    copyL1ToL0B(...);
}
```

原因是 Cube3 对多个 K/V token 输出块计算时，右矩阵 `Q/dOut(ky, headdim)` 不随 `n_loop_index` 变化，可以复用，减少 GM 和 L1 搬运。

## 7. causal skip 与 workspace 压缩

在 causal mask 下，一些 attention block 是无效的，不需要参与 `P`、`dS` 或后续梯度计算。

Cube3 会跳过两类边界：

```cpp
if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
    is_skip = true;
}

if (n_loop_index == 0 && m_loop_index == m_loop - 1 && lowerLeft) {
    is_skip = true;
}
```

其中：

- `upperRight` 对应右上角 causal 无效块。
- `lowerLeft` 对应左下角边界处理。
- `skip_num` 记录已经被跳过的无效块数量。

因为 `P/dS workspace` 只连续保存有效块，所以读取偏移必须减去 `skip_num`：

```cpp
gLeft[(n_loop_index * m_loop + m_loop_index - skip_num) * 128 * 128]
```

## 8. L0C 初始化、累加与输出

Cube3 对固定的 `n_loop_index`，沿 `m_loop_index` 累加 Q 方向：

```cpp
bool init_c = (m_loop_index == 0);
bool out_c = (m_loop_index == (m_loop - 1));
```

含义是：

```text
m_loop_index == 0           -> 初始化 L0C
0 < m_loop_index < last     -> 继续累加 L0C
m_loop_index == last        -> 完成归约并写回
```

MMAD 调用：

```cpp
tileMmad(*l0_c_buf_tensor,
         *l0_a_buf_tensor,
         *l0_b_buf_tensor,
         real_n,
         l1_k_size,
         real_m,
         init_c,
         unit_flag);
```

对应公式：

```text
P^T/dS^T(real_n, real_m) x dOut/Q(real_m, headdim)
    -> dV/dK(real_n, headdim)
```

其中：

- `real_n` 是当前 K/V token 子块行数。
- `real_m` 是当前 Q token 归约子块长度。
- `l1_k_size` 是 `headdim`。

## 9. 为什么写回使用 atomic add

写回时：

```cpp
AscendC::SetAtomicType<float>();
copyL0CToGm(...);
AscendC::SetAtomicNone();
```

原因是不同 query block 或不同调度任务可能对同一个 `dK/dV` 行产生贡献：

```text
dK[k, :] = sum_over_q dS[q, k] * Q[q, :]
dV[k, :] = sum_over_q P[q, k]  * dOut[q, :]
```

这些贡献需要累加到同一个 fp32 workspace，因此使用 fp32 atomic add。

## 10. 示例一：单个 128x128 attention block

假设：

```text
ky = 128
kx = 128
headdim = 64
```

计算 `dV` 时：

```text
P       : (128, 128)
dOut    : (128, 64)
P^T     : (128, 128)
dV      : (128, 64)
```

流程：

```text
P workspace(128x128) -> L1A -> L0A
          dOut(128x64) -> L1B -> L0B
                          │
                          ▼
          MMAD: P^T(128x128) * dOut(128x64)
                          │
                          ▼
                   dV workspace(128x64)
```

因为只有一个 `m_loop_index`，L0C 初始化、累加、输出都发生在同一轮。

## 11. 示例二：Q 方向分成两个 128 块

假设：

```text
ky = 256
kx = 128
headdim = 64
```

计算 `dK` 时：

```text
dS      : (256, 128)
Q       : (256, 64)
dS^T    : (128, 256)
dK      : (128, 64)
```

Cube3 会沿 Q 方向分两段累加：

```text
第 1 段 Q token:
    dS^T[0:128, 0:128] * Q[0:128, :]
        -> 初始化 L0C 并写入部分和

第 2 段 Q token:
    dS^T[0:128, 128:256] * Q[128:256, :]
        -> 继续累加 L0C
        -> 写回 dK[0:128, :]
```

图示：

```text
                  Q / dOut
             ┌───────────────┐
             │ Q block 0     │
             ├───────────────┤
             │ Q block 1     │
             └───────────────┘
                    ▲
                    │ 沿 Q 方向归约
P^T / dS^T ─────────┘
                    │
                    ▼
              dK / dV rows
```

## 12. 总结

`fag_mmad_cube3.hpp` 是 FAG backward 中负责 `dK/dV` 的 Cube MMAD 模块：

- 对 `P workspace + dOut` 计算 `dV = P^T * dOut`。
- 对 `dS workspace + Q` 计算 `dK = dS^T * Q`。
- 读取 Vector epilogue 写出的 `P/dS workspace`。
- 复用 `Q/dOut` 右矩阵，减少搬运。
- 沿 Q 方向做归约，输出 K/V 方向梯度。
- 对 causal 无效块进行 skip，并通过 `skip_num` 适配压缩 workspace。
- 使用 fp32 atomic add 写回 `dk/dv workspace`。
- 后续由 `FAGPost` 将 fp32 workspace 转成最终输出。
