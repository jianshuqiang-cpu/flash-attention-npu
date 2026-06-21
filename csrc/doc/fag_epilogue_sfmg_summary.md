# fag_epilogue_sfmg.hpp 简要解析

## 文件职责

`csrc/flash_attn_npu/fag_epilogue_sfmg.hpp` 定义了 FlashAttention 反向传播 FAG 的 softmax gradient 前置归约模块：

```cpp
BlockEpilogue<
    EpilogueAtlasA2FAGSfmg,
    ElementVecDtype,
    std::integral_constant<InputLayout, inputLayout>>
```

它在 `FAGPre` 清零 workspace 后、`FAGOp` 计算 `dS` 前执行。

核心任务：

```text
读取 dout 和前向输出 out
    ↓
逐行计算 rowsum(dout * out)
    ↓
写入 sfmg workspace
```

后续 `FAGOp` 会读取该辅助项，用于计算 softmax 反向：

```text
dS = P * (dP - rowsum(dP * P))
```

在本实现中，`FAGSfmg` 计算的是每个 token/head 行上的 `dout * out` 归约项，写入 `sfmgWorkspaceGm`。

## 在整体反向流程中的位置

调用链位于 `mha_varlen_bwd.cpp`：

```text
FAGPre
  ↓
FAGSfmg
  ↓
FAGOp
  ↓
FAGPost
```

对应关系：

```text
FAGPre:
    清零 dq/dk/dv workspace

FAGSfmg:
    计算 rowsum(dout * out)，写入 sfmg workspace

FAGOp:
    读取 sfmg workspace，计算 dS

FAGPost:
    输出最终 dq/dk/dv
```

## 输入输出

### 输入

```cpp
GlobalTensor<ElementVecDtype> doutGm;
GlobalTensor<ElementVecDtype> outGm;
```

含义：

- `doutGm`：上游传回的 attention 输出梯度
- `outGm`：前向 attention 输出

### 输出

```cpp
GlobalTensor<float> sfmgWorkspaceGm;
```

每个 token/head 行输出 `8` 个 float：

```text
BLOCK_SIZE = 8
BLOCK_BYTE_SIZE = 32
```

图示：

```text
一行 dout: [d0, d1, ..., dD]
一行 out : [o0, o1, ..., oD]

rowsum(dout * out)
    = d0*o0 + d1*o1 + ... + dD*oD

输出到 sfmg workspace，每行占 8 个 float。
```

## 构造函数做了什么

构造函数主要完成：

### 1. 读取 tiling 数据

从 `tiling_data` 中读取：

```text
batch
total_q
nheads_k
g
headdim
sfmgWorkspaceOffset
coreNum
```

并计算：

```text
nheads = nheads_k * g
dAlign = align16(headdim)
```

### 2. 处理输入布局

支持两种布局：

```cpp
InputLayout::BSND
InputLayout::TND
```

BSND 是定长 batch：

```text
seq_q = total_q / batch
```

TND 是变长打平布局：

```text
seq_q = 0
真实长度通过 cu_seq_qlen_addr 动态计算
```

### 3. 计算每个 core 的任务量

任务轴是：

```text
normalAxisSize = total_q * nheads
```

这里每个单位表示一个 token/head 行。

每个 core 处理：

```text
normalCoreSize = ceil(normalAxisSize / coreNum)
```

实际使用的 core 数：

```text
usedCoreNum = ceil(normalAxisSize / normalCoreSize)
```

### 4. 计算每个 loop 处理多少行

每行需要搬运 `dAlign` 个元素：

```text
singleLoopNBurstNum = inputBufferLen / sizeof(float) / dAlign
```

也就是说，`headdim` 越大，单 loop 能处理的 token/head 行数越少。

## 核心函数

### InitIndex

```cpp
void InitIndex(int64_t startIdx, int64_t& curS, GM_ADDR seqS)
```

作用：把展平下标恢复为三维索引：

```text
batch index: bIdx
head index : nIdx
seq index  : sIdx
```

BSND 下：

```text
startIdx -> bIdx / nIdx / sIdx
```

TND 下：

```text
通过 cu_seq_qlen_addr 找到 startIdx 属于哪个 batch
```

### DoCopyIn

```cpp
void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS)
```

作用：从 GM 中搬入 `curNBurst` 个 token/head 行的：

```text
dout
out
```

搬入 UB 时，每行从 `headdim` pad 到 `dAlign`。

### CopyInSfmg

```cpp
void CopyInSfmg(int64_t leftNburst, int64_t &curS, GM_ADDR seqS)
```

作用：连续搬入 `leftNburst` 行。

如果当前 sequence 剩余行数不够，会自动跨：

```text
seq -> head -> batch
```

图示：

```text
当前 head 的 seq 不够
        ↓
切到下一个 head
        ↓
head 也用完
        ↓
切到下一个 batch
```

## operator() 主流程

主入口：

```cpp
void operator()()
```

流程：

```text
1. 判断当前 core 是否参与计算
2. 计算当前 core 的 loop 次数
3. 每个 loop 搬入 dout/out
4. cast 到 float
5. 调用 SoftmaxGradFront 计算 rowsum(dout * out)
6. 将每行 8 个 float 写入 sfmg workspace
```

流程图：

```text
operator()
  │
  ├── 当前 core 是否 < usedCoreNum？
  │      └── 否：空闲
  │
  └── 是
       │
       ├── 根据 cBlockIdx 计算 startIdx
       │
       └── for each loop
             │
             ├── CopyInSfmg: dout/out GM -> UB
             ├── Cast dout -> float
             ├── Cast out  -> float
             ├── SoftmaxGradFront
             │      └── rowsum(dout * out)
             └── DataCopy -> sfmgWorkspaceGm
```

## SoftmaxGradFront 做了什么

调用：

```cpp
SoftmaxGradFront<float, true/false>(outputBuf, sfmgClc1, sfmgClc2, tempBuf);
```

其中：

```text
sfmgClc1 = dout, float
sfmgClc2 = out,  float
outputBuf = 每行归约结果
```

核心计算：

```text
output[row] = sum_d(dout[row, d] * out[row, d])
```

高性能路径条件：

```text
nBurst % 8 == 0 && dAlign % 64 == 0
```

满足条件时走 `SoftmaxGradFront<float, true>`，否则走通用路径。

## 示例一：BSND 定长布局

假设：

```text
batch = 2
seq_q = 4
nheads = 2
headdim = 64
```

则：

```text
total_q = batch * seq_q = 8
normalAxisSize = total_q * nheads = 16
```

也就是一共有 16 个 token/head 行：

```text
B0,H0,S0
B0,H0,S1
B0,H0,S2
B0,H0,S3
B0,H1,S0
...
B1,H1,S3
```

每一行计算：

```text
sum over headdim: dout[row, :] * out[row, :]
```

图：

```text
row = (batch, head, seq)

       headdim
        ────────────────→
dout: [d0 d1 d2 ... d63]
out : [o0 o1 o2 ... o63]
        │  │  │      │
        ▼  ▼  ▼      ▼
      d0o0+d1o1+...+d63o63
        ↓
sfmgWorkspace[row] = 8 floats
```

## 示例二：跨 head 搬运

假设当前需要搬入：

```text
leftNburst = 5 行
当前 batch = 0
当前 head = 0
当前 sIdx = 2
curS = 4
nheads = 2
```

当前 head 剩余 sequence 行数：

```text
curS - sIdx = 4 - 2 = 2
```

所以 `CopyInSfmg` 会：

```text
先搬 B0,H0,S2 ~ S3，共 2 行
剩余 3 行
切到 H1,S0
再搬 B0,H1,S0 ~ S2，共 3 行
```

图：

```text
需要 5 行

B0,H0: S0 S1 [S2 S3]
              └─ 2 行
B0,H1: [S0 S1 S2] S3
        └─ 3 行
```

如果 head 也不够，就继续切到下一个 batch。

## 和 FAGOp 的关系

`FAGSfmg` 写出的 `sfmgWorkspaceGm` 会被 `FAGOp` 读取：

```text
FAGSfmg:
    sfmgWorkspace[row] = rowsum(dout * out)

FAGOp/SubGrapB:
    dS = P * (dP - sfmgWorkspace[row])
```

关系图：

```text
dout + out
    │
    ▼
FAGSfmg
    │ rowsum(dout * out)
    ▼
sfmgWorkspace
    │
    ▼
FAGOp SubGrapB
    │
    ▼
dS workspace
```

## 注意点

- 任务轴是 `total_q * nheads`，不是 `total_q * nheads * headdim`。
- 每个任务单位是一行 token/head，行内长度是 `headdim`，搬运时对齐到 `dAlign`。
- `sfmg workspace` 每行占 `8` 个 float。
- BSND 直接用固定 `seq_q`；TND 需要通过 `cu_seq_qlen_addr` 定位 batch 长度。
- 高性能路径要求 `nBurst` 是 8 的倍数，并且 `dAlign` 是 64 的倍数。
