# fag_epilogue_pre.hpp 简要解析

## 文件职责

`csrc/flash_attn_npu/fag_epilogue_pre.hpp` 定义了 FlashAttention 反向传播 FAG 的 Vector 侧预处理模块：

```cpp
BlockEpilogue<
    EpilogueAtlasA2FAGPre,
    ElementVecDtype>
```

它在真正的 Cube/Vector 反向计算前执行，主要负责把 fp32 的 `dq/dk/dv` workspace 清零。

核心职责：

```text
1. 从 tiling_data 读取 dq/dk/dv workspace 偏移
2. 读取 qSize 和 kvSize
3. 根据 coreNum 把 workspace 清零任务切给各个 AIV core
4. 当前 core 清零自己负责的 dq/dk/dv workspace 分片
```

## 在整体反向流程中的位置

`FAGPre` 是 FAG 反向 Vector 分支最先执行的阶段：

```text
FAGPre
  ↓
FAGSfmg
  ↓
FAGOp / Cube 主循环
  ↓
FAGPost
```

对应调用链：

```text
AscendC::TPipe pipePre;
EpilogueFAGPre epilogueFagPre(...);
epilogueFagPre();
pipePre.Destroy();
```

它和 `FAGPost` 首尾对应：

```text
FAGPre:
    fp32 workspace = 0

中间 Cube/Vector:
    fp32 workspace += partial gradients

FAGPost:
    fp32 workspace -> scale/cast -> final dq/dk/dv
```

## 输入和清零对象

FAGPre 清零的是 workspace，不是最终输出 tensor：

```cpp
AscendC::GlobalTensor<float> dqWorkSpaceGm;
AscendC::GlobalTensor<float> dkWorkSpaceGm;
AscendC::GlobalTensor<float> dvWorkSpaceGm;
```

其中：

- `dqWorkSpaceGm` 对应 `qSize`
- `dkWorkSpaceGm` 对应 `kvSize`
- `dvWorkSpaceGm` 对应 `kvSize`

图示：

```text
workspace
  ├── dqWorkSpaceGm: qSize  个 fp32 元素
  ├── dkWorkSpaceGm: kvSize 个 fp32 元素
  └── dvWorkSpaceGm: kvSize 个 fp32 元素
```

## 构造函数做了什么

构造函数主要完成三件事。

### 1. 读取 tiling 数据

从 `tiling_data` 中读取：

```text
dqWorkSpaceOffset
dkWorkSpaceOffset
dvWorkSpaceOffset
qSize
kvSize
coreNum
```

### 2. 按 core 切分清零任务

对于 dq：

```cpp
qPreBlockFactor = ceil(qSize / coreNum)
qPreBlockTotal = ceil(qSize / qPreBlockFactor)
qPreBlockTail = qSize % qPreBlockFactor or qPreBlockFactor
```

对于 dk/dv：

```cpp
kvPreBlockFactor = ceil(kvSize / coreNum)
kvPreBlockTotal = ceil(kvSize / kvPreBlockFactor)
kvPreBlockTail = kvSize % kvPreBlockFactor or kvPreBlockFactor
```

### 3. 计算当前 core 的 offset 和 size

```cpp
dqOffset = cBlockIdx * qPreBlockFactor
initdqSize = 当前 core 需要清零的 dq 元素数

dkvOffset = cBlockIdx * kvPreBlockFactor
initdkSize = 当前 core 需要清零的 dk/dv 元素数
```

## operator() 主流程

主入口：

```cpp
void operator()()
```

逻辑很简单：

```text
if 当前是 AIV core 且 cBlockIdx < qPreBlockTotal:
    清零 dq workspace 当前分片

if 当前是 AIV core 且 cBlockIdx < kvPreBlockTotal:
    清零 dk workspace 当前分片
    清零 dv workspace 当前分片
```

流程图：

```text
operator()
  │
  ├── 是否 AIV core？
  │      │
  │      ├── 否：不做清零
  │      │
  │      └── 是
  │           │
  │           ├── cBlockIdx < qPreBlockTotal ?
  │           │       └── InitOutput(dqWorkSpaceGm[dqOffset], initdqSize, 0)
  │           │
  │           └── cBlockIdx < kvPreBlockTotal ?
  │                   ├── InitOutput(dkWorkSpaceGm[dkvOffset], initdkSize, 0)
  │                   └── InitOutput(dvWorkSpaceGm[dkvOffset], initdkSize, 0)
```

## 示例一：qSize 可以均匀分给所有 core

假设：

```text
qSize = 1024
coreNum = 4
```

则：

```text
qPreBlockFactor = ceil(1024 / 4) = 256
qPreBlockTotal = ceil(1024 / 256) = 4
qPreBlockTail = 256
```

每个 core 的清零范围：

```text
core0: offset = 0,   size = 256
core1: offset = 256, size = 256
core2: offset = 512, size = 256
core3: offset = 768, size = 256
```

图：

```text
dq workspace, qSize = 1024

0        256       512       768      1024
|---------|---------|---------|---------|
| core0   | core1   | core2   | core3   |
```

## 示例二：qSize 不能均匀分配

假设：

```text
qSize = 1000
coreNum = 4
```

则：

```text
qPreBlockFactor = ceil(1000 / 4) = 250
qPreBlockTotal = ceil(1000 / 250) = 4
qPreBlockTail = 250
```

这个例子仍然刚好均匀。换一个更能体现尾块的例子：

```text
qSize = 1001
coreNum = 4
```

则：

```text
qPreBlockFactor = ceil(1001 / 4) = 251
qPreBlockTotal = ceil(1001 / 251) = 4
qPreBlockTail = 1001 % 251 = 248
```

每个 core 的范围：

```text
core0: offset = 0,   size = 251
core1: offset = 251, size = 251
core2: offset = 502, size = 251
core3: offset = 753, size = 248
```

图：

```text
dq workspace, qSize = 1001

0        251       502       753       1001
|---------|---------|---------|----------|
| core0   | core1   | core2   | core3    |
| 251     | 251     | 251     | 248 tail |
```

## dk/dv 的切分

`dk` 和 `dv` 共享 `kvPre*` 参数，因为它们的展平元素数都来自 `kvSize`：

```text
dk workspace: kvSize 个 fp32 元素
dv workspace: kvSize 个 fp32 元素
```

同一个 core 会清零相同 offset 和 size 的 dk/dv 分片：

```text
core i:
    dk[dkvOffset : dkvOffset + initdkSize] = 0
    dv[dkvOffset : dkvOffset + initdkSize] = 0
```

图：

```text
kv workspace range for core i

        dkvOffset
            ↓
dk: |---- previous ----|---- clear by core i ----|---- next ----|
dv: |---- previous ----|---- clear by core i ----|---- next ----|
```

## 为什么要先清零 workspace？

反向传播中，`dq/dk/dv` 会由多个 block、多个 head、多个 sequence tile 贡献部分结果。

中间阶段不是直接写最终结果，而是在 fp32 workspace 中累加：

```text
partial gradient 1
partial gradient 2
partial gradient 3
...
        ↓
fp32 workspace 累加
```

因此计算开始前必须先清零：

```text
不清零：workspace 可能包含旧值，累加结果错误
清零：workspace 从 0 开始累加，结果正确
```

## 与 FAGPost 的关系

```text
FAGPre:
    清零 dq/dk/dv fp32 workspace

FAGPost:
    读取 dq/dk/dv fp32 workspace
    dq/dk 乘 softmax_scale
    cast 到输出 dtype
    写回最终 dq/dk/dv
```

首尾关系图：

```text
FAGPre
  │
  ├── dqWorkSpace = 0
  ├── dkWorkSpace = 0
  └── dvWorkSpace = 0
        ↓
中间计算累加 partial gradients
        ↓
FAGPost
  │
  ├── dqWorkSpace -> dq
  ├── dkWorkSpace -> dk
  └── dvWorkSpace -> dv
```

## 注意点

- `FAGPre` 只在 `AIV` core 上执行清零。
- `dq` 使用 `qSize` 切分。
- `dk/dv` 使用 `kvSize` 切分。
- `dk` 和 `dv` 清零范围完全一致。
- 清零对象是 fp32 workspace，而不是最终输出 tensor。
- 该阶段没有复杂向量计算，主要是 `InitOutput<float>(..., 0)`。
