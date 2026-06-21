# fag_tiling.cpp 解析

## 1. 文件定位

`fag_tiling.cpp` 是 FAG backward 的 Host 侧 tiling 生成文件。它不直接执行 NPU kernel 计算，而是在 kernel 启动前根据输入 shape、head 数、headdim 和硬件 core 数生成一块 `tiling_data`。

这块 `tiling_data` 后续会被拷贝到 NPU 侧，供 FAG kernel 的多个阶段读取：

```text
flash_api.cpp
    │
    │ 解析 q/k/v/dout shape
    ▼
FAGTiling::GetFATilingParam
    │
    │ 生成 tiling_cpu_tensor
    ▼
tiling_gpu_tensor
    │
    ▼
FAG kernel
    │
    ├── FAGPre
    ├── FAGSfmg
    ├── Cube1 / Cube2 / Cube3
    ├── Vector FAGOp
    └── FAGPost
```

它主要负责四类信息：

1. 基础 shape 和 GQA/MQA 分组信息；
2. softmax / softmaxGrad 的向量 tiling 参数；
3. NPU core 数；
4. FAG backward workspace 内部各区域的 byte offset。

---

## 2. 相关文件关系

### 2.1 Host 侧调用

`flash_api.cpp` 中会构造 `FAGInfo` 并调用：

```cpp
FAGTiling::GetFATilingParam(
    fagInfo,
    blockDim,
    reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>())
);
```

调用之后：

```text
tiling_cpu_tensor -> tiling_gpu_tensor -> FAG kernel 参数 tiling_data
```

### 2.2 Kernel 侧读取

kernel 侧通过 `common_header.h` 中的 `TILING_*` 常量读取对应字段。

例如：

```cpp
int64_t headdim = tilingData.GetValue(TILING_D);
int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
```

softmax tiling 因为字段是 `uint32_t`，所以 kernel 侧会按 `uint32_t` 视角读取：

```cpp
softmaxTilingData.srcM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2);
```

---

## 3. `FAGInfo` 结构

```cpp
struct FAGInfo {
    float scaleValue;

    int64_t seqQShapeSize;
    int64_t queryShape_0;
    int64_t queryShape_1;
    int64_t queryShape_2;
    int64_t keyShape_0;
    int64_t keyShape_1;
    int64_t valueShape_0;
    int64_t valueShape_1;
};
```

字段含义：

| 字段 | 含义 |
|---|---|
| `scaleValue` | softmax scale，通常是 `1 / sqrt(headdim)` |
| `seqQShapeSize` | batch 数或 varlen 场景下的序列段数 |
| `queryShape_0` | 展平后的 Q token 总数 |
| `queryShape_1` | Q head 数，也就是 `nheads` |
| `queryShape_2` | head dimension，也就是 `headdim` |
| `keyShape_0` | 展平后的 K/V token 总数 |
| `keyShape_1` | KV head 数，也就是 `nheads_k` |
| `valueShape_0/valueShape_1` | 当前文件中未实际使用 |

对于 GQA/MQA：

```text
g = nheads / nheads_k
```

即每个 KV head 对应多少个 query head。

---

## 4. `printFAGTilingData`

这是调试辅助函数，用于把 tiling_data 中的内容打印出来。

比较特殊的是，同一块 `tilingHost` 内存会被三种方式解释：

```text
int64_t*   -> 读取 shape / workspace offset
float*     -> 读取 scaleValue
uint32_t*  -> 读取 coreNum / SoftMaxTiling 字段
```

因为一个 `int64_t` 等于两个 `uint32_t`，所以读取 `uint32_t` 字段时需要：

```text
TILING_xxx * CONST_2
```

例如：

```cpp
tilingHostU32[TILING_CORE_NUM * CONST_2]
```

---

## 5. `GetFATilingParam` 总览

`GetFATilingParam` 是本文件最核心的函数。

它可以分成以下步骤：

```text
1. 写入 scaleValue
2. 写入 batch / total_q / total_k / nheads / nheads_k / headdim
3. 计算 GQA 分组 g
4. 计算 qSize / kvSize / sfmgSize
5. 生成 softmax tiling
6. 生成 SoftmaxGradFront tiling
7. 写入 AIV core 数和 tiling 结构体
8. 依次计算 workspace offset
```

流程图：

```text
FAGInfo
  │
  ├── shape 信息
  │      ├── total_q
  │      ├── total_k
  │      ├── nheads
  │      ├── nheads_k
  │      └── headdim
  │
  ├── scaleValue
  │
  ├── softmax tiling
  │
  ├── softmaxGrad tiling
  │
  └── workspace offset
          │
          ├── dq workspace
          ├── dk workspace
          ├── dv workspace
          ├── sfmg workspace
          ├── mm1 workspace
          ├── mm2 workspace
          ├── p workspace
          └── ds workspace
```

---

## 6. Shape 和 size 计算

### 6.1 GQA/MQA 分组

```cpp
uint64_t g = fagInfo.queryShape_1 / fagInfo.keyShape_1;
```

也就是：

```text
g = nheads / nheads_k
```

如果：

```text
nheads = 16
nheads_k = 4
```

那么：

```text
g = 4
```

表示每个 KV head 服务 4 个 query head。

### 6.2 qSize

```cpp
qSize = total_q * nheads_k * g * headdim
```

因为：

```text
nheads_k * g = nheads
```

所以：

```text
qSize = total_q * nheads * headdim
```

它对应 `dq workspace` 的 fp32 元素数。

### 6.3 kvSize

```cpp
kvSize = total_k * nheads_k * headdim
```

它对应 `dk workspace` 和 `dv workspace` 的 fp32 元素数。

### 6.4 sfmgSize

```cpp
sfmgSize = total_q * nheads * 8
```

`sfmg workspace` 中每个 Q token/head 行保存 8 个 float，用来存放 32B 对齐后的：

```text
sum(dout * out)
```

---

## 7. softmax tiling

```cpp
constexpr uint32_t tmpBufferSize = 33 * 1024;
constexpr uint32_t s1VecSize = 64;
constexpr uint32_t s2VecSize = 128;
std::vector<uint32_t> softmaxShape = {s1VecSize, s2VecSize};

SoftMaxTilingFunc(softmaxShape, sizeof(float), tmpBufferSize, softmaxTilingData);
```

这部分为 Vector epilogue 中的 softmax 重算生成 tiling 参数。

逻辑上可以理解为以固定的：

```text
64 x 128
```

作为基础块形状估算 UB 使用量，实际 kernel 侧再结合真实有效长度、causal mask 和边界处理。

生成的 `SoftMaxTiling` 会被写入：

```text
TILING_SOFTMAX_TILING_DATA
```

后续由 `fag_epilogue_op.hpp` 读取。

---

## 8. softmaxGrad tiling

这部分服务于 `FAGSfmg` 阶段，用于计算：

```text
sum(dout * out)
```

关键代码：

```cpp
constexpr uint32_t inputBufferLen = 24 * 1024;
constexpr uint32_t castBufferLen = 48 * 1024;
uint32_t outputBufferLen = (castBufferLen + headdim - 1) / headdim * 8;
uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / headdim;
std::vector<int64_t> softmaxGradShape = {singleLoopNBurstNum, headdim};
```

其中：

```text
singleLoopNBurstNum = 每轮最多处理多少个 token/head 行
```

例如 `headdim = 128`：

```text
singleLoopNBurstNum = 24KB / 4 / 128
                    = 24576 / 4 / 128
                    = 48
```

也就是 sfmg 每轮最多处理 48 行。

生成的 `SoftMaxTiling` 会被写入：

```text
TILING_SOFTMAX_GRAD_TILING_DATA
```

后续由 `fag_epilogue_sfmg.hpp` 或相关 sfmg 逻辑读取。

---

## 9. coreNum 和 vectorCoreNum

```cpp
uint32_t coreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAic();
uint32_t vectorCoreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
```

两者用途不同：

| 变量 | 来源 | 主要用途 |
|---|---|---|
| `coreNum` | AIC 数 | 计算 Cube 侧 per-AIC workspace 大小 |
| `vectorCoreNum` | AIV 数 | 写入 tiling，供 Vector/Pre/Post 按 vector core 数切分任务 |

写入 tiling 的是：

```cpp
tilingHostU32[TILING_CORE_NUM * CONST_2] = vectorCoreNum;
```

---

## 10. workspace 布局

workspace 从 16MB 预留区之后开始分配：

```cpp
constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
constexpr size_t GM_ALIGN = 512;
```

每段都按 512B 对齐。

整体布局：

```text
workspace base
  │
  ├── [0, 16MB) runtime/system reserved
  │
  ├── dq fp32 workspace
  │      size = qSize * sizeof(float)
  │
  ├── dk fp32 workspace
  │      size = kvSize * sizeof(float)
  │
  ├── dv fp32 workspace
  │      size = kvSize * sizeof(float)
  │
  ├── sfmg workspace
  │      size = total_q * nheads * 8 * sizeof(float)
  │
  ├── mm1 workspace
  │      size = coreNum * 16 * 128 * 128 * sizeof(float) * 2
  │      meaning = dOut * V^T
  │
  ├── mm2 workspace
  │      size = coreNum * 16 * 128 * 128 * sizeof(float) * 2
  │      meaning = Q * K^T
  │
  ├── p workspace
  │      size = coreNum * 16 * 128 * 128 * sizeof(half) * 2
  │      meaning = P
  │
  └── ds workspace
         size = coreNum * 16 * 128 * 128 * sizeof(half) * 2
         meaning = dS
```

其中：

```text
matmulSize = 16 * 128 * 128
DB_NUM     = 2
```

`DB_NUM=2` 表示 ping-pong 双缓冲。

---

## 11. workspace 与 FAG backward 阶段的关系

```text
FAGPre
  │
  └── 清零 dq/dk/dv fp32 workspace

FAGSfmg
  │
  └── 写 sfmg workspace = sum(dout * out)

Cube1
  │
  ├── 写 mm2 workspace = Q * K^T
  └── 写 mm1 workspace = dOut * V^T

Vector FAGOp
  │
  ├── 读 mm1/mm2/sfmg/LSE
  ├── 写 p workspace
  └── 写 ds workspace

Cube2 / Cube3
  │
  ├── 读 ds workspace，写 dq/dk workspace
  └── 读 p workspace，写 dv workspace

FAGPost
  │
  └── 读 dq/dk/dv fp32 workspace，cast/scale 后写最终 dq/dk/dv
```

---

## 12. 例子一：普通 MHA，无 GQA

假设：

```text
total_q  = 1024
total_k  = 1024
nheads   = 8
nheads_k = 8
headdim  = 128
```

那么：

```text
g = nheads / nheads_k = 1
```

```text
qSize  = 1024 * 8 * 128 = 1,048,576 float
kvSize = 1024 * 8 * 128 = 1,048,576 float
sfmgSize = 1024 * 8 * 8 = 65,536 float
```

含义：

```text
dq workspace: 1,048,576 个 fp32 元素
dk workspace: 1,048,576 个 fp32 元素
dv workspace: 1,048,576 个 fp32 元素
sfmg workspace: 每个 Q token/head 一行，每行 8 个 float
```

---

## 13. 例子二：GQA 场景

假设：

```text
total_q  = 2048
total_k  = 2048
nheads   = 16
nheads_k = 4
headdim  = 128
```

那么：

```text
g = 16 / 4 = 4
```

```text
qSize  = total_q * nheads_k * g * headdim
       = 2048 * 4 * 4 * 128
       = 2048 * 16 * 128
       = 4,194,304 float

kvSize = total_k * nheads_k * headdim
       = 2048 * 4 * 128
       = 1,048,576 float
```

这个例子说明：

```text
Q/DQ 按 query heads 计数，所以有 16 个 head；
K/V/DK/DV 按 KV heads 计数，所以只有 4 个 head；
每个 KV head 对应 4 个 query head。
```

---

## 14. 一个需要注意的实现细节

`common_header.h` 中有如下定义：

```cpp
const int32_t TILING_N1 = 8;
const int32_t TILING_N2 = 8;
```

也就是说 `TILING_N1` 和 `TILING_N2` 使用了同一个 int64 slot。

在 `fag_tiling.cpp` 中：

```cpp
tilingHost[TILING_N1] = nheads;
tilingHost[TILING_N2] = nheads_k;
```

后一次写入会覆盖前一次。因此 kernel 侧实际主要读取：

```cpp
nheads_k = tilingData.GetValue(TILING_N2);
g = tilingData.GetValue(TILING_G);
nheads = nheads_k * g;
```

也就是说，真正传给 kernel 的 Q head 数是通过：

```text
nheads = nheads_k * g
```

重新推导出来的。

这不是本次修改引入的问题，只是理解 tiling_data 布局时需要特别注意。

---

## 15. 总结

`fag_tiling.cpp` 是 FAG backward 的 Host 侧参数规划器。它把 Python/C++ 前端看到的 shape 信息转换成 NPU kernel 能快速读取的 `tiling_data`。

它的核心职责是：

```text
shape -> tiling_data -> workspace layout -> kernel execution plan
```

最关键的输出包括：

- `scaleValue`；
- `batch / total_q / total_k / nheads_k / g / headdim`；
- `qSize / kvSize`；
- softmax tiling；
- softmaxGrad tiling；
- AIV core 数；
- `dq/dk/dv/sfmg/mm1/mm2/p/ds` workspace offset。

这些字段共同支撑 FAG backward 的完整流水：

```text
Pre 清零 -> sfmg 归约 -> Cube1 -> Vector epilogue -> Cube2/Cube3 -> Post 输出
```
