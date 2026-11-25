# 67_ascend950_batched_matmul

## 代码组织

```text
├── 67_ascend950_batched_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── batched_matmul.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 67_ascend950_batched_matmul -DCATLASS_ARCH=3510
cd output/bin
# 基本用法：可执行文件名 batch轴|m轴|n轴|k轴|Device ID
# Device ID 可选，默认为0
./67_ascend950_batched_matmul 5 256 512 1024 0
```

## 使用说明

BatchedMatmul当前支持的`DispatchPolicy`有`MmadPingpong`和`MmadMultiBatch`两种。

### MmadPingpong

`DispatchPolicy MmadPingpong`支持以下几个模板参数：

| 模板参数         | 默认值 | 参数说明                                         |
| ---------------- | ------ | ------------------------------------------------ |
| ArchTag          | 无     | 指定架构型号                                     |
| enableUnitFlag   | false  | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
| useHF32          | false  | 是否开启HF32，仅float类型支持                    |
| l0CStages        | 1      | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲    |
| enableL1Resident | false  | 是否开启L1常驻                                   |

### MmadMultiBatch

`DispatchPolicy MmadMultiBatch`支持以下几个模板参数：

| 模板参数  | 默认值 | 参数说明                                      |
| --------- | ------ | --------------------------------------------- |
| ArchTag   | 无     | 指定架构型号                                  |
| useHF32   | false  | 是否开启HF32，仅float类型支持                 |
| l0CStages | 2      | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲 |

当前L0A、L0B与L0C的缓冲区数量支持分开设置，默认情况下L0A、L0B与L0C的Stages都设置为2，开启Double Buffer。以下情况可尝试将l0CStages调大，开启多缓冲：

设矩阵Shape为`M N K`, L0上的分块大小为`m0 n0 k0`，L0A的分块大小`l0ATileSize = m0 * k0 * sizeof(ElementA)`, L0B的分块大小`l0BTileSize = n0 * k0 * sizeof(ElementB)`, L0C的分块大小`l0CTileSize = m0 * n0 * sizeof(ElementC)`。

则L0A一次能放下的batch数`l0ABatches = L0A_SIZE / l0ABStages / l0ATileSize`，L0B一次能放下的batch数`l0BBatches = L0B_SIZE / l0ABStages / l0BTileSize`，L0C一次能放下的batch数`l0CBatches = L0C_SIZE / l0CStages / l0CTileSize`。

当`l0CBatches`明显大于`l0ABatches`与`l0BBatches`时，可考虑将l0CStages调大，开启多缓冲。
