# Dual-Level Quant MX Batch Matmul

本样例实现二级量化 + MX FP4 Batch Matmul。AIV 侧先将 A/B 全量量化到 workspace，然后通过 `AscendC::SyncAll<false>()` 通知 AIC 侧执行 MX FP4 matmul。

## 代码组织

```text
experimental
├── matmul
│   ├── 63_ascend950_dual_level_quant_mx_batch_matmul
│   │   ├── CMakeLists.txt
│   │   ├── README.md
│   │   ├── dual_level_quant_mx_batch_matmul.cpp
│   │   └── gen_data.py
```

依赖的新增 CATLASS 头文件：

```text
include/catlass/epilogue/block/block_epilogue_dual_level_quant_mx.hpp
include/catlass/epilogue/tile/tile_copy_dual_level_quant_mx.hpp
include/catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp
```

## 使用示例

编译：

```bash
bash scripts/build.sh 63_ascend950_dual_level_quant_mx_batch_matmul -DCATLASS_ARCH=3510
```

生成测试数据：

```bash
python3 examples/63_ascend950_dual_level_quant_mx_batch_matmul/gen_data.py 1 1024 1024 1024
```

执行算子：

```bash
./output/bin/63_ascend950_dual_level_quant_mx_batch_matmul 1 1024 1024 1024 0
```

## 约束说明

- K 必须为偶数，满足 FP4 打包要求。
- K 轴需满足 MX 布局约束。
- 量化仅沿 K 轴进行。
- `LEVEL0_BLOCK_SIZE = 512`，`LEVEL1_BLOCK_SIZE = 32`。

## 核心组件

| 组件             | 选型                                  |
| ---------------- | ------------------------------------- |
| ArchTag          | `Arch::Ascend950`                     |
| Kernel           | `DualLevelQuantMxBatchedMatmulTla`    |
| AIV BlockQuant   | `BlockQuantDualLevelMx`               |
| TileCopy (Quant) | `TileCopyDualLevelQuantMx`            |
| BlockMmad        | `BlockMmadTla`                        |
| BlockScheduler   | `GemmIdentityBlockSwizzle<3, 0/1>`    |
| DispatchPolicy   | `MmadMx<Ascend950, true, 16>`         |
| L1TileShape      | `Shape<Int<256>, Int<256>, Int<512>>` |
| L0TileShape      | `Shape<Int<256>, Int<256>, Int<256>>` |
| ElementInput     | `float16_t`                           |
| ElementA/B       | `float4_e2m1x2_t`                     |
| ElementMxScale   | `float8_e8m0_t`                       |
| ElementC         | `bfloat16_t`                          |

## 前置条件

- 当前实现依赖 mix kernel 中 AIV/AIC 两侧配对调用 `AscendC::SyncAll<false>()`。
- 多流生产场景上线前需要确认 batchmode 已开启，否则 `SyncAll` 可能死锁。
