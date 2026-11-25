# Gemm/Tile 类模板概述

Gemm的tile层API作为[blockMmad](../block/block_mmad.md)的模板参数，一般不需要专门传入（blockMmad会设置默认值），仅在为了特定场景性能优化、或者实现特定功能时，需要在kernel模板组装时做声明。

## API 清单

| 组件名                                         |                   描述                    |
| :--------------------------------------------- | :---------------------------------------: |
| [tile_copy](./tile_copy/README.md)             | 完成mmad所需要的所有tile层搬运模板的集合  |
| [tile_mmad](./tile_mmad/README.md)             |              tile层mmad计算               |
| [tile_muls](./tile_muls.md)                    |              tile层标量乘法               |
| [tile_traits](./tile_traits.md)                |            Prologue trait包装             |
| [tile_copy_tla](./tile_copy_tla/README.md)     |       TLA搬运模板基类声明和实现索引       |
| [copy_gm_to_l1](./copy_gm_to_l1/README.md)     |           将tile块从GM搬运到L1            |
| [copy_l1_to_l0a](./copy_l1_to_l0a/README.md)   |        将A矩阵tile块从L1搬运到L0A         |
| [copy_l1_to_l0b](./copy_l1_to_l0b/README.md)   |        将B矩阵tile块从L1搬运到L0B         |
| [copy_l1_to_bt](./copy_l1_to_bt/README.md)     |         将Bias Table从L1搬运到BT          |
| [copy_l1_to_fp](./copy_l1_to_fp.md)            |     将数据通过FixPipe通道从L1搬运到GM     |
| [copy_l0c_to_dst](./copy_l0c_to_dst/README.md) | L0C搬运共享基础设施（量化模式、枚举定义） |
| [copy_l0c_to_gm](./copy_l0c_to_gm/README.md)   |            L0C累加结果搬运到GM            |
| [copy_l0c_to_ub](./copy_l0c_to_ub.md)          |            L0C累加结果搬运到UB            |
| [copy_gm_to_ub](./copy_gm_to_ub/README.md)     |            将数据从GM搬运到UB             |
| [copy_ub_to_gm](./copy_ub_to_gm/README.md)     |            将数据从UB搬运到GM             |
| [cast_fp8_to_fp16](./cast_fp8_to_fp16.md)      |           FP8反量化并转换为FP16           |
| [cast_int4_to_int8](./cast_int4_to_int8.md)    |              INT4转换为INT8               |
| [cast_int8_to_fp16](./cast_int8_to_fp16.md)    |          INT8反量化并转换为FP16           |
