# 常见问题

## 如何指定CATLASS的目标硬件架构？

我们于2026年3月第一次社区会议正式确定CATLASS社区主线将开始新增对下一代昇腾硬件Ascend 950PR/Ascend 950DT的支持，对应的正式版本从1.5.0开始。为在不同平台区分底层接口的实现，该新增支持将引入新的编译宏，用户需要注意在对应编译命令中进行相应适配。

- 新增宏：`CATLASS_ARCH`，用于指定目标架构。其取值可在[SIMD BuiltIn关键字](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_10053.html)中查询（`__NPU_ARCH__`列）。
  - `Atlas A2 训练系列产品 / Atlas A2 推理系列产品`：`2201`
  - `Atlas A3 训练系列产品 / Atlas A3 推理系列产品`：`2201`
  - `Ascend 950PR/Ascend 950DT`：`3510`

- 相关场景说明：
  - `bisheng`命令行场景：`bisheng ... -DCATLASS_ARCH=2201 ...`
  - `cmake`场景：`add_compile_definitions(CATLASS_ARCH=2201)`
  - `msopgen/aclnn`工程场景：
    - 旧写法（CANN < 9.0.0）：`add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 ...)`
    - 新写法（CANN >= 9.0.0）：`npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCATLASS_ARCH=2201)`（msopgen工程中，第一个参数默认为`ascendc_kernels`，可根据实际情况进行调整）
  - CATLASS源码仓：`bash scripts/build.sh -DCATLASS_ARCH=2201 ...`
  - 库上代码参考：[examples/CMakeLists.txt](https://gitcode.com/cann/catlass/blob/master/examples/CMakeLists.txt)
