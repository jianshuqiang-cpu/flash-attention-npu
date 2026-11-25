# <<<>>>直调新开发算子

## 写在前面

该文档主要说明，开发者完成CATLASS新算子开发后，如何通过Ascend C的`<<<>>>`算子调用符直接启动新开发的算子，以及相关注意事项。

## <<<>>> 语法

算子调用符`<<<...>>>`是Ascend C提供的一种语法，封装了对Runtime API的调用，方便开发者将算子调度到NPU的AI Core上执行。
以下是通过`<<<>>>`调用Kernel执行的示例：

```cpp
kernel_name<<<blockDim, l2ctrl, stream>>>(argument_list);
```

三个参数的含义如下：

| 参数       | 类型          | 说明                         |
| ---------- | ------------- | ---------------------------- |
| `blockDim` | `uint32_t`    | 用多少个AI Core来执行该算子  |
| `l2ctrl`   | `void*`       | 保留参数，固定设为`nullptr`  |
| `stream`   | `aclrtStream` | 管理异步操作执行顺序的流对象 |

其中`blockDim`表示需要多少个硬件AI Core来执行算子，一般通过`platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()`接口来获取。算子内可通过`GetBlockIdx()`获取当前核索引，通过`GetBlockNum()`获取总核数。

> **说明**：算子的调用是异步的，`<<<>>>`调用结束后控制权立刻返回给Host端。如需等待执行完成，需调用`aclrtSynchronizeStream(stream)`。

## 直调流程

基于CATLASS模板组件开发新Kernel后，使用`<<<>>>`直调的整体流程如下：

1. 环境初始化：`aclInit`、`aclrtSetDevice`、`aclrtCreateStream`。
2. 数据准备：`aclrtMallocHost`分配并初始化Host内存，`aclrtMalloc`分配Device内存，`aclrtMemcpy`将数据拷入Device。
3. 组装模板组件：选择ArchTag、DispatchPolicy、TileShape、数据类型，组装BlockMmad等组件，拼出Kernel类型。
4. 使用`<<<>>>`调用算子：

   ```cpp
   auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
   Catlass::KernelAdapter<MyKernel><<<aicCoreNum, nullptr, stream>>>(params);
   ```

5. 结果拷回：通过`aclrtMemcpy`将Device上的运算结果拷贝回Host。
6. 同步等待：`aclrtSynchronizeStream(stream)`。
7. 资源释放：`aclrtDestroyStream(stream)`、`aclrtResetDevice(deviceId)`、`aclFinalize()`。

## <<<>>> 直调与DeviceGemm的对比

CATLASS的Device层（如`DeviceGemm`）内部就是通过`<<<>>>`启动算子的，并额外封装了`CanImplement`检查、Workspace管理等便利功能。

| 对比维度      | `<<<>>>`直调                           | `DeviceGemm`                 |
| ------------- | -------------------------------------- | ---------------------------- |
| 调用方式      | 手写`KernelAdapter<...><<<>>>(params)` | `matmulOp(stream, blockDim)` |
| Workspace管理 | 需手动处理                             | 封装在适配器内               |
| 推荐场景      | 原型开发、调测阶段、非标Kernel         | 标准GEMM/GEMV/Conv           |

简单来说，开发标准算子推荐用`DeviceGemm`；需要更多控制权时再选择直调。
