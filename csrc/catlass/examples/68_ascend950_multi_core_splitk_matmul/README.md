# MultiCoreSplitkMatmul Example Readme

## 代码组织

```text
├── 68_ascend950_multi_core_splitk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── multi_core_splitk_matmul.cpp # 主文件
```

## 模板说明

该模板为多核切K模板，通过切分`K`，划分出更多的任务块，从而利用更多的计算核心。

```sh
# 编译指定用例
bash scripts/build.sh 68_ascend950_multi_core_splitk_matmul -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./68_ascend950_multi_core_splitk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 使用场景说明

该模板主要用于解决负载不均衡，如果划分出来的任务块比核心数量还少，那么可以通过切分`K`，划分出更多计算任务，从而让各个核心都能参与计算。
设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，AI Core数量为`C`，当`CeilDiv(M, m1) * CeilDiv(N, n1) <= C/2`时候，采用该模板能获取较优的性能。

**测试性能建议注释掉精度比较代码。**
