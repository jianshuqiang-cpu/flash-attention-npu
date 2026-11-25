# Ascend950 Matmul EVG 示例

本目录集中展示 EVG（Epilogue Visitor Graph）在 Ascend950 GEMM 尾处理中的典型用法，包含 7 个可执行文件：

| 可执行文件                           | 源文件                      | 场景               | EVG 组织                   | 数据通路         |
| ------------------------------------ | --------------------------- | ------------------ | -------------------------- | ---------------- |
| `64_ascend950_matmul_evg_add`        | `matmul_evg_add.cpp`        | D = A×B + X        | TreeVisitor                | GM workspace     |
| `64_ascend950_matmul_evg_leaky_relu` | `matmul_evg_leaky_relu.cpp` | D = LeakyRelu(A×B) | TreeVisitor                | GM workspace     |
| `64_ascend950_matmul_evg_sigmoid`    | `matmul_evg_sigmoid.cpp`    | D = Sigmoid(A×B)   | TreeVisitor                | GM workspace     |
| `64_ascend950_matmul_evg_silu`       | `matmul_evg_silu.cpp`       | D = Silu(A×B)      | TreeVisitor                | GM workspace     |
| `64_ascend950_matmul_evg_tanh`       | `matmul_evg_tanh.cpp`       | D = Tanh(A×B)      | TopologicalVisitor         | GM workspace     |
| `64_ascend950_matmul_evg_bias`       | `matmul_evg_bias.cpp`       | D = A×B + bias     | TreeVisitor + RowBroadcast | GM workspace     |
| `64_ascend950_matmul_evg_add_ub`     | `matmul_evg_add_ub.cpp`     | D = A×B + X        | TreeVisitor                | L0C→UB workspace |

## 编译

`build.sh` 每次只编译并 **install 一个** CMake target 到 `output/bin/`。

```bash
source ${ASCEND_HOME}/ascend-toolkit/set_env.sh

# 仅编译单个示例
bash scripts/build.sh -DCATLASS_ARCH=3510 64_ascend950_matmul_evg_add

# 编译本目录全部 7 个可执行文件并安装到 output/bin/
for t in 64_ascend950_matmul_evg_add \
         64_ascend950_matmul_evg_leaky_relu \
         64_ascend950_matmul_evg_sigmoid \
         64_ascend950_matmul_evg_silu \
         64_ascend950_matmul_evg_tanh \
         64_ascend950_matmul_evg_bias \
         64_ascend950_matmul_evg_add_ub; do
  bash scripts/build.sh -DCATLASS_ARCH=3510 "$t"
done

# 仅编译、不逐个 install 时，可先构建聚合 target，再按组件安装：
# cmake --build build --target 64_ascend950_matmul_evg_all -j
# cmake --install build --component 64_ascend950_matmul_evg_add --component 64_ascend950_matmul_evg_leaky_relu ...
```

编译全部 Catlass 示例（含本目录及其他 example）：

```bash
bash scripts/build.sh -DCATLASS_ARCH=3510 catlass_examples
```

## 运行

```bash
cd output/bin
./64_ascend950_matmul_evg_add 256 512 1024 0
./64_ascend950_matmul_evg_tanh 512 256 1024 0
```

更多设计说明见 `docs/zh/2_Design/03_evg/`.
