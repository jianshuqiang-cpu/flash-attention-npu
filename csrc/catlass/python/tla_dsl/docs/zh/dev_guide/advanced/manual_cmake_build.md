# 手动 CMake 构建

正常开发构建请使用 [`build.sh`](../01_build_and_test.md#2-构建)。本文只面向需要直接配置 `csrc/mlir`、调试 CMake 选项的开发者。

## 1. 前置条件

先加载 CANN，并设置环境变量 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 为已构建的 AscendNPU-IR **源码根目录**：

```bash
source /path/to/cann/set_env.sh
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="/path/to/AscendNPU-IR"
export PYTHONPATH="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}"
```

检查当前 Python 与关键依赖：

```bash
test -n "${ASCEND_HOME_PATH}"
test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
python -c "import pybind11; import mlir"
```

手动构建不会安装缺失依赖。环境准备方式见[环境准备](../00_environment_setup.md)。

## 2. 配置和编译

在 DSL 子项目根目录执行：

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl

cmake -S csrc/mlir -B build/cmake/manual \
  -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/cmake/manual --target tla-compiler
```

主要产物位于：

```text
build/cmake/manual/python/catlass/_tla_type_bridge_native*.so
build/cmake/manual/tools/tla-compile/TlaCompile
```

检查产物：

```bash
test -x build/cmake/manual/tools/tla-compile/TlaCompile
test -n "$(find build/cmake/manual/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
```

该构建目录不会把扩展链接到源码包。完整开发构建请使用 `./build.sh`。

## 3. CMake 配置项

以下配置项由 `csrc/mlir/CMakeLists.txt` 读取：

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `ENABLE_CPU_TRACE_INTRINSIC` | `OFF` | 启用 CPU trace intrinsic |
| `BISHENGIR_BUILD_TEMPLATE` | `ON` | 构建 HIVM template bitcode |
| `CATLASS_INCLUDE_DIR` | `/path/to/catlass/include` | CATLASS 公共头文件目录 |

例如配置 Release 构建并关闭 HIVM template bitcode：

```bash
cmake -S csrc/mlir -B build/cmake/manual-release \
  -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBISHENGIR_BUILD_TEMPLATE=OFF
```

`MLIR_DIR`、`LLVM_DIR` 和 `MLIR_TBLGEN_INCLUDE_DIR` 由 CMake 根据 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 指向的源码树自动设置。

## 4. 常见错误

### 找不到 AscendNPU-IR 头文件或库

`CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 的取值是源码根目录。所需文件见 [AscendNPU-IR 构建](ascend_npu_ir.md#3-验证构建产物)。

### 找不到 `pybind11`

```bash
python -c "import pybind11; print(pybind11.get_include())"
```

手动 CMake 构建不使用 PEP 517 构建隔离，因此 `pybind11` 必须能在 `Python3_EXECUTABLE` 对应的环境中导入。

### 无法导入 `mlir` 或找不到 `libMLIRPythonCAPI.so`

```bash
python -c "import mlir._mlir_libs as libs; print(libs.__file__)"
```

若导入失败，检查 `PYTHONPATH` 是否包含 `${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core`。若导入成功但动态库加载失败，还需确认系统动态库搜索路径包含 AscendNPU-IR 的 MLIR Python 库目录。
