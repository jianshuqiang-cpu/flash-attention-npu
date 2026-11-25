# 环境准备

本文档汇总 CATLASS DSL 的环境要求和安装入口。

## 1. 选择安装方式

任选一种安装方式：

- [Conda 安装](environment/conda.md)：使用仓库的 `environment.yml` 创建开发环境（**不**包含 CANN Toolkit 与 AscendNPU-IR，需另行自行安装 CANN 并手动构建 AscendNPU-IR）。
- [Docker 安装](environment/docker.md)：使用仓库的 `Dockerfile` 构建开发镜像，包含任何开发依赖。

两种方式使用相同的[编译与测试](01_build_and_test.md)流程。

## 2. 环境要求

支持的昇腾产品为 Ascend950PR 和 Ascend950DT。

| 组件 | 版本或要求 |
| --- | --- |
| 操作系统 | CANN 支持的 Linux |
| CANN Toolkit | `>=9.1.0` |
| Python | `>=3.10,<3.14` |
| CMake | `>=3.28,<4.0` |
| Ninja | `>=1.12` |
| Clang / Clang++ | 构建 AscendNPU-IR 时 `>=10`；推荐 19 |
| lld | 与 Clang 配套 |
| AscendNPU-IR | `feature/regbase@a07821269ede7a5e683ac02c8a2d291608083741` |

Python 依赖约束由以下文件维护：

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | Python 构建、运行时和开发依赖的最小约束 |
| `requirements.txt` | 仓库开发环境使用的 Python 工具集合 |
| `requirements-docs.txt` | 文档站（MkDocs）构建依赖 |
| `environment.yml` | Conda 环境定义 |

## 3. 环境检查

基础环境检查：

```bash
python --version
cmake --version
ninja --version
test -n "${ASCEND_HOME_PATH}"
test -n "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"
test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
PYTHONPATH="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}" \
  python -c "import mlir"
```

以上命令均无任何输出、退出码为 `0`，即表示环境就绪；`test` 类命令失败时退出码非零并输出错误信息。

NPU 端到端示例还需要：

```bash
npu-smi info
python -c "import torch; import torch_npu"
```
