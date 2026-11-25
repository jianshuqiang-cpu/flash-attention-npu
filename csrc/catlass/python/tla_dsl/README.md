# CATLASS DSL

CATLASS DSL 是 CATLASS 的 Python 前端。它在 AscendNPU-IR 的基础上构建 TLA Dialect，将 Python 中描述的 TLA 操作降级为 MLIR，并由 AscendNPU-IR 工具链编译为可执行的核函数产物。

目前，CATLASS DSL 主要对 TLA 框架提供了DSL封装，可以使用`tla.make_tensor`等API来描述TLA操作，完成算子的实现。未来，我们将继续扩展CATLASS DSL，在全量支持TLA tensor抽象接口、矩阵向量计算等接口的基础之上，进行类似CATLASS 模板的封装，实现对齐CATLASS C++模板算子的开发体验。

## 文档

- [环境准备](docs/zh/dev_guide/00_environment_setup.md)：环境要求及安装入口。
- [编译与测试](docs/zh/dev_guide/01_build_and_test.md)：构建与端到端用例。
- [API 文档](docs/zh/dev_guide/02_api_docs.md)：生成并预览 API 文档。
- [AscendNPU-IR 构建](docs/zh/dev_guide/advanced/ascend_npu_ir.md)：手动构建AscendNPU-IR。
- [手动 CMake 构建](docs/zh/dev_guide/advanced/manual_cmake_build.md)：直接配置 `csrc/mlir` 的进阶用法。

## 快速开始

先按照[环境准备](docs/zh/dev_guide/00_environment_setup.md)完成安装，并按[构建 AscendNPU-IR](docs/zh/dev_guide/advanced/ascend_npu_ir.md)预先构建依赖，同时设置 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR`。`build.sh` 本身**不会**构建 AscendNPU-IR。

下文命令均在 DSL 子项目根目录执行，其中 `/path/to/catlass` 需替换为你 clone 的 CATLASS 仓库根目录：

```bash
cd /path/to/catlass/python/tla_dsl
```

### 1. 构建

```bash
./build.sh
```

`build.sh` 默认执行 Debug 开发构建：检查 AscendNPU-IR 构建产物、生成 TLA Python op 绑定、在 `csrc/mlir/build` 构建 `TlaCompile` 与类型桥接动态库，并以 editable 模式安装 `ascend-catlass-dsl`。

### 2. NPU 端到端示例（需要 NPU）

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --device 0
```

更多构建选项（Release wheel、清理重建）与仓库级回归的说明见[编译与测试](docs/zh/dev_guide/01_build_and_test.md)。

## 兼容性

CATLASS DSL 各版本支持的硬件平台及所需的最低 CANN 版本如下表：

| CATLASS DSL 版本 | 最低支持 CANN 包版本 | 支持昇腾产品 |
| --- | --- | --- |
| 当前 | `>= 9.1.0` | Ascend 950PR / Ascend 950DT |

软硬件环境要求：

- CPU 架构：`aarch64` / `x86_64`
- 系统：CANN 支持的 Linux
- 软件依赖：Python `>= 3.10, < 3.14`、CMake `>= 3.28, < 4.0`、Ninja `>= 1.12`、Clang / Clang++ `>= 10`（构建 AscendNPU-IR 时；推荐 19）、lld、AscendNPU-IR `feature/regbase@a07821269…`

完整的环境要求与安装方式见[环境准备](docs/zh/dev_guide/00_environment_setup.md#1-环境要求)。

## 目录概览

```text
python/tla_dsl/
├── 3rdparty/AscendNPU-IR/  # AscendNPU-IR 子模块
├── catlass/                # Python 前端与运行时
├── csrc/mlir/              # TLA Dialect 与编译器实现
├── docs/zh/                # 中文文档
├── docs/en/                # 英文文档
├── examples/               # 端到端示例
├── Dockerfile              # 开发环境镜像定义
├── build_docker_image.sh   # 镜像构建入口
├── build.sh                # 项目构建入口
└── requirements.txt        # 非 CANN Python 依赖
```
