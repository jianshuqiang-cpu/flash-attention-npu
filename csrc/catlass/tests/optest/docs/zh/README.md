# torch-catlass 测试框架文档

## 目录结构

```text
docs/zh/
├── README.md                          ← 本文：目录索引
├── 1_Architecture/                    ← 架构总览
│   ├── 01_overview.md                 ─ 体系结构与分层设计
│   └── 02_data_flow.md                ─ 端到端数据流
├── 2_Layer_Guide/                     ← 层次指南
│   ├── 01_python_layer.md             ─ Python 包层
│   ├── 02_bridge_layer.md             ─ C++ 桥接层（torch op 注册、NPU 分发）
│   ├── 03_jit_compiler_layer.md       ─ JIT 编译层（编译管线、缓存、UUID）
│   └── 04_kernel_common_layer.md      ─ 内核公共层（Padding、Runner、Workspace）
└── 3_Build_and_Test/                  ← 构建与测试
    ├── 01_build_system.md             ─ 构建系统（CMake、scikit-build、多架构）
    └── 02_testing.md                  ─ 集成测试（pytest 模式、精度标准）
```

## 分层概览

| 层次           | 职责                                           | 关键文件                             |
| -------------- | ---------------------------------------------- | ------------------------------------ |
| **Python 层**  | 用户 API、动态库加载、版本管理                 | `torch_catlass/*.py`                 |
| **C++ 桥接层** | `torch.ops` 注册、NPU runtime 分发             | `src/*`                              |
| **JIT 编译层** | 运行时内核编译、缓存、宏注入                   | `kernels/jit/*`, `kernels/include/*` |
| **内核公共层** | 内核模板共享工具（padding、runner、workspace） | `kernels/common/*`                   |
| **构建系统**   | 多架构预编译、JIT 模板验证、wheel              | `CMakeLists.txt`, `build.sh`         |

## 设计原则

1. **JIT 优先**：新算子默认 JIT 编译，dtype/layout/shape 变化无需重新链接。
2. **SHA256 UUID**：每种编译宏组合 → 排序 → SHA256 → 确定性的 64 字符 hex 标识符。
3. **Workspace 注入**：NPU 内存由 torch 层统一管理，JIT 内核通过全局符号 `g_catlassWorkspaceAlloc` 获取分配器。
4. **ABI 稳定**：所有 JIT 内核入口统一为 3 参数 `(blockNum, stream, params)`。
5. **预编译回退**：关键内核可预编译为特定架构的 `.so` 随包分发。
