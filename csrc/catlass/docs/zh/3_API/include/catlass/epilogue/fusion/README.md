# Epilogue/Fusion 类模板概述

`Fusion` 是 EVG 的图组织层与节点层实现所在目录。当前真实代码位于：

- `include/catlass/epilogue/fusion/fusion.hpp`
- `include/catlass/epilogue/fusion/*.hpp`

这份文档只负责目录索引。节点参数、`Arguments` 写法、常用算子和样例索引统一放在 [`evg_api`](../../../../evg_api.md)。

## 文件分工

| 组件                      | 说明                                                 |
| ------------------------- | ---------------------------------------------------- |
| `fusion.hpp`              | 总入口头文件，直接包含常用图组织器与节点             |
| `tree_visitor.hpp`        | `TreeVisitor` 的组织逻辑                             |
| `topological_visitor.hpp` | `TopologicalVisitor` 的组织逻辑                      |
| `VisitorImplBase`         | 汇总 `Arguments`、`Params`、workspace 与可实现性检查 |
| `VisitorImpl`             | 统一构造 callbacks，并为所有节点提供通用骨架         |
| `visitor_*.hpp`           | 具体节点实现                                         |
| `operations.hpp`          | `VisitorCompute` 使用的算子封装                      |
| `visitor_impl_base.hpp`   | `VisitStage` 与基础框架定义                          |

## 相关文档

- [evg_api](../../../../evg_api.md)
- [01_evg_design](../../../../../2_Design/03_evg/01_evg_design.md)
- [02_evg_extension](../../../../../2_Design/03_evg/02_evg_extension.md)
