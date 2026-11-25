# CATLASS_DSL功能性能调测

作为基于[CATLASS](../../../../README.md#catlass)封装的 Python 侧 DSL 工具，CATLASS_DSL 示例工程可无缝适配大多数[CANN](https://www.hiascend.com/cann)调测工具。建议在算子开发初期，先基于此工程完成功能与性能的快速验证（免除工具适配成本），待核心指标达标后再迁移至目标生产工程。

下述文档介绍使用[CANN](https://www.hiascend.com/cann)已有的工具进行调测、调优的开发实践。

## 性能调优

工具介绍：

- [msProf&Profiling](evaluation/dsl_performance_tools.md) - 基于性能调优工具[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/Profiling/atlasprofiling_16_0010.html)和[Profiling](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlasag_25_0056.html)进行调优实践
  - [单算子性能分析：msProf](evaluation/dsl_performance_tools.md#用msprof进行单算子性能分析)
  - [整网性能分析：Profiling](evaluation/dsl_performance_tools.md#用profiling进行整网性能分析)
