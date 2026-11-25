# CATLASS功能性能调测

CATLASS示例工程可适配大多数[CANN](https://www.hiascend.com/cann)提供的调测工具，算子开发阶段，可基于CATLASS示例工程进行初步开发调优，无需关注具体的工具适配操作，待算子基础功能、性能达到预期，再迁移到其他工程中。

下述文档介绍使用[CANN](https://www.hiascend.com/cann)已有的工具进行调测、调优的开发实践。

## 功能调试

工具介绍：

- [msDebug](./evaluation/msdebug.md) - 类gdb/lldb的调试工具[msDebug快速入门](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)
- [printf](./evaluation/print.md) - 基于[CCE Intrinsic](https://www.hiascend.com/document/redirect/CannCommunityccedev)，在算子device侧进行打印调试
- [ascendc_dump](./evaluation/ascendc_dump.md) - 基于[原生AscendC API](https://www.hiascend.com/document/redirect/CannCommunityascendcapidumptensor)，对关键数据打点调测

相关实践：

- [精度分析基础](./evaluation/precision_analysis_basics.md)
- [样例精度问题定位](./evaluation/precision_debug.md)

## 性能调优

工具介绍：

- [msProf&Profiling](./evaluation/performance_tools.md) - 基于性能调优工具[msProf](https://www.hiascend.com/document/redirect/CannCommunitymsopprofuserguide)和[Profiling](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlasag_25_0056.html)进行调优实践
  - [单算子性能分析：msProf](./evaluation/performance_tools.md#msprof-single-operator-analysis)
  - [整网性能分析：Profiling](./evaluation/performance_tools.md#profiling-network-analysis)
- [msTuner_CATLASS](../../../tools/tuner/README.md) - Tiling自动寻优工具

相关实践：

- [bottleneck_analysis_and_optimization](./evaluation/bottleneck_analysis_and_optimization.md)：性能瓶颈分析及优化手段
