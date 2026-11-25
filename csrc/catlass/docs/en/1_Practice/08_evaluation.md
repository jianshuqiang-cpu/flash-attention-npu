# CATLASS Functional Performance Profiling and Tuning

The CATLASS sample project natively integrates with most profiling and debugging tools provided by [CANN](https://www.hiascend.com/cann). During the operator development phase, you can perform initial development and optimization directly in the CATLASS sample project without needing to configure complex tool configurations. Once the foundational functionality and performance of the operator meet your target expectations, the implementation can be migrated to other production projects.

The following sections detail development practices for profiling, debugging, and tuning using existing [CANN](https://www.hiascend.com/cann) tools.

## Functional Debugging

Introduction:

- [msDebug](./evaluation/msdebug.md): A GDB/LLDB-like debugging tool [msDebug](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0062.html).
- [printf](./evaluation/print.md): Provides device-side print debugging within the operator kernel, powered by [CCE Intrinsic](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/cceintrinsicguide/cceprogram_0001.html).
- [ascendc_dump](./evaluation/ascendc_dump.md): Inspects and debugs critical data points by leveraging the [native AscendC API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0192.html).

Related practices:

- [precision_analysis_basics](./evaluation/precision_analysis_basics.md): precision analysis basics
- [precision_debug](./evaluation/precision_debug.md): sample precision issue locating

## Performance Tuning

Introduction:

- [msProf&Profiling](./evaluation/performance_tools.md): Optimization practices driven by the performance profiling tools [msProf](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html) and [Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/graph/graphdevg/atlasag_25_0056.html).
  - [Single-operator profiling: msProf](./evaluation/performance_tools.md#single-operator-profiling-using-msprof)
  - [Whole-network profiling: Profiling](./evaluation/performance_tools.md#whole-network-profiling)
- [msTuner_CATLASS](../../../tools/tuner/README.md): automatic tiling optimization tool

Related practices:

- [bottleneck_analysis_and_optimization](./evaluation/bottleneck_analysis_and_optimization.md): performance bottleneck analysis and optimization
