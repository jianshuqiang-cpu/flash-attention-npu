# Epilogue/Fusion Templates Overview

`Fusion` is the directory where the graph orchestrator and node layers of EVG are implemented. The actual code is located in:

- `include/catlass/epilogue/fusion/fusion.hpp`
- `include/catlass/epilogue/fusion/*.hpp`

This document serves only as a directory index. Node parameters, `Arguments` structures, common operators, and sample indices are centrally located in [evg_api](../../../../evg_api.md).

## File Responsibilities

| Component| Description|
| --- | --- |
| `fusion.hpp` | Main entry header file, which directly includes common graph orchestrators and nodes.|
| `tree_visitor.hpp` | Organization logic for `TreeVisitor`.|
| `topological_visitor.hpp` | Organization logic for `TopologicalVisitor`.|
| `VisitorImplBase` | Aggregates `Arguments`, `Params`, workspace calculations, and feasibility checks.|
| `VisitorImpl` | Constructs callbacks in a unified manner and provides a general framework for all nodes.|
| `visitor_*.hpp` | Specific node implementation.|
| `operations.hpp` | Operator encapsulation used by `VisitorCompute`.|
| `visitor_impl_base.hpp` | Definitions for `VisitStage` and the core basic framework.|

## References

- [evg_api](../../../../evg_api.md)
- [01_evg_design](../../../../../2_Design/03_evg/01_evg_design.md)
- [02_evg_extension](../../../../../2_Design/03_evg/02_evg_extension.md)
