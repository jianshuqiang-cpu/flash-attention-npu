#!/usr/bin/env python3
"""Generate kernel entry .cpp for a JIT example.

Reads examples/<dir>/CMakeLists.txt to determine kernel type (cube→AIC, mix→MIX),
then generates the JIT entry file with the correct JitKernelType.

Usage: python gen_entry.py <nn> <name> [--macros apply_opt|scheduler|none]
"""

import re
import sys
from pathlib import Path


KERNEL_TYPE_MAP = {"cube": "AIC", "mix": "MIX"}

ENTRY_SIMPLE = """#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {{

extern "C" void {FuncName}(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{{
    auto* entry = JitCompiler::instance().getKernel(
        "{TemplateName}", JitMacroGenerator<TParams>::generate("{KernelName}", tParams), JitKernelType::{KernelType});
    if (entry) {{
        entry(blockNum, stream, &params);
    }}
    aclrtSynchronizeStream(stream);
}}

}} // namespace CatlassKernel
"""

ENTRY_SCHEDULER = """#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {{

extern "C" void {FuncName}(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{{
    auto macros = JitMacroGenerator<TParams>::generate("{KernelName}", tParams);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = (params.m > params.n) ? "30" : "31";
    auto* entry = JitCompiler::instance().getKernel("{TemplateName}", macros, JitKernelType::{KernelType});
    if (entry) {{
        entry(blockNum, stream, &params);
    }}
    aclrtSynchronizeStream(stream);
}}

}} // namespace CatlassKernel
"""

ENTRY_APPLYOPT = """#include "catlass_kernel.h"
#include "common/optimized_macro_generator.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {{

extern "C" void {FuncName}(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{{
    auto macros = JitMacroGenerator<TParams>::generate("{KernelName}", tParams);
    ApplyOptMacros(
        macros, params.m, params.n, params.k, tParams.nz("A"), tParams.trans("A"), tParams.nz("B"), tParams.trans("B"));
    auto* entry = JitCompiler::instance().getKernel("{TemplateName}", macros, JitKernelType::{KernelType});
    if (entry) {{
        entry(blockNum, stream, &params);
    }}
    aclrtSynchronizeStream(stream);
}}

}} // namespace CatlassKernel
"""


def parse_kernel_type(example_cmake: Path) -> str:
    """Parse kernel type from example CMakeLists.txt. Returns 'AIC' or 'MIX'."""
    text = example_cmake.read_text()
    m = re.search(r"catlass_example_add_executable\s*\(\s*\S+\s+(cube|mix)\b", text)
    if m:
        return KERNEL_TYPE_MAP[m.group(1)]
    return "AIC"  # default


def func_name(name: str) -> str:
    """snake_case to PascalCase (with handling for TLA/known suffixes)."""
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <nn> <name> [--macros none|scheduler|apply_opt]")
        sys.exit(1)

    nn = sys.argv[1]
    name = sys.argv[2]
    macro_mode = "none"
    for arg in sys.argv[3:]:
        if arg == "--macros" and len(sys.argv) > sys.argv.index(arg) + 1:
            macro_mode = sys.argv[sys.argv.index(arg) + 1]

    root = Path(__file__).resolve().parents[4]  # catlass repo root
    example_dir = root / "examples" / f"{nn}_{name}"
    cmake_file = example_dir / "CMakeLists.txt"

    kernel_type = parse_kernel_type(cmake_file) if cmake_file.exists() else "AIC"
    fname = func_name(name)
    tpl_name = f"{name}_impl.cpp"

    if macro_mode == "apply_opt":
        content = ENTRY_APPLYOPT.format(
            FuncName=fname, TemplateName=tpl_name, KernelName=name, KernelType=kernel_type)
    elif macro_mode == "scheduler":
        content = ENTRY_SCHEDULER.format(
            FuncName=fname, TemplateName=tpl_name, KernelName=name, KernelType=kernel_type)
    else:
        content = ENTRY_SIMPLE.format(
            FuncName=fname, TemplateName=tpl_name, KernelName=name, KernelType=kernel_type)

    print(content)


if __name__ == "__main__":
    main()
