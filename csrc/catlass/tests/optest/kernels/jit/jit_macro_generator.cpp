/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "jit_macro_generator.h"

#include <algorithm>

#include "kernel_utils.h"

namespace CatlassKernel {

static const char* LayoutToStr(bool isTranspose, bool isNz)
{
    if (isNz)
        return isTranspose ? "nZ" : "zN";
    return isTranspose ? "ColumnMajor" : "RowMajor";
}

void JitMacroGenerator<TParams>::appendTo(std::unordered_map<std::string, std::string>& macros, const TParams& p)
{
    for (auto& [k, dtype] : p.element) {
        macros["CATLASS_JIT_ELEMENT_" + k] = AclDtypeToBishengTypeStr(dtype);
    }
    for (auto& [k, _] : p.transpose) {
        macros["CATLASS_JIT_LAYOUT_" + k] = LayoutToStr(p.trans(k), p.nz(k));
    }
    macros["CATLASS_JIT_LAYOUT_C"] = "RowMajor";
}

std::string JitMacroGenerator<TParams>::makeKernelName(
    const char* kernelName, const std::unordered_map<std::string, std::string>& macros)
{
    std::vector<std::string> keys;
    keys.reserve(macros.size());
    for (auto& [k, _] : macros)
        keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    std::string name = kernelName;
    for (auto& k : keys)
        name += "_" + macros.at(k);
    return name;
}

std::unordered_map<std::string, std::string> JitMacroGenerator<TParams>::generate(
    const char* kernelName, const TParams& p)
{
    std::unordered_map<std::string, std::string> macros;
    appendTo(macros, p);
    auto jitName = makeKernelName(kernelName, macros);
    macros["CATLASS_KERNEL_NAME"] = kernelName;
    macros["CATLASS_JIT_KERNEL_NAME"] = jitName;
    return macros;
}

} // namespace CatlassKernel
