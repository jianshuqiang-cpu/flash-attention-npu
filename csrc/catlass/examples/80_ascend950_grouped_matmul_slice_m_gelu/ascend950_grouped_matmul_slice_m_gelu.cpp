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
#include <iostream>
#include <vector>
#include <cstdlib>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "launcher/grouped_matmul_slice_m_gelu_launcher.h"
#include "helper.hpp"
#include "golden.hpp"

using namespace Catlass;
using namespace tla;

using Options = GroupedGemmOptions;

void Run(Options const& options)
{
    std::string datapath = "./examples/80_ascend950_grouped_matmul_slice_m_gelu/data/";

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenO = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeO = lenO * sizeof(fp16_t);

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagO = layout::RowMajor;

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostO(lenO);
    std::vector<float> hostO_golden(lenO);

    ReadFile(datapath + "tensor_a.bin", (void*)hostA.data(), sizeA);
    ReadFile(datapath + "tensor_b.bin", (void*)hostB.data(), sizeB);
    ReadFile(datapath + "golden_gelu_out.bin", (void*)hostO_golden.data(), lenO * sizeof(float));

    std::vector<int64_t> groupList(problemCount);
    ReadFile(datapath + "group_list.bin", (void*)groupList.data(), problemCount * sizeof(int64_t));

    size_t sizeGroupList = problemCount * sizeof(int64_t);
    uint8_t* deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceO{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceO), sizeO, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagO tagO{m, n};
    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutO = MakeLayoutFromTag(tagO);

    grouped_matmul_slice_m_gelu_launcher(
        stream, options.problemShape, problemCount, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceO,
        &deviceWorkspace);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostO.data(), sizeO, deviceO, sizeO, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<uint64_t> errorIndices = golden::CompareData(hostO, hostO_golden, k, groupList[problemCount - 1] * n);
    if (errorIndices.empty()) {
        std::cout << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare success." << std::endl;
        std::cerr << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare success." << std::endl;
    } else {
        std::cout << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cout << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
        std::cerr << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "case: " << problemCount << " " << int(m / problemCount) << " " << m << " " << n << " " << k
                  << ", Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceO));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}
