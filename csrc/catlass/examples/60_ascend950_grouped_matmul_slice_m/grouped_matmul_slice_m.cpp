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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <cstdlib>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"

using namespace Catlass;
using namespace tla;

using Options = GroupedGemmOptions;

void Run(Options const& options)
{
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
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData(hostA, -5.0, 5.0);
    golden::FillRandomData(hostB, -5.0, 5.0);
    auto groupList = golden::GenerateGroupList<int64_t>(m, problemCount);

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

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    size_t sizeWorkspace = 0;
    uint8_t* deviceWorkspace{nullptr};
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<64>>;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmadTla = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;
    if (m / problemCount >= n) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMTla<BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC};

        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aicCoreNum);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMTla<BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC};

        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aicCoreNum);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<GemmCoord> problemShapeList(problemCount);
    std::vector<LayoutTagA> layoutAList(problemCount);
    std::vector<LayoutTagB> layoutBList(problemCount);
    std::vector<LayoutTagC> layoutCList(problemCount);
    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t currentM = (i == 0) ? groupList[0] : (groupList[i] - groupList[i - 1]);
        problemShapeList[i] = GemmCoord{currentM, n, k};
        layoutAList[i] = LayoutTagA{currentM, k};
        layoutBList[i] = LayoutTagB{k, n};
        layoutCList[i] = LayoutTagC{currentM, n};
    }

    std::vector<float> hostGolden(lenC);
    golden::ComputeGroupedMatmul(
        problemCount, problemShapeList, hostA, layoutAList, hostB, layoutBList, hostGolden, layoutCList);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k, groupList[problemCount - 1] * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if (sizeWorkspace > 0) {
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
