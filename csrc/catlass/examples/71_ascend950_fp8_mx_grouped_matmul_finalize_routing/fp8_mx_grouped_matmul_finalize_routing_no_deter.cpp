/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in
 * the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <array>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle_grouped_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_no_deter_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

using namespace Catlass;
using namespace tla;

struct GroupedGemmFinalizeRoutingOptions : public GroupedGemmOptions {
    const std::string HELPER =
        "problem_count m n k trans_b group_list_type enable_bias batch data_parallel_size enable_shared_input "
        "shared_input_weight shared_input_offset quant_type [device_id]]]";

    uint32_t transB{0};
    uint32_t batchSize{0};
    uint32_t groupListType{0};
    uint32_t enableBias{0};
    float sharedInputWeight{0.0f};
    uint32_t sharedInputOffset{0};
    std::string quantDataType = "float8_e4m3fn";
    uint32_t dataParallelSize{1};
    uint32_t enableSharedInput{0};
    const std::array<std::string, 2> quantDataTypes = {"float8_e4m3fn", "float8_e5m2"};

    GroupedGemmFinalizeRoutingOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            GROUP_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            TRANS_B_INDEX,
            GROUP_LIST_TYPE_INDEX,
            ENABLE_BIAS,
            BATCH_INDEX,
            DATA_PARALLEL_SIZE_INDEX,
            ENABLE_SHARED_INPUT_INDEX,
            SHARED_INPUT_WEIGHT_INDEX,
            SHARED_INPUT_OFFSET_INDEX,
            QUANT_TYPE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        transB = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::TRANS_B_INDEX)]);
        enableBias = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::ENABLE_BIAS)]);
        groupListType = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_LIST_TYPE_INDEX)]);
        batchSize = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_INDEX)]);
        dataParallelSize = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DATA_PARALLEL_SIZE_INDEX)]);
        enableSharedInput = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::ENABLE_SHARED_INPUT_INDEX)]);
        sharedInputWeight = std::atof(argv[static_cast<uint32_t>(ArgsIndex::SHARED_INPUT_WEIGHT_INDEX)]);
        sharedInputOffset = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::SHARED_INPUT_OFFSET_INDEX)]);
        quantDataType = std::string(argv[static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)]);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        if (dataParallelSize == 0) {
            std::cerr << "Invalid data_parallel_size: must not be 0" << std::endl;
            return -1;
        }

        if (quantDataTypes.end() == std::find(quantDataTypes.begin(), quantDataTypes.end(), quantDataType)) {
            std::cerr << "Invalid quantData type: " << quantDataType << std::endl;
            return -1;
        }

        return 0;
    }
};

using Options = GroupedGemmFinalizeRoutingOptions;

#define MALLOC_AND_COPY_TO_NPU_DEVICE(descr, hostData, size)                                           \
    uint8_t* device##descr{nullptr};                                                                   \
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device##descr), size, ACL_MEM_MALLOC_HUGE_FIRST)); \
    ACL_CHECK(aclrtMemcpy(device##descr, size, (hostData).data(), size, ACL_MEMCPY_HOST_TO_DEVICE))

#define MALLOC_ON_NPU_DEVICE(descr, size) \
    uint8_t* device##descr{nullptr};      \
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device##descr), size, ACL_MEM_MALLOC_HUGE_FIRST))

template <typename T>
bool SaveResult(const std::string& filename, const std::vector<T>& data, const size_t dataSize)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    std::vector<char> buffer(dataSize);
    std::memcpy(buffer.data(), data.data(), dataSize);

    file.write(buffer.data(), dataSize);
    if (!file) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

template <
    typename ElementA, class ElementB, class L1TileShape, class L0TileShape, bool transB = false,
    bool enableBias = false, bool enableSharedInput = false>
void MxGroupedMatmulFinalizeRouting(Options const& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t groupCount = options.problemCount;
    uint32_t batch = options.batchSize;
    uint32_t groupListType = options.groupListType;
    float sharedInputWeight = options.sharedInputWeight;
    uint32_t sharedInputOffset = options.sharedInputOffset;
    uint32_t dataParallelSize = options.dataParallelSize;
    uint32_t bsdp = batch / dataParallelSize;

    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementC = float;
    using ElementMxScale = float8_e8m0_t;
    using ElementGroupList = int64_t;
    using ElementBias = std::conditional_t<enableBias, bfloat16_t, void>;
    using ElementLogit = float;
    using ElementRowIndex = int64_t;
    using ElementSharedInput = std::conditional_t<enableSharedInput, bfloat16_t, void>;
    using ElementOut = float;
    static_assert(
        (std::is_same_v<ElementA, float8_e5m2_t> || std::is_same_v<ElementA, float8_e4m3_t>) &&
            (std::is_same_v<ElementB, float8_e5m2_t> || std::is_same_v<ElementB, float8_e4m3_t>) &&
            std::is_same_v<ElementMxScale, float8_e8m0_t>,
        "ElementA and ElementB must be float8_e5m2_t or float8_e4m3_t, ElementMxScale must be float8_e8m0_t");
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(k, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity() * groupCount;
    size_t lenMxScaleA = m * mxScaleAlignedK;
    size_t lenMxScaleB = mxScaleAlignedK * n * groupCount;
    size_t lenBias = n * groupCount;
    size_t lenLogit = m;
    size_t lenRowIndex = m;
    size_t lenSharedInput = bsdp * n;
    size_t lenOut = batch * n;

    size_t sizeA = lenA * SizeOfBits<ElementA>::value / 8;
    size_t sizeB = lenB * SizeOfBits<ElementB>::value / 8;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeBias = 0;
    if constexpr (enableBias) {
        sizeBias = lenBias * sizeof(ElementBias);
    }
    size_t sizeGroupList = groupCount * sizeof(ElementGroupList);
    size_t sizeLogit = lenLogit * sizeof(ElementLogit);
    size_t sizeRowIndex = lenRowIndex * sizeof(ElementRowIndex);
    size_t sizeSharedInput = 0;
    if constexpr (enableSharedInput) {
        sizeSharedInput = lenSharedInput * sizeof(ElementSharedInput);
    }
    size_t sizeOut = lenOut * sizeof(ElementOut);
    size_t sizeWorkspace = 0;

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(sizeMxScaleA);
    std::vector<int8_t> hostMxScaleB(sizeMxScaleB);
    std::vector<int8_t> hostBias(sizeBias);
    std::vector<int8_t> hostLogit(sizeLogit);
    std::vector<int8_t> hostRowIndex(sizeRowIndex);
    std::vector<int8_t> hostGroupList(sizeGroupList);
    std::vector<int8_t> hostSharedInput;
    if constexpr (enableSharedInput) {
        hostSharedInput.resize(sizeSharedInput);
    }

    const auto releaseAcl = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile("./data/a_8.bin", hostA.data(), sizeA)) {
        releaseAcl();
        return;
    }
    if (!ReadFile("./data/b_8.bin", hostB.data(), sizeB)) {
        releaseAcl();
        return;
    }
    if (!ReadFile("./data/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        releaseAcl();
        return;
    }
    if (!ReadFile("./data/b_scale.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        releaseAcl();
        return;
    }
    if constexpr (enableBias) {
        if (!ReadFile("./data/bias.bin", reinterpret_cast<char*>(hostBias.data()), sizeBias)) {
            releaseAcl();
            return;
        }
    }
    if (!ReadFile("./data/logit.bin", reinterpret_cast<char*>(hostLogit.data()), sizeLogit)) {
        releaseAcl();
        return;
    }
    if (!ReadFile("./data/row_index.bin", reinterpret_cast<char*>(hostRowIndex.data()), sizeRowIndex)) {
        releaseAcl();
        return;
    }
    if (!ReadFile("./data/group_list.bin", reinterpret_cast<char*>(hostGroupList.data()), sizeGroupList)) {
        releaseAcl();
        return;
    }
    if constexpr (enableSharedInput) {
        if (!ReadFile("./data/shared_input.bin", reinterpret_cast<char*>(hostSharedInput.data()), sizeSharedInput)) {
            releaseAcl();
            return;
        }
    }

    MALLOC_AND_COPY_TO_NPU_DEVICE(GroupList, hostGroupList, sizeGroupList);
    MALLOC_AND_COPY_TO_NPU_DEVICE(A, hostA, sizeA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(B, hostB, sizeB);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleA, hostMxScaleA, sizeMxScaleA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleB, hostMxScaleB, sizeMxScaleB);
    uint8_t* deviceBias{nullptr};
    if constexpr (enableBias) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    MALLOC_AND_COPY_TO_NPU_DEVICE(Logit, hostLogit, sizeLogit);
    MALLOC_AND_COPY_TO_NPU_DEVICE(RowIndex, hostRowIndex, sizeRowIndex);

    uint8_t* deviceSharedInput{nullptr};
    if constexpr (enableSharedInput) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void**>(&deviceSharedInput), sizeSharedInput, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(
            deviceSharedInput, sizeSharedInput, hostSharedInput.data(), sizeSharedInput, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    MALLOC_ON_NPU_DEVICE(Out, sizeOut);

    uint8_t* deviceWorkspace{nullptr};

    const auto cleanupAndFinalize = [&]() {
        if (deviceA)
            ACL_CHECK(aclrtFree(deviceA));
        if (deviceB)
            ACL_CHECK(aclrtFree(deviceB));
        if (deviceGroupList)
            ACL_CHECK(aclrtFree(deviceGroupList));
        if (deviceBias)
            ACL_CHECK(aclrtFree(deviceBias));
        if (deviceLogit)
            ACL_CHECK(aclrtFree(deviceLogit));
        if (deviceRowIndex)
            ACL_CHECK(aclrtFree(deviceRowIndex));
        if (deviceOut)
            ACL_CHECK(aclrtFree(deviceOut));
        if (deviceSharedInput)
            ACL_CHECK(aclrtFree(deviceSharedInput));
        if (deviceWorkspace)
            ACL_CHECK(aclrtFree(deviceWorkspace));
        if (deviceMxScaleA)
            ACL_CHECK(aclrtFree(deviceMxScaleA));
        if (deviceMxScaleB)
            ACL_CHECK(aclrtFree(deviceMxScaleB));
        releaseAcl();
    };

    // GemmGroupedAswtTailSplitSwizzle: 滚动核分配 + 窗口调度，支持尾块拆分。
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
    using vecTileShape = MatrixShape<tla::get<0>(L1TileShape{}) / 2, tla::get<1>(L1TileShape{})>;

    using TileCopy = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, ElementBias, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Gemm::Block::BlockMmadMxFinalizeRoutingTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    constexpr uint32_t UB_STAGES = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAscend950FinalizeRouting<UB_STAGES>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueFinalizeRoutingNoDeter<
        EpilogueDispatchPolicy, ArchTag, vecTileShape, ElementC, ElementRowIndex, ElementSharedInput>;
    using BlockScheduler = typename Gemm::Block::GemmGroupedAswtTailSplitSwizzle<>;
    using MatmulKernel = Gemm::Kernel::GroupedMxMatmulFinalizeRoutingNoDeterTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList, ElementSharedInput>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{
        aicCoreNum,
        options.problemShape,
        options.problemCount,
        deviceGroupList,
        deviceA,
        layoutA,
        deviceB,
        layoutB,
        deviceMxScaleA,
        layoutMxScaleA,
        deviceMxScaleB,
        layoutMxScaleB,
        deviceWorkspace,
        layoutC,
        deviceBias,
        deviceLogit,
        deviceRowIndex,
        deviceSharedInput,
        groupListType,
        sharedInputWeight,
        sharedInputOffset,
        batch,
        bsdp,
        deviceOut};

    MatmulAdapter matmul_op;
    if (matmul_op.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "Cannot implement the arguments" << std::endl;
        cleanupAndFinalize();
        return;
    }
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<ElementOut> hostOut(lenOut);
    ACL_CHECK(aclrtMemcpy(hostOut.data(), sizeOut, deviceOut, sizeOut, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outputFileName = "./data/result.bin";
    SaveResult<ElementOut>(outputFileName, hostOut, sizeOut);

    cleanupAndFinalize();
}

template <bool TB, bool EnableBias, bool EnableSharedInput>
void Run(Options const& options)
{
    if (options.quantDataType == "float8_e4m3fn") {
        MxGroupedMatmulFinalizeRouting<
            float8_e4m3_t, float8_e4m3_t, Shape<Int<256>, Int<256>, Int<256>>, Shape<Int<256>, Int<256>, Int<128>>, TB,
            EnableBias, EnableSharedInput>(options);
    } else if (options.quantDataType == "float8_e5m2") {
        MxGroupedMatmulFinalizeRouting<
            float8_e5m2_t, float8_e5m2_t, Shape<Int<256>, Int<256>, Int<256>>, Shape<Int<256>, Int<256>, Int<128>>, TB,
            EnableBias, EnableSharedInput>(options);
    } else {
        std::cerr << "Unexpected quant data-type mismatch." << std::endl;
    }
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    auto dispatch = [&]<bool TB>() {
        bool hasBias = static_cast<bool>(options.enableBias);
        bool hasSharedInput = static_cast<bool>(options.enableSharedInput);
        if (hasBias && hasSharedInput) {
            Run<TB, true, true>(options);
        } else if (hasBias) {
            Run<TB, true, false>(options);
        } else if (hasSharedInput) {
            Run<TB, false, true>(options);
        } else {
            Run<TB, false, false>(options);
        }
    };

    options.transB ? dispatch.template operator()<true>() : dispatch.template operator()<false>();
    return 0;
}
