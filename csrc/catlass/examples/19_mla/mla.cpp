/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
// Helper methods to check for errors
#include "golden.hpp"
#include "helper.hpp"
#include "mla_kernel.cpp"
#include "mla_kernel_tp1_spec.cpp"
#include "amla_kernel_tp1_spec.cpp"
#include "mla_tiling.h"

using namespace std;

// Helper function to split comma-separated string into vector of uint32_t
vector<uint32_t> splitSeqList(const string& seqStr)
{
    vector<uint32_t> seqList;
    stringstream ss(seqStr);
    string item;
    while (getline(ss, item, ',')) {
        seqList.push_back(static_cast<uint32_t>(stoul(item)));
    }
    return seqList;
}

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: mla batch \"qSeqlenList\" \"kvSeqlenList\" numHeads numBlocks blockSize [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n"
        "Example: ./19_mla 4 \"1,2,3,4\" \"128,256,512,1024\" 16 16 128\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    vector<uint32_t> qSeqlenList;
    vector<uint32_t> kvSeqlenList;
    uint32_t numHeads{0};
    uint32_t numBlocks{0};
    uint32_t blockSize{0};
    uint32_t deviceId{0};

    uint32_t maskType{0};
    uint32_t kvHeads{1};
    uint32_t embeddingSize{512};
    uint32_t embeddingSizeRope{64};
    string dataType = "half";
    string dataPath = "../../examples/19_mla/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        string qSeqStr = argv[argIndex++];
        string kvSeqStr = argv[argIndex++];
        qSeqlenList = splitSeqList(qSeqStr);
        kvSeqlenList = splitSeqList(kvSeqStr);
        if (qSeqlenList.size() != batch || kvSeqlenList.size() != batch) {
            printf("Error: The number of elements in qSeqlenList and kvSeqlenList must match the batch size.\n");
            printf(HELPER);
            return -1;
        }
        numHeads = atoi(argv[argIndex++]);
        numBlocks = atoi(argv[argIndex++]);
        blockSize = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

static void AllocMem(uint8_t** host, uint8_t** device, size_t size)
{
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t* host, uint8_t* device)
{
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS MLA kernel.
static void Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int32_t embeddingSizeRope = options.embeddingSizeRope;
    int32_t numBlocks = options.numBlocks;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = 0;
    if (!options.kvSeqlenList.empty()) {
        maxKvSeqlen = options.kvSeqlenList[0];
        for (size_t i = 1; i < options.kvSeqlenList.size(); ++i) {
            if (options.kvSeqlenList[i] > maxKvSeqlen) {
                maxKvSeqlen = options.kvSeqlenList[i];
            }
        }
    }

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }

    int32_t dTypeKey = (dataType == "half") ? 0 : 1;
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;

    // 3 bits for tilingKey(specStraKey : 1, dTypeKey : 2)
    uint32_t dTypeKeyBitLen = 2;
    uint32_t tilingKey = (specStraKey << dTypeKeyBitLen) + dTypeKey;

    // read qNtokens num
    void* qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    if ((numHeads == MLATiling::NUM128) && (numTokens % aicCoreNum <= 10) && (batch <= 40)) {
        tilingKey = (dTypeKey == 0) ? 7 : 8;
    }
    std::cout << "tilingKey : " << tilingKey << std::endl;

    // read qSeq
    void* qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    ReadFile(dataPath + "/q_seqlen.bin", qSeq, batch * sizeof(int32_t));

    // read kvSeq num
    void* kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeq, batch * sizeof(int32_t));

    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t qRopeSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSizeRope * sizeof(fp16_t);
    uint64_t kvSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kRopeSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSizeRope * sizeof(fp16_t);
    uint64_t maskSize = (uint64_t)numTokens * (uint64_t)maxKvSeqlen * sizeof(fp16_t);
    uint64_t blockTableSize =
        static_cast<uint64_t>(batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t));
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey > 0) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t* qHost;
    uint8_t* qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q_rope.
    uint8_t* qRopeHost;
    uint8_t* qRopeDevice;
    AllocMem(&qRopeHost, &qRopeDevice, qRopeSize);
    ReadFile(dataPath + "/q_rope.bin", qRopeHost, qRopeSize);
    ACL_CHECK(aclrtMemcpy(qRopeDevice, qRopeSize, qRopeHost, qRopeSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t* kHost;
    uint8_t* kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k_rope.
    uint8_t* kRopeHost;
    uint8_t* kRopeDevice;
    AllocMem(&kRopeHost, &kRopeDevice, kRopeSize);
    ReadFile(dataPath + "/k_rope.bin", kRopeHost, kRopeSize);
    ACL_CHECK(aclrtMemcpy(kRopeDevice, kRopeSize, kRopeHost, kRopeSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t* blockTableHost;
    uint8_t* blockTableDevice;
    AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
    ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in device memory for workspace.
    uint8_t* sDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&sDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* pDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&pDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(fp16_t) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oTmpDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&oTmpDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* globaloDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&globaloDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oDevice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&oDevice), static_cast<size_t>(qoSize), ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void* tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    MLATiling::MLAInfo mlaInfo;
    mlaInfo.numTokens = numTokens;
    mlaInfo.numHeads = numHeads;
    mlaInfo.embeddingSize = embeddingSize;
    mlaInfo.embeddingSizeRope = embeddingSizeRope;
    mlaInfo.numBlocks = numBlocks;
    mlaInfo.blockSize = blockSize;
    mlaInfo.maxKvSeqlen = maxKvSeqlen;
    mlaInfo.kvHeads = kvHeads;
    mlaInfo.batch = batch;
    mlaInfo.qSeqLen = static_cast<int32_t*>(qSeq);
    mlaInfo.kvSeqLen = static_cast<int32_t*>(kvSeq);
    MLATiling::GetMLATilingParam(mlaInfo, blockDim, (uint32_t*)tilingHost);

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t maxKvSplitCoreNum = *((uint32_t*)tilingHost + MLATiling::TILING_KVCORENUM);
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * maxKvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * maxKvSplitCoreNum * sizeof(float);

    uint8_t* oCoreTmpDevice;
    ACL_CHECK(aclrtMalloc((void**)(&oCoreTmpDevice), oFdSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* lDevice;
    ACL_CHECK(aclrtMalloc((void**)(&lDevice), lSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare hardware sync address
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    // use Tp1Spec kernel to get better performance when numHeads = 128
    switch (tilingKey) {
        case 0:
            MLA<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 1:
            MLA<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 4:
            AMLATp1Spec<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 5:
            AMLATp1Spec<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 7:
            MLATp1Spec<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 8:
            MLATp1Spec<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        default:
            break;
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // Copy the result from device to host
    vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
    vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16), (bfloat16)2.1);
    if (dataType == "half") {
        ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
    } else if (dataType == "bf16") {
        ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    // Compute the golden result
    vector<float> goldenHost(qoSize / sizeof(fp16_t));
    const size_t goldenSize = qoSize * 2;
    ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

    // Compute the cpulow result
    // cpu_low.bin is written in fp16 (same format as result.bin), NOT fp32 like golden.bin.
    // Read it as fp16 first, then widen to float for comparison.
    vector<float> cpulowHost(qoSize / sizeof(fp16_t));
    vector<fp16_t> cpulowFp16Tmp(qoSize / sizeof(fp16_t));
    const size_t cpulowSize = qoSize; // fp16 binary: qoSize bytes
    ReadFile(dataPath + "/cpu_low.bin", cpulowFp16Tmp.data(), cpulowSize);
    for (size_t i = 0; i < cpulowFp16Tmp.size(); ++i) {
        cpulowHost[i] = static_cast<float>(cpulowFp16Tmp[i]);
    }

    // Compute error metrics
    auto errorMetrics = (dataType == "half") ?
                            golden::ComputeErrorMetrics(oHostHalf, cpulowHost, goldenHost, 10.0, 2.0, 2.0) :
                            golden::ComputeErrorMetrics(oHostBf16, cpulowHost, goldenHost, 10.0, 2.0, 2.0);
    if (errorMetrics.passed) {
        cout << "Compare success." << endl;
    } else {
        cerr << "Error ratios exceed thresholds:" << endl;
        cerr << "MARE ratio: " << errorMetrics.mareRatio << " (threshold: 10)" << endl;
        cerr << "MERE ratio: " << errorMetrics.mereRatio << " (threshold: 2)" << endl;
        cerr << "RMSE ratio: " << errorMetrics.rmseRatio << " (threshold: 2)" << endl;
    }

    // Free host memory allocations.
    FreeMem(qHost, qDevice);
    FreeMem(qRopeHost, qRopeDevice);
    FreeMem(kHost, kDevice);
    FreeMem(kRopeHost, kRopeDevice);
    FreeMem(blockTableHost, blockTableDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTmpDevice);
    aclrtFree(globaloDevice);
    aclrtFree(oCoreTmpDevice);
    aclrtFree(lDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to mla example.

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
