/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

// Helper methods to check for errors
#include "fai_kernel.cpp"
#include "fai_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"

using namespace std;
using namespace optiling;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads qkHeadSize vHeadSize isVariedLen numblocks blocksize "
        "[--dtype DTYPE "
        "--cache_layout CACHE_LAYOUT --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 10;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t qkHeadSize{0};
    uint32_t vHeadSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    uint32_t cacheMode{0};
    uint32_t pageShape{0};
    uint32_t layout{0};
    uint32_t numBlocks{0};
    uint32_t innerPrec{0};
    uint32_t lseFlag{0};
    string dataType = "half";
    string cacheLayout = "nd";
    string dataPath = "../../examples/70_ascend950_flash_attention_chunk_prefill/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        // The number of arguments must >= 10.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        qkHeadSize = atoi(argv[argIndex++]);
        vHeadSize = atoi(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);
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
            } else if (flag == "--cache_layout") {
                cacheLayout = string(argv[argIndex++]);
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
// CATLASS FAI kernel.
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
    int32_t qSeqlen = options.qSeqlen;
    int32_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t qkHeadSize = options.qkHeadSize;
    int32_t vHeadSize = options.vHeadSize;
    int32_t blockSize = options.blockSize;
    int32_t numBlocks = options.numBlocks;
    int32_t maskType = 1;
    int32_t cacheMode = 1;
    int32_t pageShape = 1;
    int32_t layout = 1;
    int32_t innerPrec = 0;
    int32_t lseFlag = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    string cacheLayout = options.cacheLayout;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;

    int64_t requiredMinBlocks = static_cast<int64_t>(batch) * ((maxKvSeqlen + blockSize - 1) / blockSize);
    if (numBlocks < requiredMinBlocks) {
        cerr << "[ERROR] numBlocks (" << numBlocks << ") is less than required "
             << "batch * ceil(maxKvSeqlen / blockSize) = " << batch << " * ceil(" << maxKvSeqlen << " / " << blockSize
             << ") = " << requiredMinBlocks << endl;
        return;
    }

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }
    if ((cacheLayout != "nz") && (cacheLayout != "nd")) {
        cerr << "[ERROR] cacheLayout must be 'nz' or 'nd'." << endl;
        return;
    }

    if (qkHeadSize != 64 && qkHeadSize != 128 && qkHeadSize != 192) {
        cerr << "[ERROR] qkHeadSize must be 64, 128 or 192, got " << qkHeadSize << endl;
        return;
    }

    if (vHeadSize != 64 && vHeadSize != 128) {
        cerr << "[ERROR] vHeadSize must be 64 or 128, got " << vHeadSize << endl;
        return;
    }

    if (blockSize != 128 && blockSize != 256 && blockSize != 512 && blockSize != 1024) {
        cerr << "[ERROR] blockSize must be 128, 256, 512 or 1024, got " << blockSize << endl;
        return;
    }

    // read qNtokens num
    void* qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    void* kvNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
    int32_t kvNumTokens = static_cast<int32_t*>(kvNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t seqArraySizeTND = (batch + 1) * sizeof(int64_t);
    uint64_t qSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)qkHeadSize * sizeof(fp16_t);
    uint64_t oSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)vHeadSize * sizeof(fp16_t);
    uint64_t lseSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(int32_t);
    uint64_t kSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)qkHeadSize * sizeof(fp16_t);
    uint64_t vSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)vHeadSize * sizeof(fp16_t);
    uint64_t maskSize = 2048 * 2048 * sizeof(fp16_t);
    uint64_t blockTableSize =
        static_cast<uint64_t>(batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t));
    uint32_t tilingSize = sizeof(FAInferTilingData);

    // Allocate matrices in host and device memory.
    uint8_t* qSeqHost;
    uint8_t* qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySizeTND);
    ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySizeTND);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySizeTND, qSeqHost, seqArraySizeTND, ACL_MEMCPY_HOST_TO_DEVICE));
    // Allocate matrices in host and device memory.
    uint8_t* kvSeqHost;
    uint8_t* kvSeqDevice;
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t* qHost;
    uint8_t* qDevice;
    AllocMem(&qHost, &qDevice, qSize);
    ReadFile(dataPath + "/q.bin", qHost, qSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t* kHost;
    uint8_t* kDevice;
    AllocMem(&kHost, &kDevice, kSize);
    if (cacheLayout == "nd") {
        ReadFile(dataPath + "/k.bin", kHost, kSize);
    } else if (cacheLayout == "nz") {
        ReadFile(dataPath + "/k_nz.bin", kHost, kSize);
    }
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t* vHost;
    uint8_t* vDevice;
    AllocMem(&vHost, &vDevice, vSize);
    if (cacheLayout == "nd") {
        ReadFile(dataPath + "/v.bin", vHost, vSize);
    } else if (cacheLayout == "nz") {
        ReadFile(dataPath + "/v_nz.bin", vHost, vSize);
    }
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix mask.
    uint8_t* maskHost;
    uint8_t* maskDevice;
    AllocMem(&maskHost, &maskDevice, maskSize);
    ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
    ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t* blockTableHost;
    uint8_t* blockTableDevice;
    AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
    ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));

    bool enableDN = false;
    // 当前逻辑只支持half
    if (maskType == 0 && qkHeadSize <= 128 && vHeadSize <= 128 && innerPrec == 0) {
        enableDN = true;
    }

    FAInferTilingData faiTilingData;
    FAInferContext faiContext;

    faiContext.pagedCacheFlag = (cacheMode == 1);
    faiContext.pagedShapeFlag = (pageShape == 1);
    faiContext.kvcacheNzFlag = (cacheLayout == "nz");
    faiContext.numHeads = numHeads;
    faiContext.numBlocks = numBlocks;
    faiContext.blockSize = blockSize;
    faiContext.kvHeads = kvHeads;
    faiContext.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * qkHeadSize));
    faiContext.layout = "TND";
    faiContext.lseFlag = lseFlag;
    if (faiContext.pagedCacheFlag) {
        faiContext.maxNumBlocksPerBatch = (maxKvSeqlen + blockSize - 1) / blockSize;
    }
    faiContext.embeddingSize = qkHeadSize;
    faiContext.embeddingSizeV = vHeadSize;

    faiContext.maskType = static_cast<optiling::MaskType>(maskType);
    faiContext.dataType = static_cast<optiling::DataType>(dataType == "bf16");
    faiContext.batch = batch;
    faiContext.qSeqlenList = reinterpret_cast<int64_t*>(qSeqHost);
    faiContext.kvSeqlenList = reinterpret_cast<int64_t*>(kvSeqHost);
    faiContext.preToken = preToken;
    faiContext.nextToken = nextToken;
    cout << "preToken " << preToken << endl;
    cout << "nextToken " << nextToken << endl;
    cout << "cacheLayout " << cacheLayout << endl;
    cout << "maskType " << maskType << endl;

    // flashDecodeFlag determination
    int64_t maxQSeqlenCalc = 0;
    int64_t minQSeqlenCalc = INT64_MAX;
    int64_t minKVSeqlenCalc = INT64_MAX;
    for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int64_t qSeqlenVal = *(faiContext.qSeqlenList + batchIdx);
        int64_t kvSeqlenVal = *(faiContext.kvSeqlenList + batchIdx);
        if (faiContext.layout == "TND") {
            if (batchIdx > 0) {
                int64_t prevQSeqlenSum = *(faiContext.qSeqlenList + batchIdx - 1);
                qSeqlenVal = qSeqlenVal - prevQSeqlenSum;
            }
        }
        if (qSeqlenVal > maxQSeqlenCalc) {
            maxQSeqlenCalc = qSeqlenVal;
        }
        if (qSeqlenVal < minQSeqlenCalc) {
            minQSeqlenCalc = qSeqlenVal;
        }
        if (kvSeqlenVal < minKVSeqlenCalc) {
            minKVSeqlenCalc = kvSeqlenVal;
        }
    }
    faiContext.maxQSeqlen = maxQSeqlenCalc;
    uint32_t numTasks = faiContext.batch * faiContext.kvHeads;
    bool isLongSeq = (numTasks <= 0.8 * aicCoreNum) && (minKVSeqlenCalc >= aicCoreNum * 512);
    bool isShortSeq = (numTasks <= 0.4 * aicCoreNum) && (minKVSeqlenCalc >= 1024);
    faiContext.flashDecodeFlag = false;
    cout << "faiContext.flashDecodeFlag " << faiContext.flashDecodeFlag << endl;

    FAInferTiling fai_tiling(faiContext);
    fai_tiling.SetCoreNum(aicCoreNum);
    fai_tiling.DoTiling(faiTilingData);
    uint64_t tilingKey = fai_tiling.GetTilingKey();

    uint8_t* workspaceDevice{nullptr};
    faiTilingData.workSpaceSize = 1024 * 1024 * 32 * 4;
    cout << "faiTilingData.workSpaceSize " << faiTilingData.workSpaceSize << endl;
    ACL_CHECK(aclrtMalloc((void**)(&workspaceDevice), faiTilingData.workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oDevice{nullptr};
    cout << "oSize " << oSize << endl;
    ACL_CHECK(aclrtMalloc((void**)(&oDevice), oSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* lseDevice{nullptr};
    cout << "lseSize " << lseSize << endl;
    ACL_CHECK(aclrtMalloc((void**)(&lseDevice), lseSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* tilingDevice;
    cout << "tilingSize " << tilingSize << endl;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void* tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;
    if (faiContext.flashDecodeFlag) {
        auto needCoreNum = faiTilingData.get_needCoreNum();
        if (needCoreNum != 0) {
            blockDim = needCoreNum;
        }
    }

    // tiling output
    cout << "faiTilingData.numHeads" << faiTilingData.numHeads << endl;
    cout << "faiTilingData.embeddingSize" << faiTilingData.embeddingSize << endl;
    cout << "faiTilingData.embeddingSizeV" << faiTilingData.embeddingSizeV << endl;
    cout << "faiTilingData.numBlocks" << faiTilingData.numBlocks << endl;
    cout << "faiTilingData.blockSize" << faiTilingData.blockSize << endl;
    cout << "faiTilingData.maxQSeqlen" << faiTilingData.maxQSeqlen << endl;
    cout << "faiTilingData.maxKvSeqlen" << faiTilingData.maxKvSeqlen << endl;
    cout << "faiTilingData.kvHeads" << faiTilingData.kvHeads << endl;
    cout << "faiTilingData.batch" << faiTilingData.batch << endl;
    cout << "faiTilingData.maxNumBlocksPerBatch" << faiTilingData.maxNumBlocksPerBatch << endl;
    cout << "faiTilingData.totalTaskNum" << faiTilingData.totalTaskNum << endl;
    cout << "faiTilingData.maskType" << faiTilingData.maskType << endl;
    cout << "faiTilingData.qkOutSize" << faiTilingData.qkOutSize << endl;
    cout << "faiTilingData.smOnlineOutSize" << faiTilingData.smOnlineOutSize << endl;
    cout << "faiTilingData.pvOutSize" << faiTilingData.pvOutSize << endl;
    cout << "faiTilingData.UpdateSize" << faiTilingData.UpdateSize << endl;
    cout << "faiTilingData.workSpaceSize" << faiTilingData.workSpaceSize << endl;
    cout << "faiTilingData.scaleValue" << faiTilingData.scaleValue << endl;
    cout << "faiTilingData.firstBatchTaskNum" << faiTilingData.firstBatchTaskNum << endl;
    tilingHost = reinterpret_cast<void*>(&faiTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    cout << "tilingkey: " << tilingKey << endl;

    for (int i = 0; i < 1; i++) {
        if (cacheLayout == "nd") {
            if (dataType == "half") {
                FAInfer<
                    half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD,
                    MaskCategory::MASK_CAUSAL, CacheLayout::nd><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice, qSeqDevice,
                    kvSeqDevice, workspaceDevice, tilingDevice);
            } else if (dataType == "bf16") {
                FAInfer<
                    bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD,
                    MaskCategory::MASK_CAUSAL, CacheLayout::nd><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice, qSeqDevice,
                    kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else { // cacheLayout == "nz"
            if (dataType == "half") {
                FAInfer<
                    half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD,
                    MaskCategory::MASK_CAUSAL, CacheLayout::nz><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice, qSeqDevice,
                    kvSeqDevice, workspaceDevice, tilingDevice);
            } else if (dataType == "bf16") {
                FAInfer<
                    bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD,
                    MaskCategory::MASK_CAUSAL, CacheLayout::nz><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice, qSeqDevice,
                    kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(oSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(oSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        void* output = nullptr;
        aclrtMallocHost(&output, 128 * 128 * sizeof(fp16_t));
        aclrtMemcpy(
            output, sizeof(fp16_t) * 128 * 128, workspaceDevice, sizeof(fp16_t) * 128 * 128, ACL_MEMCPY_DEVICE_TO_HOST);

        // Compute the golden result
        vector<float> goldenHost(oSize / sizeof(fp16_t));
        const size_t goldenSize = oSize * 2;
        ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen) :
                                                               golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success." << endl;
        } else {
            cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
        }
    }
    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    FreeMem(maskHost, maskDevice);
    FreeMem(blockTableHost, blockTableDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(workspaceDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);

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
