/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#include <acl/acl.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstring>

#include "softmax_tiling.cpp"
#include "fag_common/common_header.h"

namespace FAGTiling {
// Host 侧传入 GetFATilingParam 的 FAG 反向基础信息。
// flash_api.cpp 会从 q/k/v/dout 的 shape 中填充这些字段，再生成 kernel 可直接读取的 tiling_data。
struct FAGInfo {
    float scaleValue;

    int64_t seqQShapeSize;
    int64_t queryShape_0;
    int64_t queryShape_1;
    int64_t queryShape_2;
    int64_t keyShape_0;
    int64_t keyShape_1;
    int64_t valueShape_0;
    int64_t valueShape_1;
};

// 调试辅助函数：按 common_header.h 中定义的 TILING_* 下标打印 tiling_data。
// 注意同一块 tilingHost 内存会按 int64_t、float、uint32_t 三种视角解释，因此访问下标需要乘 CONST_2。
void printFAGTilingData(int64_t *tilingHost) {
    std::cout << "FAGTilingDATA batch: " << tilingHost[TILING_B] << std::endl;
    std::cout << "FAGTilingDATA: total_q: " << tilingHost[TILING_T1] << std::endl;
    std::cout << "FAGTilingDATA: total_k: " << tilingHost[TILING_T2] << std::endl;
    std::cout << "FAGTilingDATA: nheads: " << tilingHost[TILING_N1] << std::endl;
    std::cout << "FAGTilingDATA: nheads_k: " << tilingHost[TILING_N2] << std::endl;
    std::cout << "FAGTilingDATA: G: " << tilingHost[TILING_G] << std::endl;
    std::cout << "FAGTilingDATA: headdim: " << tilingHost[TILING_D] << std::endl;
    std::cout << "FAGTilingDATA: q size: " << tilingHost[TILING_Q_SIZE] << std::endl;
    std::cout << "FAGTilingDATA: kv size: " << tilingHost[TILING_KV_SIZE] << std::endl;
    std::cout << "FAGTilingDATA: dq_workspace_offset: " << tilingHost[TILING_DQ_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: dk_workspace_offset: " << tilingHost[TILING_DK_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: dv_workspace_offset: " << tilingHost[TILING_DV_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: sfmg_workspace_offset: " << tilingHost[TILING_SFMG_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: mm1_workspace_offset: " << tilingHost[TILING_MM1_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: mm2_workspace_offset: " << tilingHost[TILING_MM2_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: p_workspace_offset: " << tilingHost[TILING_P_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: ds_workspace_offset: " << tilingHost[TILING_DS_WORKSPACE_OFFSET] << std::endl;

    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    std::cout << "FAGTilingDATA scale value: " << tilingHostFp[TILING_SCALE_VALUE * CONST_2] << std::endl;
    
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    std::cout << "FAGTilingDATA coreNum: " << tilingHostU32[TILING_CORE_NUM * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 1] << std::endl;
    std::cout << "FAGTilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 2] << std::endl;
    std::cout << "FAGTilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 3] << std::endl;
    std::cout << "FAGTilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 4] << std::endl;
    std::cout << "FAGTilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 5] << std::endl;
    std::cout << "FAGTilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 6] << std::endl;
    std::cout << "FAGTilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 7] << std::endl;
    std::cout << "FAGTilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 8] << std::endl;
    std::cout << "FAGTilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 9] << std::endl;
    std::cout << "FAGTilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 10] << std::endl;
    std::cout << "FAGTilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 11] << std::endl;
    std::cout << "FAGTilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 12] << std::endl;
    std::cout << "FAGTilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 13] << std::endl;
    std::cout << "FAGTilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 14] << std::endl;
    std::cout << "FAGTilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 15] << std::endl;

    // softmax grad data
    std::cout << "FAGTilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 1] << std::endl;
    std::cout << "FAGTilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 2] << std::endl;
    std::cout << "FAGTilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 3] << std::endl;
    std::cout << "FAGTilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 4] << std::endl;
    std::cout << "FAGTilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 5] << std::endl;
    std::cout << "FAGTilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 6] << std::endl;
    std::cout << "FAGTilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 7] << std::endl;
    std::cout << "FAGTilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 8] << std::endl;
    std::cout << "FAGTilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 9] << std::endl;
    std::cout << "FAGTilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 10] << std::endl;
    std::cout << "FAGTilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 11] << std::endl;
    std::cout << "FAGTilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 12] << std::endl;
    std::cout << "FAGTilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 13] << std::endl;
    std::cout << "FAGTilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 14] << std::endl;
    std::cout << "FAGTilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 15] << std::endl;
}

// 生成 FAG backward kernel 使用的 tiling_data。
// 该函数运行在 Host 侧：负责写入 shape、GQA 分组、softmax tiling、core 数和 workspace 分区偏移。
int32_t GetFATilingParam(const FAGInfo fagInfo, uint32_t &blockDim, int64_t *tilingHost)
{
    // scaleValue 是 float 字段，但 tilingHost 主体按 int64_t 分配，因此这里按 float 视角写入。
    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    tilingHostFp[TILING_SCALE_VALUE * CONST_2] = fagInfo.scaleValue;

    // 基础 shape 信息：T1/T2 是展平后的总 Q/K token 数，N1/N2 是 Q/KV 头数，D 是 headdim。
    tilingHost[TILING_B] = fagInfo.seqQShapeSize;
    tilingHost[TILING_T1] = fagInfo.queryShape_0;
    tilingHost[TILING_T2] = fagInfo.keyShape_0;
    tilingHost[TILING_N1] = fagInfo.queryShape_1;
    tilingHost[TILING_N2] = fagInfo.keyShape_1;
    tilingHost[TILING_D] = fagInfo.queryShape_2;

    // GQA/MQA 分组数：每个 KV head 对应 g 个 query head。
    uint64_t g = fagInfo.queryShape_1 / fagInfo.keyShape_1;
    tilingHost[TILING_G] = fagInfo.queryShape_1 / fagInfo.keyShape_1;

    // qSize/kvSize 是 dq/dk/dv fp32 workspace 的元素数，不是字节数。
    // qSize 等价于 total_q * nheads * headdim；kvSize 等价于 total_k * nheads_k * headdim。
    int64_t qSize = fagInfo.queryShape_0 * fagInfo.keyShape_1 * g * fagInfo.queryShape_2;
    int64_t kvSize = fagInfo.keyShape_0 * fagInfo.keyShape_1 * 1 * fagInfo.queryShape_2;
    // sfmg 每个 Q token/head 行保存 8 个 float，用于 32B 对齐的 sum(dout * out) 行归约结果。
    int64_t sfmgSize = fagInfo.queryShape_0 * fagInfo.queryShape_1 * 8;

    tilingHost[TILING_Q_SIZE] = qSize;
    tilingHost[TILING_KV_SIZE] = kvSize;

    // Vector epilogue 中 softmax 重算的 tiling。
    // 这里固定以 64x128 的块形状估算 UB 使用量，kernel 侧再按实际有效长度和 mask 处理边界。
    constexpr uint32_t tmpBufferSize = 33 * 1024;
    constexpr uint32_t s1VecSize = 64;
    constexpr uint32_t s2VecSize = 128;
    std::vector<uint32_t> softmaxShape = {s1VecSize, s2VecSize};

    SoftMaxTiling softmaxTilingData;
    SoftMaxTilingFunc(
        softmaxShape, sizeof(float), tmpBufferSize, softmaxTilingData);

    // FAGSfmg 阶段的 SoftmaxGradFront tiling，用于计算每行 sum(dout * out)。
    constexpr uint32_t inputBufferLen = 24 * 1024;
    constexpr uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    // 输出每行固定 8 个 float，因此按 headdim 反推当前 castBuffer 能容纳的行数后乘以 8。
    uint32_t outputBufferLen = (castBufferLen + fagInfo.queryShape_2 - 1) /  fagInfo.queryShape_2 * 8;
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    // singleLoopNBurstNum 表示 sfmg 每轮最多处理多少个 token/head 行。
    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / fagInfo.queryShape_2;
    std::vector<int64_t> softmaxGradShape = {singleLoopNBurstNum, fagInfo.queryShape_2};

    SoftMaxTiling softmaxGradTilingData;
    SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, 
        softmaxGradTilingData);

    // 写入 softmax/softmaxGrad tiling 数据。
    // coreNum 用于 Cube 侧 per-AIC workspace 大小计算；vectorCoreNum 写入 tiling，供 Vector/Pre/Post 按 AIV 数切分任务。
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t vectorCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    tilingHostU32[TILING_CORE_NUM * CONST_2] = vectorCoreNum;

    // SoftMaxTiling 由 32-bit 字段组成，写入 int64_t tilingHost 时同样需要按 uint32_t 视角拷贝。
    uint32_t* softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);
    softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxGradTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);

    // workspace 内部按 byte offset 连续切分。
    // 前 16MB 预留给运行时/系统用途，后续各子 workspace 都按 512B 对齐，方便 GM 访问和 DMA 搬运。
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    // Cube1/2/3 每个 AIC 的单拍矩阵中间块：16 * 128 * 128 个元素；DB_NUM 表示 ping-pong 双缓冲。
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t workspaceOffset = WORKSPACE_RSV_BYTE;
    // dq/dk/dv 是最终梯度的 fp32 累加 workspace，Pre 先清零，Cube2/Cube3 atomic add 写入，Post 再 cast 到输出 dtype。
    tilingHost[TILING_DQ_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + qSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // dk workspace，元素数为 kvSize。
    tilingHost[TILING_DK_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // dv workspace，元素数同样为 kvSize。
    tilingHost[TILING_DV_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // sfmg workspace 保存每个 Q token/head 行的 8-float softmax grad front 归约值。
    tilingHost[TILING_SFMG_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + sfmgSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // mm1/mm2 是 Cube1 输出的 fp32 中间矩阵：mm1=dOut*V^T，mm2=Q*K^T。
    // 按 AIC 数和双缓冲预留空间，Vector epilogue 会消费它们生成 P/dS。
    tilingHost[TILING_MM1_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_MM2_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    constexpr uint32_t size_of_half = 2;
    // p/ds workspace 是 Vector epilogue 写给 Cube2/Cube3 的低精度中间矩阵。
    // p 用于 dV=P^T*dOut，ds 用于 dQ=dS*K 和 dK=dS^T*Q。
    tilingHost[TILING_P_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_DS_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    return 0;
}

} // namespace FAGTiling