/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * softmax_tiling.cpp —— FlashAttention NPU 反向传播 Softmax/SoftmaxGrad 的 Host 侧 Tiling 计算
 * ============================================================================
 *
 * 【文件定位】
 *   本文件在 Host 侧（CPU 端）运行，为 CANN AscendC softmax/softmax_grad 算子计算
 *   UB（Unified Buffer）分块参数（Tiling），输出 SoftMaxTiling 结构体供 Device 侧
 *   kernel 使用。它不运行在 NPU 上，仅在 kernel launch 前由 Host API 调用。
 *
 *   ⚠️ 注意：本文件是 Host 侧 tiling 辅助代码，**不是** Device 侧 kernel 实现。
 *   - 前向 FlashAttention 的 softmax 使用 online_softmax.hpp（Device 侧，全在 NPU 上）
 *   - 反向传播（mha_bwd / mha_varlen_bwd）调用 CANN 内置 softmax/softmax_grad 算子，
 *     需要 Host 侧先算好 tiling 参数传给 kernel。
 *
 * 【编译方式】
 *   本文件通过 #include 被 fag_tiling.cpp 文本包含，而 fag_tiling.cpp 又被
 *   flash_api.cpp 包含，最终由 bisheng 编译器以 unity build（单编译单元）方式编译。
 *   setup.py 中只将 flash_api.cpp 作为源文件传给编译器。
 *
 * 【调用链】
 *   flash_api.cpp (mha_bwd / mha_varlen_bwd 两个反向 Host API)
 *     └── FAGTiling::GetFATilingParam()  [fag_tiling.cpp]
 *           ├── SoftMaxTilingFunc()       ← 本文件，softmax 前向 tiling
 *           └── SoftMaxGradTilingFunc()   ← 本文件，softmax 反向 tiling
 *
 * 【核心问题：Tiling 解决什么】
 *   Softmax 需要对每一行做 max→exp→sum→div，但 UB 容量有限（典型 ~33-40KB），
 *   当行数 srcM 很大时无法一次性放入 UB，需要沿行维 M 切分成多个 split：
 *
 *   srcM 行 × srcK 列的矩阵:
 *     ┌────────────────────┐  ← baseM 行（split 0）
 *     │                    │
 *     ├────────────────────┤  ← baseM 行（split 1）
 *     │                    │
 *     ├────────────────────┤
 *     │        ...         │  rangeM 个完整 split + 1 个 tail split
 *     ├────────────────────┤
 *     │                    │  ← tailM 行（tail split，若 srcM%baseM≠0）
 *     └────────────────────┘
 *
 *   Tiling 的核心是根据 UB 容量(localWorkSpaceSize)和数据类型(dataTypeSize)
 *   计算出最优的 baseM（每 split 的行数），并输出所有分块参数。
 *
 * 【Tiling 输出字段（SoftMaxTiling 共 16 个 uint32_t 字段）】
 *   - srcM/srcK/srcSize: 输入矩阵形状（展平为 2D 后）
 *   - outMaxM/outMaxK/outMaxSize: Reduce 阶段输出最大形状
 *   - splitM/splitK/splitSize: 单个 split 的形状
 *   - reduceM/reduceK/reduceSize: reduce 阶段分块形状
 *   - rangeM: 完整 split 数量
 *   - tailM: 尾 split 行数
 *   - tailSplitSize/tailReduceSize: 尾 split 大小
 *
 * 【常量说明】
 *   - SOFTMAX_DEFAULT_BLK_SIZE=32: 默认块大小(字节)
 *   - SOFTMAX_TMPBUFFER_COUNT=2: softmax 前向临时缓冲数
 *   - SOFTMAX_HALF_SIZE=2: half 类型字节数
 *   - SOFTMAX_FLOAT_SIZE=4: float 类型字节数
 *   - SOFTMAXGRAD_TMPBUFFER_COUNT=3: softmax 反向临时缓冲数
 *   - BASIC_TILE_NUM=8: 基础 tile 元素数（32B/4B=8个float）
 *   - SOFTMAX_BASICBLOCK_MIN_SIZE=256: basicblock 最小尺寸
 *   - SOFTMAX_BASICBLOCK_UNIT=64: basicblock 对齐单位
 * ============================================================================
 */

#include <vector>
#include <iomanip>

// ======================== 编译期常量 ========================
constexpr uint32_t SOFTMAX_DEFAULT_BLK_SIZE = 32;           // 默认块大小（字节），即 32B = 一个向量寄存器宽度
constexpr uint32_t SOFTMAX_TMPBUFFER_COUNT = 2;             // softmax 前向临时缓冲数（split 缓冲 + reduce 缓冲）
constexpr uint32_t SOFTMAX_HALF_SIZE = 2;                   // half/bf16 元素字节数
constexpr uint32_t SOFTMAX_FLOAT_SIZE = 4;                  // float 元素字节数
constexpr uint32_t SOFTMAXGRAD_TMPBUFFER_COUNT = 3;         // softmax 反向临时缓冲数（比前向多1个）
constexpr uint32_t BASIC_TILE_NUM = SOFTMAX_DEFAULT_BLK_SIZE / SOFTMAX_FLOAT_SIZE;  // 基础 tile 数=32B/4B=8个float
constexpr uint32_t SOFTMAX_BASICBLOCK_MIN_SIZE = 256;       // basicblock 模式最小尺寸约束
constexpr uint32_t SOFTMAX_BASICBLOCK_UNIT = 64;            // basicblock 对齐单位（元素数）

/**
 * @brief SoftMaxTiling 的本地镜像结构体
 *
 * SoftMaxTiling 类型定义在 CANN toolchain 外部头文件中（链接 -ltiling_api），
 * 本结构体字段顺序和类型与外部 SoftMaxTiling 完全一致，共16个uint32_t字段(64字节)，
 * 仅用于本地理解，实际 tiling 函数直接写入外部 SoftMaxTiling 引用。
 *
 * 字段含义：
 *   ┌─────────────┬─────────────────────────────────────────────┐
 *   │ srcM/srcK   │ 展平后输入矩阵的行数/列数(srcK=最后一维)      │
 *   │ srcSize     │ srcM × srcK（总元素数）                      │
 *   │ outMaxM/K   │ reduce阶段输出最大行数/列数(outMaxK=每块列数) │
 *   │ splitM/K    │ 单次split处理的行数/列数(splitK=srcK整行)    │
 *   │ reduceM/K   │ reduce阶段分块的行数/列数(reduceK=块宽)      │
 *   │ rangeM      │ 完整split个数（srcM/baseM）                  │
 *   │ tailM       │ 尾块行数（srcM%baseM，0表示无尾块）          │
 *   │ tail*Size   │ 尾块元素数（tailM*K）                        │
 *   └─────────────┴─────────────────────────────────────────────┘
 */
struct SoftMaxTilingLocal
{
    uint32_t srcM = 0;              // 输入矩阵行数（展平后，= 除最后一维外各维乘积）
    uint32_t srcK = 0;              // 输入矩阵列数（= 最后一维大小，即softmax归约轴长度）
    uint32_t srcSize = 0;           // 输入总元素数 = srcM × srcK
    uint32_t outMaxM = 0;           // reduce 输出最大行数（= srcM）
    uint32_t outMaxK = 0;           // reduce 输出块列数（= elementNumPerBlk）
    uint32_t outMaxSize = 0;        // reduce 输出最大元素数 = srcM × elementNumPerBlk
    uint32_t splitM = 0;            // 单次 split 处理的行数（= baseM，核心参数）
    uint32_t splitK = 0;            // 单次 split 处理的列数（= srcK，整行处理）
    uint32_t splitSize = 0;         // 单次 split 元素数 = baseM × srcK
    uint32_t reduceM = 0;           // reduce 阶段行数（= baseM）
    uint32_t reduceK = 0;           // reduce 阶段列数（= elementNumPerBlk）
    uint32_t reduceSize = 0;        // reduce 阶段元素数 = baseM × elementNumPerBlk
    uint32_t rangeM = 0;            // 完整 split 数量 = srcM / baseM
    uint32_t tailM = 0;             // 尾 split 行数 = srcM % baseM（0=无尾块）
    uint32_t tailSplitSize = 0;     // 尾 split 数据元素数 = tailM × srcK
    uint32_t tailReduceSize = 0;    // 尾 split reduce 元素数 = tailM × elementNumPerBlk
};

/**
 * @brief 将 N-D shape 展平为 2D {srcM, srcK}（softmax 归约沿最后一维）
 *
 * softmax 操作沿张量最后一维做归约（列维K），前面所有维展平为行维M。
 * 例如 shape = {B, H, S}（batch × heads × seqlen）:
 *   srcK = S（最后一维 = softmax 轴）
 *   srcM = B × H（展平行数）
 *
 * @tparam T 形状元素类型（uint32_t 或 int64_t，兼容两种调用）
 * @param srcShape 输入 N-D 形状
 * @return std::vector<uint32_t> {srcM, srcK}，2 元素向量
 */
template<typename T>
inline std::vector<uint32_t> GetLastAxisShapeND(const std::vector<T> srcShape)
{
    std::vector<uint32_t> ret;
    std::vector<int64_t> shapeDims(srcShape.begin(), srcShape.end());
    uint32_t calculateSize = 1;
    // 计算所有维度的总元素数
    for (uint32_t i = 0; i < shapeDims.size(); i++) {
        calculateSize *= shapeDims[i];
    }

    const uint32_t srcK = shapeDims.back();     // 最后一维 = softmax 归约轴 K
    uint32_t srcM = calculateSize / srcK;       // 其余维展平为行 M
    ret = { srcM, srcK };
    return ret;
}

/**
 * @brief 将 baseM 调整为 basicblock 模式对齐值
 *
 * basicblock 模式条件：baseM > 8 且 srcM 能被 8 整除 且 srcK 能被 64 整除。
 * 满足条件时执行三步调整：
 *   1. 向下对齐到 BASIC_TILE_NUM(8) 的倍数
 *   2. 逐步减 8，直到 baseM 能整除 srcM（保证完整分块无尾块或尾块整齐）
 *   3. 如果 baseM × srcK >= 64×256=16384（硬件 repeat 最大 255 限制），则不断减半
 *
 * @param baseM [in/out] 待调整的每 split 行数
 * @param srcM 总行数
 * @param srcK 列数（softmax 轴）
 */
inline void AdjustToBasicBlockBaseM(uint32_t& baseM, const uint32_t srcM, const uint32_t srcK)
{
    if (baseM > BASIC_TILE_NUM && srcM % BASIC_TILE_NUM == 0 && srcK % SOFTMAX_BASICBLOCK_UNIT == 0) { // basicblock
        // 步骤1：向下对齐到 8 的倍数（BASIC_TILE_NUM=8）
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
        // 步骤2：逐步减 8，直到 baseM 能整除 srcM（保证 split 均匀划分）
        while (srcM % baseM != 0) {
            baseM -= BASIC_TILE_NUM;
        }
        // 步骤3：硬件限制——指令 repeat 次数最大支持 255，
        //        baseM*srcK（单 split 元素数）若超过 64*256=16384 则减半
        while (baseM * srcK >= SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_BASICBLOCK_MIN_SIZE) {
            baseM = baseM / SOFTMAX_HALF_SIZE;
        }
    }
}

/**
 * @brief Softmax 前向 tiling 计算（Host 侧）
 *
 * 根据输入形状、数据类型大小、UB workspace 大小，计算 softmax 前向算子
 * 在 NPU 上的分块参数。
 *
 * 算法流程：
 *   1. 将 N-D shape 展平为 2D {srcM, srcK}（softmax 沿最后一维 K）
 *   2. 计算 elementNumPerBlk = 32 / dataTypeSize（每块元素数：float=8, half=16）
 *   3. workLocalSize = localWorkSpaceSize / 4（workspace 字节数→float 元素数）
 *   4. 估算 baseM = workLocalSize / (elementNumPerBlk + srcK + 64)
 *      其中分母是"每处理 1 行所需的 float 元素缓冲"：
 *        - elementNumPerBlk: reduce 阶段临时缓冲（max/sum 中间结果）
 *        - srcK: split 阶段行数据缓冲（1行 srcK 个元素）
 *        - 64 (SOFTMAX_BASICBLOCK_UNIT): 对齐余量
 *   5. baseM 向下对齐到 8 的倍数（若 baseM < srcM 且 > 8）
 *   6. AdjustToBasicBlockBaseM: basicblock 模式对齐调整
 *   7. 计算所有输出字段（split/reduce/range/tail 等）
 *
 * @param srcShape            输入张量 N-D 形状（最后一维为 softmax 轴）
 * @param dataTypeSize        数据类型字节数（float=4, half=2）
 * @param localWorkSpaceSize  UB workspace 大小（字节）
 * @param softmaxTiling       [out] 输出 tiling 参数（外部 CANN SoftMaxTiling 结构体）
 */
void SoftMaxTilingFunc(const std::vector<uint32_t>& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    SoftMaxTiling& softmaxTiling)
{
    // 步骤1：展平 N-D 为 2D {srcM, srcK}
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return;
    }
    // 步骤2：每块元素数（32B 块内能放几个元素）
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    // 步骤3：workspace 大小转换为 float 元素数
    const uint32_t workLocalSize = localWorkSpaceSize / SOFTMAX_FLOAT_SIZE;
    const uint32_t srcK = retVec[1];       // 列数 = softmax 轴
    const uint32_t srcM = retVec[0];       // 行数（展平后）
    // 步骤4：估算每行所需缓冲，计算 baseM（每行需要 reduce缓冲+数据缓冲+对齐余量）
    uint32_t baseM = std::min(workLocalSize / (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT), srcM);
    // 步骤5：若 baseM 未覆盖全部行且 >8，则向下对齐到 8 的倍数
    if (baseM < srcM && baseM > BASIC_TILE_NUM) {
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
    }

    // 步骤6：basicblock 模式对齐调整
    AdjustToBasicBlockBaseM(baseM, srcM, srcK);

    // 步骤7：填充输出 tiling 字段
    softmaxTiling.srcM = srcM;
    softmaxTiling.srcK =srcK;
    softmaxTiling.srcSize = srcM * srcK;

    softmaxTiling.outMaxM = srcM;
    softmaxTiling.outMaxK = elementNumPerBlk;   // reduce 块宽 = 每块元素数
    softmaxTiling.outMaxSize = srcM * elementNumPerBlk;

    softmaxTiling.splitM = baseM;               // 单次 split 行数
    softmaxTiling.splitK = srcK;                // 单次 split 列数 = 整行
    softmaxTiling.splitSize = baseM * srcK;

    softmaxTiling.reduceM = baseM;              // reduce 行数 = baseM
    softmaxTiling.reduceK = elementNumPerBlk;   // reduce 块宽
    softmaxTiling.reduceSize = baseM * elementNumPerBlk;

    // 完整 split 数和尾块行数
    const uint32_t range = srcM / baseM;
    uint32_t tail = srcM % baseM;
    softmaxTiling.rangeM = range;
    softmaxTiling.tailM = tail;

    softmaxTiling.tailSplitSize = tail * srcK;
    softmaxTiling.tailReduceSize = tail * elementNumPerBlk;
}

/**
 * @brief Softmax 反向（SoftmaxGrad）tiling 计算（Host 侧）
 *
 * 与 SoftMaxTilingFunc 逻辑类似，但反向传播需要更多临时缓冲：
 *   - 前向 softmax 需要 2 个临时缓冲（SOFTMAX_TMPBUFFER_COUNT=2）：
 *     reduce 结果 + split 数据
 *   - 反向 softmax_grad 在 half 精度下需要 3 个临时缓冲（SOFTMAXGRAD_TMPBUFFER_COUNT=3）：
 *     因为反向计算 dX = Y * (dY - sum(dY*Y)) 需要额外缓冲 dY*Y 中间结果
 *
 *   ⚠️ 当前调用（fag_tiling.cpp）始终传入 sizeof(float)=4 作为 dataTypeSize，
 *      因此走 else 分支（与前向公式相同）。half 分支（dataTypeSize==2）为预留路径。
 *
 * @param srcShape            输入张量 N-D 形状（int64_t 版本，因为反向 shape 来自 int64_t 张量维度）
 * @param dataTypeSize        数据类型字节数（当前调用始终=4即float）
 * @param localWorkSpaceSize  UB workspace 大小（字节），约 37KB
 * @param softmaxGradTiling   [out] 输出 tiling 参数
 */
void SoftMaxGradTilingFunc(const std::vector<int64_t>& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    SoftMaxTiling& softmaxGradTiling)
{
    // 步骤1：展平 N-D 为 2D {srcM, srcK}
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return;
    }
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    const uint32_t workLocalSize = localWorkSpaceSize / SOFTMAX_FLOAT_SIZE;
    const uint32_t srcK = retVec[1];
    const uint32_t srcM = retVec[0];
    uint32_t baseM = 0;

    // 步骤4：根据数据类型选择不同的 workspace 计算公式
    if (dataTypeSize == SOFTMAX_HALF_SIZE) {
        // half 精度反向：需要 3 个缓冲
        //   elementNumPerBlk × 2：2 个 reduce 临时缓冲
        //   srcK × 3：3 个行数据缓冲（Y, dY, dY*Y）
        //   +64：对齐余量
        baseM = workLocalSize /
            (elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT + srcK * SOFTMAXGRAD_TMPBUFFER_COUNT + SOFTMAX_BASICBLOCK_UNIT);
    } else {
        // float 精度（当前实际使用路径）：与前向公式相同
        //   elementNumPerBlk × 1：1 个 reduce 缓冲
        //   srcK × 1：1 个行数据缓冲
        //   +64：对齐余量
        baseM = workLocalSize / (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT);
    }

    // baseM 不超过总行数
    baseM = std::min(baseM, srcM);
    // 步骤5：对齐到 8 的倍数
    if (baseM < srcM && baseM > BASIC_TILE_NUM) {
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
    }

    // 步骤6：basicblock 模式对齐调整
    AdjustToBasicBlockBaseM(baseM, srcM, srcK);

    // 步骤7：填充输出 tiling 字段（与 SoftMaxTilingFunc 结构相同）
    softmaxGradTiling.srcM = srcM;
    softmaxGradTiling.srcK = srcK;
    softmaxGradTiling.srcSize = srcM * srcK;

    softmaxGradTiling.outMaxM = srcM;
    softmaxGradTiling.outMaxK = elementNumPerBlk;
    softmaxGradTiling.outMaxSize = srcM * elementNumPerBlk;

    softmaxGradTiling.splitM = baseM;
    softmaxGradTiling.splitK = srcK;
    softmaxGradTiling.splitSize = baseM * srcK;

    softmaxGradTiling.reduceM = baseM;
    softmaxGradTiling.reduceK = elementNumPerBlk;
    softmaxGradTiling.reduceSize = baseM * elementNumPerBlk;

    uint32_t range = srcM / baseM;
    const uint32_t tail = srcM % baseM;
    softmaxGradTiling.rangeM = range;
    softmaxGradTiling.tailM = tail;

    softmaxGradTiling.tailSplitSize = tail * srcK;
    softmaxGradTiling.tailReduceSize = tail * elementNumPerBlk;
}

/**
 * @brief 打印 SoftMaxTiling 各字段值（调试用，当前无任何调用方——死代码）
 *
 * 类似函数 printFAGTilingData 在 flash_api.cpp 中的调用也被注释掉，
 * 保留此函数是为了开发调试时可临时启用打印 tiling 参数。
 *
 * @param softmaxTilingData 待打印的 tiling 结构体
 */
void printSoftmaxTilingData(SoftMaxTiling &softmaxTilingData) {
    std::cout << "softmaxTilingData srcM:  " << softmaxTilingData.srcM << std::endl;
    std::cout << "softmaxTilingData srcK:  " << softmaxTilingData.srcK << std::endl;
    std::cout << "softmaxTilingData srcSize:  " << softmaxTilingData.srcSize << std::endl;
    std::cout << "softmaxTilingData outMaxK:  " << softmaxTilingData.outMaxK << std::endl;
    std::cout << "softmaxTilingData outMaxM:  " << softmaxTilingData.outMaxM << std::endl;
    std::cout << "softmaxTilingData: outMaxSize  " << softmaxTilingData.outMaxSize << std::endl;
    std::cout << "softmaxTilingData: splitM  " << softmaxTilingData.splitM << std::endl;
    std::cout << "softmaxTilingData: splitK  " << softmaxTilingData.splitK << std::endl;
    std::cout << "softmaxTilingData: splitSize  " << softmaxTilingData.splitSize << std::endl;
    std::cout << "softmaxTilingData: reduceM  " << softmaxTilingData.reduceM << std::endl;
    std::cout << "softmaxTilingData: reduceK  " << softmaxTilingData.reduceK << std::endl;
    std::cout << "softmaxTilingData: reduceSize  " << softmaxTilingData.reduceSize << std::endl;
    std::cout << "softmaxTilingData: rangeM  " << softmaxTilingData.rangeM << std::endl;
    std::cout << "softmaxTilingData: tailM  " << softmaxTilingData.tailM << std::endl;
    std::cout << "softmaxTilingData: tailSplitSize  " << softmaxTilingData.tailSplitSize << std::endl;
    std::cout << "softmaxTilingData: tailReduceSize  " << softmaxTilingData.tailReduceSize << std::endl;
}