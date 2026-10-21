/**
 * SplitCore core-split and FlashDecode decision, ported from
 * ops-transformer attention/common/op_host/split_core.
 * Header-only so the host tiling path and the AICPU metadata kernel share it.
 */
#ifndef FA_SPLIT_CORE_H
#define FA_SPLIT_CORE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

namespace fa_split_core {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
    TREE = 9,
};

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

template <typename T>
T Clip(T value, T minVal, T maxVal)
{
    if (value < minVal) {
        return minVal;
    }
    if (value > maxVal) {
        return maxVal;
    }
    return value;
}

// 分核功能模块输入：输入case的基本信息
struct BaseInfo {
    uint32_t bSize{0U};
    uint32_t n2Size{0U};
    uint32_t gSize{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    bool isAccumSeqS1{false};
    bool isAccumSeqS2{false};
    bool isS1G{true};
    uint32_t actualLenQDims{0U};
    uint32_t actualLenKvDims{0U};
    bool attenMaskFlag{false};
    int32_t sparseMode{0U};
    int64_t preToken{0};
    int64_t nextToken{0};
    int64_t actualSeqPrefixSize{0};
    std::vector<int64_t> actualSeqS1Size{};
    std::vector<int64_t> actualSeqS2Size{};
};

// 分核功能模块输入：切分属性，预留接口，可作为切分方案的参数入口
struct SplitParam {
    uint32_t mBaseSize{1U};
    uint32_t s2BaseSize{1U};
    uint32_t gS1BaseSizeOfFd{8U}; // FD阶段分核，m轴切分基本块大小
};

// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2IdxOfFdHead{}; // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> gS1IdxOfFdHead{}; // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> s2SplitNumOfFdHead{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> gS1SplitNumOfFdHead{}; // 每个归约任务m轴切分份数，脚标为归约任务的序号
    std::vector<uint32_t> gS1LastPartSizeOfFdHead{}; // 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
    std::vector<uint32_t>
        gS1IdxEndOfFdHead{}; // FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
    std::vector<uint32_t>
        gS1IdxEndOfFdHeadSplit{}; // FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
    // 3、每个core处理的第1个归约任务的数据应存放的workspace位置
    std::vector<uint32_t> s2SplitStartIdxOfCore{};

    FlashDecodeResult(uint32_t coreNum, uint32_t vecCubeRatio)
        : bN2IdxOfFdHead(coreNum),
          gS1IdxOfFdHead(coreNum),
          s2SplitNumOfFdHead(coreNum),
          gS1SplitNumOfFdHead(coreNum),
          gS1LastPartSizeOfFdHead(coreNum),
          gS1IdxEndOfFdHead(coreNum * vecCubeRatio),
          gS1IdxEndOfFdHeadSplit(coreNum * vecCubeRatio),
          s2SplitStartIdxOfCore(coreNum)
    {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum{0U};        // 使用的核数量
    uint32_t vecCubeRatio{0U};       // vec 与 cube 核数比例
    std::vector<uint32_t> bN2End{};  // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End{};  // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End{};   // 每个核处理数据的S2结束点
    int64_t maxCost{0};              // 慢核开销
    uint32_t numOfFdHead{0U};        // 归约任务数量
    uint32_t maxS2SplitNum{0U};      // 单个归约任务最大分核数量
    uint32_t usedVecNumOfFd{0U};     // 归约过程使用的vector数量
    FlashDecodeResult fdRes{0U, 0U}; // FD信息

    SplitResult(uint32_t coreNum, uint32_t ratio)
        : vecCubeRatio(ratio),
          bN2End(coreNum),
          gS1End(coreNum),
          s2End(coreNum),
          fdRes(coreNum, ratio) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum{};  // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum{};   // S2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize{}; // S1G方向，尾块size
    std::vector<uint32_t> s2TailSize{};  // S2方向，尾块size
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : s1GBaseNum(batchSize),
          s2BaseNum(batchSize),
          s1GTailSize(batchSize),
          s2TailSize(batchSize)
    {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};          // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch{};        // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{}; // batch最后一块的开销
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize),
          bN2BlockOfEachBatch(batchSize),
          bN2LastBlockCostOfEachBatch(batchSize)
    {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    const BaseInfo &baseInfo{0U};
    const SplitParam &splitParam{};
    CostInfo costInfo{0U};
    SplitInfo splitInfo{0U};

    explicit SplitContext(const BaseInfo &info, const SplitParam &param)
        : baseInfo(info),
          splitParam(param),
          costInfo(info.bSize),
          splitInfo(info.bSize)
    {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    int64_t nextTokenLeftUp{0};
    int64_t preTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit{0}; // 负载上限
    int64_t cost{0};      // 已分配负载
    uint32_t block{0U};   // 已分配块数
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx{0U};
    uint32_t s1GIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    int64_t s1GCost{0};
    uint32_t s1GBlock{0U};
    int64_t s1GNormalBlockCost{0};
    int64_t s1GLastBlockCost{0};
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curCoreIdx{0U};
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curS1GIdx{0U};
    uint32_t curS2Idx{0U};
    int64_t unassignedCost{0};
    uint32_t usedCoreNum{0U};
    uint32_t curKvSplitPart{1U};
    uint32_t bN2Block{0U};
    int64_t bN2Cost{0};
    bool isFinished{false};
    BatchCache batchCache{};
    S1GCache s1GCache{};
    CoreCache coreCache{};
};

// util
inline uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
inline uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
inline int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
inline int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
inline Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BatchCache &batchCache);
inline int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
inline BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
                                 uint32_t s2TailSize);

// cache calculation
inline void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
inline void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);
inline void CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes);
inline void ClearTmpResult(SplitResult &tmpResult);

// preprocess
inline void CalcSplitInfo(SplitContext &splitContext);
inline void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
inline void CalcCostInfo(SplitContext &splitContext);

// assign
inline void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
inline void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
inline void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
inline void AssignByBlock(AssignContext &assignContext);
inline void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

// FD
inline bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
inline void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

// main
inline void SplitFD(SplitResult &result);
inline void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
inline void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, SplitResult &result);



inline uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    const auto &actualSeqS1 = baseInfo.actualSeqS1Size;

    if (actualSeqS1.empty()) {
        return baseInfo.s1Size;
    }

    if (baseInfo.actualLenQDims == 1U) {
        return static_cast<uint32_t>(actualSeqS1[0]);
    }

    if (!baseInfo.isAccumSeqS1) {
        return static_cast<uint32_t>(actualSeqS1[bIdx]);
    }

    return (bIdx == 0U) ? static_cast<uint32_t>(actualSeqS1[bIdx]) :
                          static_cast<uint32_t>(actualSeqS1[bIdx] - actualSeqS1[bIdx - 1U]);
}

inline uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    const auto &actualSeqS2 = baseInfo.actualSeqS2Size;
    uint32_t prefix = static_cast<uint32_t>(baseInfo.actualSeqPrefixSize);
    if (actualSeqS2.empty()) {
        return prefix + baseInfo.s2Size;
    }

    if (baseInfo.actualLenKvDims == 1U) {
        return prefix + static_cast<uint32_t>(actualSeqS2[0]);
    }

    if (!baseInfo.isAccumSeqS2) {
        return prefix + static_cast<uint32_t>(actualSeqS2[bIdx]);
    }

    return (bIdx == 0U) ? prefix + static_cast<uint32_t>(actualSeqS2[bIdx]) :
                          prefix + static_cast<uint32_t>(actualSeqS2[bIdx] - actualSeqS2[bIdx - 1U]);
}

inline int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case SparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
        case SparseMode::TREE:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        default:
            return baseInfo.nextToken;
    }
}

inline int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

inline int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

inline BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
                                 uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

inline Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // no mask
    if (!baseInfo.attenMaskFlag) {
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) -
                           1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (!baseInfo.isS1G) {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    } else {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.gSize);
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2End = 0U;
        s2Start = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize +
            1U; // end of block index, +1 for Right-open interval

    return std::make_pair(s2Start, s2End);
}

inline void CalcSplitInfo(SplitContext &splitContext)
{
    const SplitParam &splitParam = splitContext.splitParam;
    const BaseInfo &baseInfo = splitContext.baseInfo;

    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);

        splitInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.s1GTailSize[bIdx] = (s1Size * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

inline void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const SplitParam &splitParam = splitContext.splitParam;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.s1GTailSize[bIdx],
                                        splitInfo.s2TailSize[bIdx]);
}

inline void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, baseInfo, splitParam, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num =
        (splitInfo.s2TailSize[batchCache.bIdx] != 0U && s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U :
                                                                                                                  0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

inline void CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];

        splitRes.fdRes.bN2IdxOfFdHead[i] = tmpRes.fdRes.bN2IdxOfFdHead[i];
        splitRes.fdRes.gS1IdxOfFdHead[i] = tmpRes.fdRes.gS1IdxOfFdHead[i];
        splitRes.fdRes.s2SplitNumOfFdHead[i] = tmpRes.fdRes.s2SplitNumOfFdHead[i];
        splitRes.fdRes.s2SplitStartIdxOfCore[i] = tmpRes.fdRes.s2SplitStartIdxOfCore[i];
        splitRes.fdRes.gS1SplitNumOfFdHead[i] = tmpRes.fdRes.gS1SplitNumOfFdHead[i];
        splitRes.fdRes.gS1LastPartSizeOfFdHead[i] = tmpRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }
}

inline void ClearTmpResult(SplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.usedVecNumOfFd = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.bN2IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitStartIdxOfCore[i] = 0U;
        tmpResult.fdRes.gS1SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1LastPartSizeOfFdHead[i] = 0U;
    }
}

inline void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0;

    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

inline void CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
}

inline void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx ==
        baseInfo.bSize * baseInfo.n2Size) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const CostInfo &costInfo = splitContext.costInfo;
    const BaseInfo &baseInfo = splitContext.baseInfo;

    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.isFinished = true;
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void AssignByBlock(AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost +
                                 curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

inline void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

inline bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

inline void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                                (s1Size * baseInfo.gSize - splitS1GIdx * splitParam.mBaseSize) :
                                splitParam.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + splitParam.gS1BaseSizeOfFd - 1U) / splitParam.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (splitParam.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}


inline void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        // 3、按块分配
        AssignByBlock(assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;

        // 对之前的归约信息进行记录并清理
        if (IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(splitContext, assignContext, result);
            assignContext.curKvSplitPart = 1U;
        }

        // 更新S2切分信息
        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

inline void SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector =
        static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum); // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector); // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector); // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}


inline void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, SplitResult &result)
{
    // 打印baseInfo,param信息
    SplitContext splitContext(baseInfo, param);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.gS1End[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }

    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.usedCoreNum = 1U;
    result.maxCost = INT64_MAX;

    SplitResult tmpResult{coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    // 3、存在FD任务，对FD进行负载均衡分配
    if (result.numOfFdHead > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U); // 至少使用1个core

    // 打印分核结果信息
}

} // namespace fa_split_core

#endif // FA_SPLIT_CORE_H
