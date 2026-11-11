/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_TILING_POLICY_H
#define FA950_FWD_TILING_POLICY_H
#include "fwd_resource_plan.h"
#include "fused_task.h"
#include "tilingdata.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fa950_plan {
// Host/AICPU policy: fully masked query rows do not provide Cube work. With
// bottom-right alignment, a finite right bound makes the first Q-K-right rows
// empty. Unlimited-right windows and unmasked attention keep every query row.
inline uint32_t ActiveQueryRows(uint32_t q, uint32_t kv, bool hasMask, int64_t right)
{
    if (kv == 0U)
        return 0U;
    if (!hasMask || right < 0)
        return q;
    const uint64_t rows = uint64_t(kv) + uint64_t(right);
    return rows < q ? static_cast<uint32_t>(rows) : q;
}

inline uint32_t SelectQueryTile(uint32_t d, uint32_t dv, uint32_t maxQ, uint64_t fullTasks, uint32_t cores,
                                bool isLocal, int64_t left, int64_t right)
{
    // A narrow finite window adds masked KV work when Q doubles. Single PV
    // storage also serializes Fixpipe/rescale: use a conservative four-KV-tile
    // window floor there, versus two tiles when both PV stages fit in UB.
    const uint32_t minWindowTiles = OTmpStages(dv) == 1U ? 4U : 2U;
    if (dv > 128U && isLocal && left >= 0 && right >= 0 && uint64_t(left) + right + 1U < minWindowTiles * 128U)
        return 64U;
    return QueryTileForTasks(d, dv, maxQ, fullTasks, cores);
}
} // namespace fa950_plan

namespace optiling {
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t Q_TILE_CEIL = 128;
const uint64_t FD_WORKSPACE_ALIGNMENT = 512ULL;

enum class MaskType : uint32_t {
    NO_MASK = 0,
    MASK_SPEC = 1,
    MASK_BAND = 2
};

enum class DataType : uint32_t {
    FP16 = 0,
    BF16 = 1,
    FP8 = 2
};

struct FAInferContext {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t innerPrecise = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    int32_t sparseMode = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    const int32_t* qSeqlenList{nullptr};
    const int32_t* kvSeqlenList{nullptr};
    float scaleValue = 0.0;
    float softcapValue = 0.0;
    size_t* workspaces{nullptr};
    MaskType maskType = MaskType::MASK_SPEC;
    DataType dataType = DataType::FP16;
    bool pagedCacheFlag = true;
    bool lseFlag = false;
    bool isTilingSink = false;
    bool learnableSinkFlag = false;
    bool flashDecodeFlag = false;
    uint32_t numSplits = 1;
    bool kvcacheNzFlag = false;
    FwdLayout layout = FwdLayout::BSND;
    bool pagedShapeFlag = true;
};

inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize, uint32_t qBaseTile)
{
    return fa950_task::LegacyHeadTile(qSeqlen, groupSize, qBaseTile);
}

struct FdBaseTaskHost {
    uint32_t kvTiles;
    uint32_t rowNum;
    uint32_t batchIdx;
    uint32_t qStart;
    uint32_t qRows;
    uint32_t qHeadStart;
};

struct FdCandidateHost {
    uint32_t baseTask;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint64_t cost;
};

inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

inline void FillFdCombineCoreSchedules(FAInferTilingData& tiling)
{
    struct SourcePartition {
        uint32_t task;
        uint32_t subBlock;
        uint32_t rows;
    };
    std::vector<SourcePartition> partitions;
    uint64_t totalLoad = 0U;
    for (uint32_t task = 0U; task < tiling.fdCombineTaskNum; ++task) {
        const uint32_t rows = tiling.fdCombineTaskInfo[task].rowNum;
        const uint32_t sourceSplit = std::min(
            rows, static_cast<uint32_t>(AlignUpU64(rows, tiling.quantMode == 1 ? 32U : 8U)) / FD_SOURCE_SUB_BLOCK_NUM);
        const uint32_t sourceRows[FD_SOURCE_SUB_BLOCK_NUM] = {sourceSplit, rows - sourceSplit};
        for (uint32_t sub = 0U; sub < FD_SOURCE_SUB_BLOCK_NUM; ++sub) {
            if (sourceRows[sub] != 0U) {
                partitions.push_back({task, sub, sourceRows[sub]});
            }
        }
        totalLoad += static_cast<uint64_t>(rows) * tiling.fdCombineSchedules[task].partialCount;
    }

    const uint32_t aivNum = tiling.fdActiveCoreNum * FD_SOURCE_SUB_BLOCK_NUM;
    // FD has fewer base tasks than active AICs, hence every nonempty
    // source partition can reserve at least one AIV.
    if (aivNum == 0U || partitions.size() > aivNum) {
        throw std::runtime_error("FD combine source partitions exceed available AIVs");
    }
    const uint64_t averageLoad = std::max<uint64_t>(1U, (totalLoad + aivNum - 1U) / aivNum);
    uint32_t extraAivs = aivNum - static_cast<uint32_t>(partitions.size());
    uint32_t nextAiv = 0U;
    for (const auto& partition : partitions) {
        const uint64_t load =
            static_cast<uint64_t>(partition.rows) * tiling.fdCombineSchedules[partition.task].partialCount;
        // Reserve one AIV for every remaining partition before deriving
        // the row extent. Recompute the actual count after ceil rounding.
        const uint32_t initialAivs = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(1U, load / averageLoad), std::min(partition.rows, extraAivs + 1U)));
        const uint32_t rowsPerAiv = (partition.rows + initialAivs - 1U) / initialAivs;
        const uint32_t usedAivs = (partition.rows + rowsPerAiv - 1U) / rowsPerAiv;
        for (uint32_t i = 0U; i < usedAivs; ++i) {
            const uint32_t row = i * rowsPerAiv;
            tiling.fdCombineCoreSchedules[nextAiv++] = {partition.task, partition.subBlock, row,
                                                        std::min(rowsPerAiv, partition.rows - row)};
        }
        extraAivs -= usedAivs - 1U;
    }
    tiling.fdCombineAivNum = nextAiv;

    // One combine task has no inter-task imbalance to remove. For the
    // no-LSE path, also retain uniform dispatch when its busiest AIV's
    // aggregate partial O fits the existing input buffer: extra metadata
    // and per-AIV rounds can cost more than the small reduction in work.
    // Derive this from actual partials/rows, never the requested splits.
    tiling.fdCombineUseUniform = 0U;
    if (tiling.fdCombineTaskNum == 1U) {
        tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE;
    } else {
        const uint32_t rowTile =
            std::min(FD_COMBINE_ROW_TILE, std::max(1U, (tiling.fdCombineRowNum + aivNum - 1U) / aivNum));
        std::vector<uint64_t> uniformLoads(aivNum, 0U);
        std::vector<uint32_t> uniformRounds(aivNum, 0U);
        uint32_t tile = 0U;
        for (const auto& partition : partitions) {
            for (uint32_t row = 0U; row < partition.rows; row += rowTile) {
                const uint32_t core = tile++ % aivNum;
                ++uniformRounds[core];
                uniformLoads[core] += static_cast<uint64_t>(std::min(rowTile, partition.rows - row)) *
                                      tiling.fdCombineSchedules[partition.task].partialCount;
            }
        }
        const uint64_t maxInputElems =
            *std::max_element(uniformLoads.begin(), uniformLoads.end()) * AlignUpU64(tiling.embeddingSizeV, 8U);
        if (maxInputElems <= FD_COMBINE_O_INPUT_ELEMS) {
            tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE;
            uint32_t balancedRounds = 0U;
            for (uint32_t core = 0U; core < tiling.fdCombineAivNum; ++core) {
                balancedRounds =
                    std::max(balancedRounds, (tiling.fdCombineCoreSchedules[core].rowCount + FD_COMBINE_ROW_TILE - 1U) /
                                                 FD_COMBINE_ROW_TILE);
            }
            // With no multi-buffer input to remove, adding a whole extra
            // row round can also outweigh balancing the LSE stores.
            if (balancedRounds > *std::max_element(uniformRounds.begin(), uniformRounds.end())) {
                tiling.fdCombineUseUniform |= FD_COMBINE_UNIFORM_WITH_LSE;
            }
        }
    }
}
inline void FillFlashDecodeTilingData(FAInferTilingData& tiling, const FAInferContext& faInfo_,
                                     uint32_t blockNum_)
{
    tiling.fdCombineAivNum = 0U;
    tiling.fdCombineUseUniform = 0U;
    for (auto& schedule : tiling.fdCombineCoreSchedules) {
        schedule = {};
    }
    for (uint32_t i = 0; i < MAX_FD_ACTIVE_CORE_NUM; ++i) {
        tiling.fdDecodeSchedules[i] = {-1, -1, -1, -1};
    }
    for (uint32_t i = 0; i < MAX_FD_COMBINE_TASK_NUM; ++i) {
        tiling.fdCombineSchedules[i] = {-1, -1, -1, 0};
        tiling.fdCombineTaskInfo[i] = {};
    }

    const uint32_t cfdMax = std::min(blockNum_, MAX_FD_ACTIVE_CORE_NUM);
    const uint32_t bfdMax = std::min(MAX_FD_COMBINE_TASK_NUM, blockNum_ == 0U ? 0U : (3U * blockNum_ - 1U) / 10U);
    tiling.fdPartialCapacity = (bfdMax == 0U || cfdMax == 0U) ? 0U : bfdMax + cfdMax - 1U;

    std::vector<FdBaseTaskHost> bases;
    const uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; ++batchIdx) {
        uint32_t qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx]);
        uint32_t kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx]);
        if (faInfo_.layout == FwdLayout::TND) {
            qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx + 1] - faInfo_.qSeqlenList[batchIdx]);
            if (!faInfo_.pagedCacheFlag) {
                kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx + 1] - faInfo_.kvSeqlenList[batchIdx]);
            }
        }
        const uint32_t kvTileNum = (kvLen + tiling.kvBaseTile - 1U) / tiling.kvBaseTile;
        const auto shape = fa950_task::MakeShape(qLen, faInfo_.numHeads, faInfo_.kvHeads,
            tiling.qBaseTile, faInfo_.batch, blockNum_, faInfo_.layout != FwdLayout::TND,
            tiling.quantMode != 0U,
            faInfo_.maskType != MaskType::NO_MASK || faInfo_.maxKvSeqlen <= 128U);
        for (uint32_t task = 0; task < shape.totalTasks; ++task) {
            const auto t = fa950_task::Decode(shape, task, qLen, faInfo_.numHeads,
                                             faInfo_.kvHeads, tiling.qBaseTile);
            bases.push_back({kvTileNum, t.qRows * t.heads, static_cast<uint32_t>(batchIdx),
                             t.qStart, t.qRows, t.headStart});
        }
    }

    tiling.fdBaseTaskNum = static_cast<uint32_t>(bases.size());
    if (bases.empty() || bases.size() > bfdMax || cfdMax < 2U) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    std::vector<uint32_t> splitCounts(bases.size(), 1U);
    if (faInfo_.numSplits > 1U) {
        for (uint32_t base = 0; base < bases.size(); ++base) {
            splitCounts[base] = std::min(faInfo_.numSplits, bases[base].kvTiles);
        }
    } else {
        uint32_t totalSegments = static_cast<uint32_t>(bases.size());
        while (totalSegments < cfdMax) {
            uint32_t bestBase = static_cast<uint32_t>(bases.size());
            uint64_t bestCost = 0U;
            for (uint32_t base = 0; base < bases.size(); ++base) {
                if (splitCounts[base] >= bases[base].kvTiles) {
                    continue;
                }
                const uint64_t cost =
                    static_cast<uint64_t>(bases[base].rowNum) * bases[base].kvTiles / splitCounts[base];
                if (bestBase == bases.size() || cost > bestCost) {
                    bestBase = base;
                    bestCost = cost;
                }
            }
            if (bestBase == bases.size()) {
                break;
            }
            ++splitCounts[bestBase];
            ++totalSegments;
        }
    }

    std::vector<FdCandidateHost> candidates;
    for (uint32_t base = 0; base < bases.size(); ++base) {
        const uint32_t count = splitCounts[base];
        for (uint32_t split = 0; split < count; ++split) {
            const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * split / count);
            const uint32_t end =
                static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * (split + 1U) / count);
            candidates.push_back({base, begin, end, static_cast<uint64_t>(bases[base].rowNum) * (end - begin)});
        }
    }

    const uint32_t activeCores = std::min(cfdMax, static_cast<uint32_t>(candidates.size()));
    if (activeCores <= bases.size()) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    uint64_t totalCost = 0U;
    for (const auto& candidate : candidates) {
        totalCost += candidate.cost;
    }
    std::vector<uint32_t> coreBegin(activeCores);
    std::vector<uint32_t> coreEnd(activeCores);
    uint32_t candidateBegin = 0U;
    uint64_t consumedCost = 0U;
    for (uint32_t core = 0; core < activeCores; ++core) {
        coreBegin[core] = candidateBegin;
        const uint32_t coresLeft = activeCores - core;
        const uint32_t maxEnd = static_cast<uint32_t>(candidates.size()) - (coresLeft - 1U);
        uint32_t end = candidateBegin + 1U;
        const uint64_t target = totalCost * (core + 1U) / activeCores;
        while (end < maxEnd && consumedCost + candidates[end - 1U].cost < target) {
            consumedCost += candidates[end - 1U].cost;
            ++end;
        }
        consumedCost += candidates[end - 1U].cost;
        coreEnd[core] = end;
        candidateBegin = end;

        const auto& first = candidates[coreBegin[core]];
        const auto& last = candidates[end - 1U];
        tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(first.baseTask),
                                          static_cast<int32_t>(last.baseTask + 1U), static_cast<int32_t>(first.kvBegin),
                                          static_cast<int32_t>(last.kvEnd)};
    }

    uint32_t partialStart = 0U;
    uint32_t combineCount = 0U;
    uint32_t combineRows = 0U;
    for (uint32_t base = 0; base < bases.size(); ++base) {
        int32_t firstCore = -1;
        uint32_t partialCount = 0U;
        for (uint32_t core = 0; core < activeCores; ++core) {
            const auto& schedule = tiling.fdDecodeSchedules[core];
            if (schedule.baseTaskStart <= static_cast<int32_t>(base) &&
                static_cast<int32_t>(base) < schedule.baseTaskEnd) {
                if (firstCore < 0) {
                    firstCore = static_cast<int32_t>(core);
                }
                ++partialCount;
            }
        }
        if (partialCount > 1U) {
            tiling.fdCombineSchedules[combineCount] = {static_cast<int32_t>(base), firstCore,
                                                       static_cast<int32_t>(partialStart),
                                                       static_cast<int32_t>(partialCount)};
            const auto& info = bases[base];
            tiling.fdCombineTaskInfo[combineCount++] = {info.batchIdx, info.qStart, info.qRows, info.qHeadStart,
                                                        info.rowNum};
            partialStart += partialCount;
            combineRows += bases[base].rowNum;
        }
    }

    if (combineCount == 0U || partialStart > tiling.fdPartialCapacity) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    tiling.fdActiveCoreNum = activeCores;
    tiling.fdCombineTaskNum = combineCount;
    tiling.fdPartialTaskNum = partialStart;
    tiling.fdCombineRowNum = combineRows;
    FillFdCombineCoreSchedules(tiling);
    uint32_t maxFdRows = 0U;
    for (const auto& base : bases) {
        maxFdRows = std::max(maxFdRows, base.rowNum);
    }
    tiling.fdRowCapacity = static_cast<uint32_t>(AlignUpU64(maxFdRows, 8U));
    tiling.fdLseSubStride = tiling.quantMode == 1
                                ? static_cast<uint32_t>(AlignUpU64(tiling.fdRowCapacity, 32U) / 2U)
                                : static_cast<uint32_t>(AlignUpU64((tiling.fdRowCapacity + 1U) / 2U, 8U));

    // The Ascend950 FD pipeline keeps QK, softmax, PV and rescale buffers
    // on chip. Its GM workspace only needs the partial LSE and O regions.
    tiling.set_qkOutSize(0U);
    tiling.set_smOnlineOutSize(0U);
    tiling.set_pvOutSize(0U);
    tiling.set_UpdateSize(0U);
    tiling.fdPartialLseOffset = 0U;
    const uint64_t partialLseSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * 2U * tiling.fdLseSubStride * sizeof(float);
    tiling.fdPartialOOffset = AlignUpU64(tiling.fdPartialLseOffset + partialLseSize, FD_WORKSPACE_ALIGNMENT);
    const uint64_t partialOSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * tiling.fdRowCapacity * faInfo_.embeddingSizeV * sizeof(float);
    tiling.fdWorkspaceEnd = tiling.fdPartialOOffset + partialOSize;
    tiling.workSpaceSize = tiling.fdWorkspaceEnd;
}

// Safe for every valid schedule, including explicit splits and FP8. This bound
// depends only on public geometry, never on a host read of device metadata.
inline uint64_t FlashDecodeWorkSpaceSize(uint32_t blockNum, uint32_t embeddingSizeV)
{
    const uint32_t cfdMax = std::min(blockNum, MAX_FD_ACTIVE_CORE_NUM);
    const uint32_t bfdMax = std::min(MAX_FD_COMBINE_TASK_NUM, blockNum == 0U ? 0U : (3U * blockNum - 1U) / 10U);
    if (cfdMax == 0U || bfdMax == 0U) return 0U;
    const uint64_t fdPartialCapacity = bfdMax + cfdMax - 1U;
    return AlignUpU64(fdPartialCapacity * 2U * 64U * sizeof(float), FD_WORKSPACE_ALIGNMENT) + fdPartialCapacity * 128U * embeddingSizeV * sizeof(float);
}
} // namespace optiling
#endif
