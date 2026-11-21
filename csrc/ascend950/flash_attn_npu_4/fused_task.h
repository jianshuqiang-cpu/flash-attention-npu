// Copyright (c) 2026, Minghua Shen.
#pragma once

#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_TASK_INLINE __forceinline__[host, aicore]
#else
#define FA950_TASK_INLINE inline
#endif

namespace fa950_task {
// Nonempty FP16/BF16 segments are bounded in both metadata producers.
constexpr uint32_t MAX_Q_ROWS = 128U;
constexpr uint32_t KV_TILE = 128U;

struct QkL1Layout {
    uint32_t qSlots;
    uint32_t kRows;
    uint32_t kDepth;
};

// Each K slot holds one KV tile. For equal-width 16-bit inputs at D=256,
// Q/K/V use 128 KiB each and P uses 96 KiB (480 KiB of the 512 KiB L1).
inline QkL1Layout MakeQkL1Layout(uint32_t dq, uint32_t dv, uint32_t kvTile, bool fp8)
{
    if (dq == dv && dq > 0U && dq <= 256U) {
        return {2U, kvTile, (dq + 15U) / 16U * 16U};
    }
    return {1U, 256U, fp8 ? 128U : 192U};
}

FA950_TASK_INLINE uint32_t CeilDiv(uint32_t n, uint32_t d)
{
    return (n + d - 1U) / d;
}

FA950_TASK_INLINE uint32_t HeadTile(uint32_t rows, uint32_t groupSize, uint32_t qTile)
{
    uint32_t tile = rows == 0U ? MAX_Q_ROWS : (MAX_Q_ROWS / rows) / 2U * 2U;
    if (qTile < MAX_Q_ROWS && rows != 0U) {
        const uint32_t capacity = qTile / rows;
        tile = tile < capacity ? tile : capacity;
    }
    tile = tile < groupSize ? tile : groupSize;
    if (tile > 2U) {
        tile = tile / 2U * 2U;
    }
    return tile == 0U ? 1U : tile;
}

struct Shape {
    uint32_t fullBlocks;
    uint32_t headsPerFullTask;
    uint32_t fullHeadTasks;
    uint32_t fullTasks;
    uint32_t headsPerTailTask;
    uint32_t totalTasks;
};

// Zero and one are the two legacy metadata encodings. A larger value
// caps how many distinct KV heads one task may own without losing a core wave.
FA950_TASK_INLINE uint32_t HeadGroupBudget(uint32_t batch, uint32_t kvHeads, uint32_t cores)
{
    const uint64_t parallel = static_cast<uint64_t>(batch) * kvHeads;
    const uint32_t groups = cores == 0U ? 1U : static_cast<uint32_t>(parallel / cores);
    return groups < 2U ? 1U : (groups < kvHeads ? groups : kvHeads);
}

FA950_TASK_INLINE Shape MakeShape(uint32_t qLength, uint32_t groupSize, uint32_t kvHeads, uint32_t qTile,
                                  uint32_t headGroupBudget)
{
    const uint32_t tailRows = qLength % qTile;
    const bool separateTail = tailRows != 0U && (headGroupBudget > 1U ||
                                                (headGroupBudget == 1U && qLength > qTile));
    Shape shape{};
    shape.fullBlocks = separateTail ? qLength / qTile : CeilDiv(qLength, qTile);
    shape.headsPerFullTask = HeadTile(qLength, groupSize, qTile);
    shape.fullHeadTasks = kvHeads * CeilDiv(groupSize, shape.headsPerFullTask);
    shape.fullTasks = shape.fullBlocks * shape.fullHeadTasks;
    shape.totalTasks = shape.fullTasks;
    if (separateTail) {
        shape.headsPerTailTask = HeadTile(tailRows, groupSize, qTile);
        // Q and P in L1 need an independently aligned slice per KV head.
        // Groups which do not fit together keep the original within-head schedule.
        uint32_t groups = 1U;
        if (headGroupBudget > 1U && groupSize <= qTile / tailRows) {
            const uint32_t paddedRows = CeilDiv(tailRows * groupSize, 16U) * 16U;
            const uint32_t capacity = qTile / paddedRows;
            const uint32_t limit = capacity < headGroupBudget ? capacity : headGroupBudget;
            groups = CeilDiv(kvHeads, CeilDiv(kvHeads, limit));
        }
        if (groups > 1U) {
            shape.headsPerTailTask = groups * groupSize;
            shape.totalTasks += CeilDiv(kvHeads, groups);
        } else {
            shape.totalTasks += kvHeads * CeilDiv(groupSize, shape.headsPerTailTask);
        }
    }
    return shape;
}

struct Task {
    uint32_t qTileIndex, firstHead, heads;
};

FA950_TASK_INLINE Task Decode(const Shape& shape, uint32_t index, uint32_t group, uint32_t kvHeads)
{
    const bool tail = index >= shape.fullTasks;
    const uint32_t qTileIndex = tail ? shape.fullBlocks : index / shape.fullHeadTasks;
    const uint32_t tile = tail ? shape.headsPerTailTask : shape.headsPerFullTask;
    const uint32_t headIndex = tail ? index - shape.fullTasks : index % shape.fullHeadTasks;
    if (tile > group) {
        const uint32_t first = headIndex * tile;
        const uint32_t remaining = kvHeads * group - first;
        return {qTileIndex, first, remaining < tile ? remaining : tile};
    }
    const uint32_t tasksPerGroup = CeilDiv(group, tile);
    const uint32_t offset = headIndex % tasksPerGroup * tile;
    const uint32_t remaining = group - offset;
    return {qTileIndex, headIndex / tasksPerGroup * group + offset, remaining < tile ? remaining : tile};
}
} // namespace fa950_task

#undef FA950_TASK_INLINE
