/** Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 * Shared Host/AICPU/AICore geometry, adapted from PR #394 for 950.
 */
#pragma once
#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_TASK_INLINE __forceinline__[host, aicore]
#else
#define FA950_TASK_INLINE inline
#endif

namespace fa950_task {
FA950_TASK_INLINE uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
FA950_TASK_INLINE uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
FA950_TASK_INLINE uint32_t Align16(uint32_t a) { return CeilDiv(a, 16U) * 16U; }

FA950_TASK_INLINE uint32_t LegacyHeadTile(uint32_t rows, uint32_t group, uint32_t qTile)
{
    uint32_t tile = rows == 0U ? 128U : (128U / rows) / 2U * 2U;
    if (qTile < 128U && rows != 0U) {
        uint32_t cap = 64U / rows;
        tile = Min(tile, cap > 0U ? cap : 1U);
    }
    tile = Min(tile, group);
    if (tile > 2U) tile = tile / 2U * 2U;
    return tile > 0U ? tile : 1U;
}

struct Task {
    uint32_t qStart, qRows, headStart, heads;
};

struct Shape {
    uint32_t fullBlocks, fullHeadTile, fullHeadTasks, fullTasks;
    uint32_t tailRows, tailHeadTile, tailHeadTasks, totalTasks;
};

// With one query head per KV head and no cross-group fusion, every Q tile
// contributes exactly `heads` tasks, including a short final tile.
FA950_TASK_INLINE Shape MakeMhaShape(uint32_t q, uint32_t heads, uint32_t qTile)
{
    Shape s{};
    s.fullBlocks = q / qTile;
    s.fullHeadTile = s.tailHeadTile = 1U;
    s.fullHeadTasks = heads;
    s.fullTasks = s.fullBlocks * heads;
    s.tailRows = q % qTile;
    s.tailHeadTasks = s.tailRows ? heads : 0U;
    s.totalTasks = s.fullTasks + s.tailHeadTasks;
    return s;
}

FA950_TASK_INLINE Task DecodeMha(uint32_t index, uint32_t q, uint32_t heads, uint32_t qTile)
{
    const uint32_t qStart = index / heads * qTile;
    return {qStart, Min(qTile, q - qStart), index % heads, 1U};
}

FA950_TASK_INLINE uint32_t HeadTasks(uint32_t heads, uint32_t group, uint32_t tile)
{
    return tile > group ? CeilDiv(heads, tile) : (heads / group) * CeilDiv(group, tile);
}

FA950_TASK_INLINE Shape MakeShape(uint32_t q, uint32_t heads, uint32_t kvHeads,
                                  uint32_t qTile, uint32_t batch, uint32_t cores,
                                  bool fixedLayout, bool quantized = false, bool allowCrossGroup = true)
{
    const uint32_t group = heads / kvHeads;
    Shape s{};
    s.fullBlocks = quantized ? CeilDiv(q, qTile) : q / qTile;
    s.fullHeadTile = quantized ? LegacyHeadTile(q, group, qTile) : 1U;
    s.fullHeadTasks = HeadTasks(heads, group, s.fullHeadTile);
    s.fullTasks = s.fullBlocks * s.fullHeadTasks;
    s.tailRows = quantized ? 0U : q % qTile;
    s.tailHeadTile = s.tailRows ? LegacyHeadTile(s.tailRows, group, qTile) : 1U;
    // On 950, serial segment GEMMs can outweigh vector fusion for a long-Q
    // tail. Keep cross-group fusion for short Q; GQA tails still merge within
    // their KV group. Unmasked multi-KV workloads retain their DN fast path.
    if (s.tailRows && q < qTile && batch >= cores && allowCrossGroup) {
        // Each KV segment has its own 16-row-aligned Q/P storage. Effective
        // rows remain compact in UB; both AIVs are bounded by 64 state rows.
        const uint32_t groups = Min(kvHeads, qTile / Align16(s.tailRows * group));
        if (groups > 1U) {
            const uint32_t balanced = CeilDiv(kvHeads, CeilDiv(kvHeads, groups));
            s.tailHeadTile = balanced * group;
        }
    }
    s.tailHeadTasks = s.tailRows ? HeadTasks(heads, group, s.tailHeadTile) : 0U;
    if (fixedLayout && group == 1U && s.tailHeadTasks &&
        s.fullTasks + s.tailHeadTasks <= cores + 2U) {
        s.tailHeadTile = Min(s.tailHeadTile, 2U);
        s.tailHeadTasks = HeadTasks(heads, group, s.tailHeadTile);
    }
    s.totalTasks = s.fullTasks + s.tailHeadTasks;
    return s;
}

FA950_TASK_INLINE Task Decode(const Shape& s, uint32_t index, uint32_t q,
                               uint32_t heads, uint32_t kvHeads, uint32_t qTile)
{
    const uint32_t group = heads / kvHeads;
    const bool tail = index >= s.fullTasks;
    const uint32_t tile = tail ? s.tailHeadTile : s.fullHeadTile;
    const uint32_t headTask = tail ? index - s.fullTasks : index % s.fullHeadTasks;
    Task t{};
    t.qStart = (tail ? s.fullBlocks : index / s.fullHeadTasks) * qTile;
    t.qRows = Min(qTile, q - t.qStart);
    if (tile > group) {
        t.headStart = headTask * tile;
        t.heads = Min(tile, heads - t.headStart);
    } else {
        const uint32_t perGroup = CeilDiv(group, tile);
        const uint32_t inGroup = headTask % perGroup * tile;
        t.headStart = headTask / perGroup * group + inGroup;
        t.heads = Min(tile, group - inGroup);
    }
    return t;
}
} // namespace fa950_task
#undef FA950_TASK_INLINE
