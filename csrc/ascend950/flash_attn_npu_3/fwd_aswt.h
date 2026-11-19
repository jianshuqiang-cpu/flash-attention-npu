#pragma once

#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_ASWT_INLINE __forceinline__[host, aicore]
#else
#define FA950_ASWT_INLINE inline
#endif

namespace fa950_plan {
constexpr uint32_t ASWT_TOKEN_GRAIN = 32U;
constexpr uint32_t ASWT_HEAD_GRAIN = 8U;

struct AswtTask {
    uint32_t base, part, parts;
};

struct AswtPlan {
    uint32_t rows{}, columns{}, tailStart{}, parts{1}, slots{};

    FA950_ASWT_INLINE constexpr AswtTask Map(uint32_t slot) const
    {
        if (columns == 0U) {
            return {slot, 0U, 1U};
        }
        const uint32_t pieces = slot < tailStart ? 1U : parts;
        const uint32_t offset = slot < tailStart ? slot : slot - tailStart;
        uint32_t base = slot < tailStart ? slot : tailStart + offset / pieces;
        if (rows > 1U && columns > 1U) {
            const uint32_t last = rows < 4U ? 0U : rows / 4U - 1U;
            const uint32_t band = base / columns / 4U;
            const uint32_t window = band < last ? band : last;
            const uint32_t first = 4U * window;
            const uint32_t height = window == last ? rows - first : 4U;
            const uint32_t local = base - first * columns;
            const uint32_t column = local / height;
            base = (first + local % height) * columns + (window % 2U ? columns - 1U - column : column);
        }
        return {base, offset % pieces, pieces};
    }
};

FA950_ASWT_INLINE constexpr AswtPlan MakeAswtPlan(uint32_t tasks, uint32_t cores, uint32_t qLen, uint32_t qTile,
                                                  uint32_t mergedHeads)
{
    if (!tasks || !cores || !qLen || !qTile) {
        return {};
    }
    const uint32_t rows = 1U + (qLen - 1U) / qTile;
    if (tasks % rows != 0U) {
        return {};
    }
    const uint32_t tail = tasks % cores;
    const uint32_t capacity =
        qLen == 1U ? mergedHeads / ASWT_HEAD_GRAIN : (qLen < qTile ? qLen : qTile) / ASWT_TOKEN_GRAIN;
    uint32_t parts = 1U;
    while (tail && parts < 4U && parts * 2U <= capacity && parts * 2U <= cores / tail) {
        parts *= 2U;
    }
    if ((rows == 1U || tasks == rows) && parts == 1U) {
        return {};
    }
    if (tail * (parts - 1U) > UINT32_MAX - tasks) {
        return {};
    }
    return {rows, tasks / rows, tasks - tail, parts, tasks + tail * (parts - 1U)};
}

FA950_ASWT_INLINE constexpr uint32_t AswtBoundary(uint32_t size, uint32_t grain, uint32_t part, uint32_t parts)
{
    const uint32_t units = (size + grain - 1U) / grain;
    const uint32_t boundary = units * part / parts * grain;
    return boundary < size ? boundary : size;
}
} // namespace fa950_plan

#undef FA950_ASWT_INLINE
