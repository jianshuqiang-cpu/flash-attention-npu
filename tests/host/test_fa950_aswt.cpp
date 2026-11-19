#include "fwd_aswt.h"
#include "fwd_resource_plan.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace fa950_plan;

static void CheckCoverage(uint32_t cores, uint32_t qLen, uint32_t tile, uint32_t columns, uint32_t heads)
{
    const uint32_t rows = (qLen + tile - 1U) / tile;
    const uint32_t tasks = rows * columns;
    const auto plan = MakeAswtPlan(tasks, cores, qLen, tile, heads);
    const uint32_t slots = plan.slots ? plan.slots : tasks;
    assert((slots + cores - 1U) / cores == (tasks + cores - 1U) / cores);
    std::vector<uint8_t> writes(qLen * heads * columns);
    for (uint32_t core = 0; core < cores; ++core) {
        for (uint32_t slot = core; slot < slots; slot += cores) {
            const auto task = plan.Map(slot);
            assert(task.base < tasks);
            const uint32_t column = task.base % columns;
            const uint32_t first = task.base / columns * tile;
            const uint32_t tokens = std::min(tile, qLen - first);
            const uint32_t extent = qLen == 1 ? heads : tokens;
            const uint32_t grain = qLen == 1 ? ASWT_HEAD_GRAIN : ASWT_TOKEN_GRAIN;
            const uint32_t begin = AswtBoundary(extent, grain, task.part, task.parts);
            const uint32_t end = AswtBoundary(extent, grain, task.part + 1U, task.parts);
            for (uint32_t index = begin; index < end; ++index) {
                if (qLen == 1) {
                    assert(++writes[column * heads + index] == 1);
                } else {
                    assert(heads == 1U);
                    assert(++writes[column * qLen + first + index] == 1);
                }
            }
        }
    }
    assert(std::all_of(writes.begin(), writes.end(), [](auto count) { return count == 1; }));
}

static void ReportCase(uint32_t queries, uint32_t keys, uint32_t width, uint32_t kvHeads, uint32_t merged)
{
    constexpr uint32_t cores = 32U;
    const uint32_t tile = QueryTileForTasks(width, width, queries, (queries / 128U) * kvHeads * merged, cores);
    const uint32_t tasks = ((queries + tile - 1U) / tile) * kvHeads;
    const auto plan = MakeAswtPlan(tasks, cores, queries, tile, merged);
    const uint32_t slots = plan.slots ? plan.slots : tasks;
    uint32_t nonempty = 0;
    for (uint32_t slot = 0; slot < slots; ++slot) {
        const auto task = plan.Map(slot);
        const uint32_t first = task.base / kvHeads * tile;
        const uint32_t extent = queries == 1 ? merged : std::min(tile, queries - first);
        const uint32_t grain = queries == 1 ? ASWT_HEAD_GRAIN : ASWT_TOKEN_GRAIN;
        nonempty +=
            AswtBoundary(extent, grain, task.part, task.parts) < AswtBoundary(extent, grain, task.part + 1, task.parts);
    }
    CheckCoverage(cores, queries, tile, kvHeads, merged);
    std::cout << queries << ',' << keys << ',' << kvHeads * merged << ',' << kvHeads << ',' << width << ',' << tile
              << ',' << tasks << ',' << slots << ',' << nonempty << '\n';
}

int main()
{
    constexpr auto picture = MakeAswtPlan(13U * 19U, 32, 13U * 64U, 64, 1);
    static_assert(picture.slots == 247 && picture.parts == 1);
    static_assert(picture.Map(31).base == 3 * 19 + 7 && picture.Map(76).base == 4 * 19 + 18);
    static_assert(picture.Map(152).base == 8 * 19 && picture.Map(156).base == 12 * 19);
    constexpr auto doc = MakeAswtPlan(29, 20, 29 * 64, 64, 1);
    static_assert(doc.parts == 2 && doc.slots == 20 + 18 && doc.Map(19).parts == 1);
    constexpr auto decode = MakeAswtPlan(4, 32, 1, 64, 16);
    static_assert(decode.slots == 8 && decode.parts == 2);
    static_assert(MakeAswtPlan(0, 32, 0, 64, 1).slots == 0);
    static_assert(MakeAswtPlan(32, 32, 1, 64, 16).slots == 0);
    uint32_t checked = 0;
    for (auto cores : {1U, 8U, 20U, 32U, 48U, 64U}) {
        for (auto tile : {64U, 128U}) {
            for (auto columns : {1U, 2U, 4U, 7U, 19U, 32U}) {
                for (auto qLen : {2U, 31U, 32U, 63U, 64U, 65U, 128U, 129U, 1024U, 2048U, 2208U}) {
                    CheckCoverage(cores, qLen, tile, columns, 1);
                    ++checked;
                }
                for (auto heads : {1U, 4U, 7U, 8U, 15U, 16U, 32U}) {
                    CheckCoverage(cores, 1, tile, columns, heads);
                    ++checked;
                }
            }
        }
    }
    std::cout << checked << " schedules: complete, disjoint output coverage\n";
    std::cout << "32-core model: Sq,Sk,Hq,Hkv,D,qTile,base,slots,nonempty\n";
    ReportCase(1, 65535, 192, 4, 16);
    ReportCase(1024, 1024, 512, 2, 1);
    ReportCase(2048, 2048, 256, 1, 1);
    ReportCase(2208, 2048, 512, 1, 1);
    ReportCase(2208, 2048, 512, 2, 1);
    ReportCase(1024, 2368, 512, 1, 1);
    ReportCase(1024, 2368, 512, 2, 1);
    ReportCase(1024, 2368, 512, 4, 1);
}
