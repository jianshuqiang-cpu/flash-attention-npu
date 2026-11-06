// Resource bounds are independently checked over all supported dimensions.
#include "fwd_resource_plan.h"
#include "fwd_tiling_policy.h"
#include <cassert>
#include <iostream>

int main()
{
    for (unsigned d = 16; d <= 256; d += 16) {
        for (unsigned dv = 16; dv <= 256; dv += 16) {
            const auto u = fa950_plan::MakeUbPlan(dv, fa950_plan::OTmpStages(dv));
            assert(u.p >= u.s + 2 * u.sBytes);
            assert(u.lo >= u.p + 2 * u.pBytes);
            assert(u.go >= u.lo + u.otmpStages * u.loBytes);
            assert(u.lm >= u.go + u.loBytes);
            assert(u.mask >= u.gl + 256);
            assert(u.total >= u.mask + 16384);
            assert(u.total <= 248 * 1024);
            assert(fa950_plan::L1Bytes(d, dv) <= 512 * 1024);
            assert(fa950_plan::QueryTile(d, dv, 129) == 128);
            assert(fa950_plan::QueryTile(d, dv, 3) == (dv > 128 ? 64 : 128));
        }
    }
    assert(fa950_plan::OTmpStages(160) == 2);
    assert(fa950_plan::OTmpStages(192) == 1);
    assert(fa950_plan::OTmpStages(256) == 1);
    assert(fa950_plan::QueryTileForTasks(192, 192, 1024, 64, 28) == 64);
    assert(fa950_plan::QueryTileForTasks(192, 192, 1024, 128, 28) == 128);
    assert(fa950_plan::QueryTileForTasks(192, 192, 1024, 111, 28) == 64);
    assert(fa950_plan::QueryTileForTasks(192, 192, 1024, 112, 28) == 128);
    assert(fa950_plan::QueryTileForTasks(192, 128, 1024, 1, 28) == 128);
    assert(fa950_plan::ActiveQueryRows(4097, 2049, true, 7) == 2056);
    assert(fa950_plan::ActiveQueryRows(4097, 2049, true, 0) == 2049);
    assert(fa950_plan::ActiveQueryRows(4097, 2049, true, -1) == 4097);
    assert(fa950_plan::ActiveQueryRows(4097, 2049, false, 0) == 4097);
    assert(fa950_plan::ActiveQueryRows(4097, 0, false, -1) == 0);
    assert(fa950_plan::SelectQueryTile(192, 192, 8192, 1024, 28, true, 31, 7) == 64);
    assert(fa950_plan::SelectQueryTile(192, 192, 8192, 1024, 28, true, 511, 0) == 128);
    assert(fa950_plan::SelectQueryTile(128, 192, 2049, 192, 28, true, 257, 63) == 64);
    assert(fa950_plan::SelectQueryTile(128, 160, 2049, 192, 28, true, 257, 63) == 128);
    assert(fa950_plan::SelectQueryTile(128, 256, 2049, 192, 28, true, 479, 31) == 64);
    assert(fa950_plan::SelectQueryTile(128, 256, 2049, 192, 28, true, 480, 31) == 128);
    assert(fa950_plan::SelectQueryTile(192, 192, 8192, 1024, 28, true, 31, -1) == 128);
    std::cout << "resource bounds passed\n";
}
