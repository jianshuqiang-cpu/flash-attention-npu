/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_RESOURCE_PLAN_H
#define FA950_FWD_RESOURCE_PLAN_H

#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_PLAN_INLINE __forceinline__[host, aicore]
#else
#define FA950_PLAN_INLINE inline
#endif

namespace fa950_plan {
constexpr uint32_t UB_CAPACITY = 248U * 1024U;
constexpr uint32_t L1_CAPACITY = 512U * 1024U;
constexpr uint32_t SCORE_STAGES = 2;
constexpr uint32_t P_STAGES = 3;
// The tiling payload retains the conservative DN/FP8 Q/K reservation.
// Ordinary FP16/BF16 kernels compact K and use two Q slots within that bound.
constexpr uint32_t Q_STAGES = 1;
constexpr uint32_t NORMAL_Q_STAGES = 2;
constexpr uint32_t K_STAGES = 2;
constexpr uint32_t V_STAGES = 2;
// FP16/BF16 task geometry caps total (sequence * merged heads) rows at 128.
constexpr uint32_t MAX_Q_ROWS = 128;
constexpr uint32_t KV_TILE = 128;
constexpr uint32_t PV_FLAG_BASE = SCORE_STAGES + P_STAGES;

struct UbPlan {
    uint32_t s, p, lo, go, lm, gm, dm, ll, gl, total;
    uint32_t finalMax, finalSum, lse;
    uint32_t sBytes, pBytes, loBytes, otmpStages;
};

// Score/P retain the 64-row capacity needed by merged GQA and DN. The
// reduction vectors also retain 64 elements because VF loads whole registers.
FA950_PLAN_INLINE constexpr UbPlan MakeUbPlan(uint32_t dv, uint32_t otmpStages)
{
    const uint32_t alignedDv = (dv + 15U) / 16U * 16U;
    UbPlan u{};
    u.sBytes = 64U * 128U * 4U;
    u.pBytes = (64U + 1U) * 128U * 2U;
    u.loBytes = 64U * alignedDv * 4U;
    u.otmpStages = otmpStages;
    u.p = SCORE_STAGES * u.sBytes;
    u.lo = u.p + SCORE_STAGES * u.pBytes;
    u.go = u.lo + otmpStages * u.loBytes;
    u.lm = u.go + u.loBytes;
    u.gm = u.lm + 64U * 4U;
    u.dm = u.gm + 64U * 4U;
    u.ll = u.dm + P_STAGES * 64U * 4U;
    u.gl = u.ll + 64U * 4U;
    // No-mask, causal and SWA share this layout. Positional masking needs no
    // UB mask matrix; place the live final-stat snapshots directly after GL.
    u.finalMax = u.gl + 64U * 4U;
    u.finalSum = u.finalMax + P_STAGES * 64U * 4U;
    u.lse = u.finalSum + P_STAGES * 64U * 4U;
    u.total = u.lse + 64U * 4U;
    return u;
}

FA950_PLAN_INLINE constexpr uint32_t OTmpStages(uint32_t dv)
{
    return MakeUbPlan(dv, 2U).total <= UB_CAPACITY ? 2U : 1U;
}

FA950_PLAN_INLINE constexpr UbPlan KernelUbPlan(uint32_t dv, uint32_t qTile)
{
    if (dv > 128U && qTile == 128U) {
        return MakeUbPlan(dv, OTmpStages(dv));
    }
    // Preserve the established small-query/DN layout and its pipeline depths.
    UbPlan u = MakeUbPlan(128U, 2U);
    u.lo = 128U * 1024U;
    u.go = 192U * 1024U;
    u.lm = 224U * 1024U;
    u.gm = u.lm + 256U;
    u.dm = u.gm + 256U;
    u.ll = u.dm + 768U;
    u.gl = u.ll + 256U;
    u.finalMax = u.gl + 256U;
    u.finalSum = u.finalMax + P_STAGES * 64U * 4U;
    u.lse = u.finalSum + P_STAGES * 64U * 4U;
    u.total = u.lse + 64U * 4U;
    return u;
}

FA950_PLAN_INLINE constexpr uint32_t L1Bytes(uint32_t d, uint32_t dv)
{
    return 128U * ((d + 15U) / 16U * 16U) * 2U + 192U * 256U * 2U * 2U + 128U * 128U * 2U * P_STAGES +
           128U * ((dv + 15U) / 16U * 16U) * 2U * 2U;
}

// Actual ordinary FP16/BF16 L1 footprint. DN retains L1Bytes above.
FA950_PLAN_INLINE constexpr uint32_t NormalL1Bytes(uint32_t d, uint32_t dv)
{
    const uint32_t alignedD = (d + 15U) / 16U * 16U;
    const uint32_t alignedDv = (dv + 15U) / 16U * 16U;
    return MAX_Q_ROWS * alignedD * 2U * NORMAL_Q_STAGES +
           KV_TILE * alignedD * 2U * K_STAGES +
           MAX_Q_ROWS * KV_TILE * 2U * P_STAGES +
           KV_TILE * alignedDv * 2U * V_STAGES;
}

static_assert(NORMAL_Q_STAGES == 2U, "Q rotation and events 0/5 require two slots");
static_assert(NormalL1Bytes(256U, 256U) <= L1Bytes(256U, 256U), "Normal L1 upper bound");

FA950_PLAN_INLINE constexpr uint32_t QueryTile(uint32_t d, uint32_t dv, uint32_t maxQ)
{
    // Preserve small-query task parallelism and existing decode grouping.
    return dv > 128U &&
                   (maxQ <= 64U || L1Bytes(d, dv) > L1_CAPACITY || MakeUbPlan(dv, OTmpStages(dv)).total > UB_CAPACITY)
               ? 64U
               : 128U;
}

FA950_PLAN_INLINE constexpr uint32_t QueryTileForTasks(uint32_t d, uint32_t dv, uint32_t maxQ, uint64_t fullQ128Tasks,
                                                       uint32_t cores)
{
    // Larger tiles reduce independent Q work. Keep several complete waves so
    // tail tasks and the single-PV-buffer latency do not dominate small inputs.
    // Count only full Q tiles; ragged/empty sequences must not inflate occupancy.
    return dv > 128U && fullQ128Tasks < uint64_t(cores) * 4U ? 64U : QueryTile(d, dv, maxQ);
}

static_assert(MakeUbPlan(160, 2).total == 225280U, "D160 dual-buffer layout");
static_assert(MakeUbPlan(192, 2).total == 249856U, "D192 dual-buffer layout");
static_assert(MakeUbPlan(256, 1).total == 233472U, "D256 single-buffer layout");
static_assert(OTmpStages(192) == 2U && OTmpStages(208) == 1U, "PV stage capacity boundary");
static_assert(KernelUbPlan(128, 128).total <= UB_CAPACITY, "Small-query/DN UB capacity");
static_assert(MakeUbPlan(256, 1).total <= UB_CAPACITY, "FP16/BF16 UB capacity");
static_assert(L1Bytes(256, 256) <= L1_CAPACITY, "FP16/BF16 L1 capacity");
} // namespace fa950_plan

#undef FA950_PLAN_INLINE
#endif
