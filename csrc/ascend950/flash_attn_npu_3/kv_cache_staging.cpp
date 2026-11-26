#include "kv_cache_staging.hpp"

#include <algorithm>
#include <limits>

#include <c10/util/Exception.h>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

constexpr uint32_t kPageRows = 128;
constexpr uint32_t kHeadDim = 192;
constexpr uint32_t kTileRows = 64;
constexpr uint32_t kTileElems = kTileRows * kHeadDim;
constexpr uint32_t kScalePitch = 8;
constexpr uint32_t kTilesPerHead = kPageRows / kTileRows;

struct CacheStrides {
    uint64_t page;
    uint32_t token;
    uint32_t head;
};

struct StagingTile {
    uint64_t inputOffset;
    uint64_t scaleOffset;
    uint64_t outputOffset;
    bool isValue;
    bool valid;
};

#ifdef __DAV_VEC__
// FP32 scaling and round-to-nearest-even BF16 conversion use the same
// register layouts as the in-attention FP8 producer.
__simd_vf__ inline void ConvertRows(__ubuf__ bfloat16_t* out, __ubuf__ uint8_t* input,
                                   __ubuf__ float* scales, uint32_t scalePitch)
{
    using namespace AscendC::Reg;
    static constexpr CastTrait toFloatEven = {RegLayout::ZERO, SatMode::NO_SAT,
                                       MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
    static constexpr CastTrait toFloatOdd = {RegLayout::ONE, SatMode::NO_SAT,
                                      MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
    static constexpr CastTrait toBf16Even = {RegLayout::ZERO, SatMode::SAT,
                                      MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16Odd = {RegLayout::ONE, SatMode::SAT,
                                     MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    MaskReg full = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg mask128 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg mask64 = CreateMask<uint16_t, MaskPattern::VL64>();
    for (uint16_t row = 0; row < kTileRows; ++row) {
        RegTensor<float8_e4m3_t> first128, last64;
        RegTensor<float> even, odd, tail, scale;
        RegTensor<bfloat16_t> bfEven, bfOdd, bfTail, packed128, packed64, unused;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, scales + row * scalePitch);
        LoadAlign<uint8_t, LoadDist::DIST_UNPACK_B16>((RegTensor<uint8_t>&)first128,
                                                     input + row * kHeadDim);
        LoadAlign<float8_e4m3_t, LoadDist::DIST_UNPACK4_B8>(
            last64, reinterpret_cast<__ubuf__ float8_e4m3_t*>(input + row * kHeadDim + 128U));
        Cast<float, float8_e4m3_t, toFloatEven>(even, first128, full);
        Cast<float, float8_e4m3_t, toFloatOdd>(odd, first128, full);
        Cast<float, float8_e4m3_t, toFloatEven>(tail, last64, full);
        Mul(even, even, scale, full);
        Mul(odd, odd, scale, full);
        Mul(tail, tail, scale, full);
        Cast<bfloat16_t, float, toBf16Even>(bfEven, even, full);
        Cast<bfloat16_t, float, toBf16Odd>(bfOdd, odd, full);
        Cast<bfloat16_t, float, toBf16Even>(bfTail, tail, full);
        Or<uint16_t>((RegTensor<uint16_t>&)packed128, (RegTensor<uint16_t>&)bfEven,
                     (RegTensor<uint16_t>&)bfOdd, mask128);
        DeInterleave(packed64, unused, bfTail, bfTail);
        StoreAlign<bfloat16_t, StoreDist::DIST_NORM_B16>(out + row * kHeadDim, packed128, mask128);
        StoreAlign<bfloat16_t, StoreDist::DIST_NORM_B16>(out + row * kHeadDim + 128U, packed64, mask64);
    }
}

__aicore__ inline StagingTile GetTile(uint32_t task, uint32_t heads, uint32_t numPages,
                                    const AscendC::GlobalTensor<int32_t>& pages,
                                    CacheStrides key, CacheStrides value)
{
    const uint32_t row = (task % kTilesPerHead) * kTileRows;
    const bool isValue = (task / kTilesPerHead) % 2U != 0;
    const uint32_t pageHead = task / (2U * kTilesPerHead);
    const uint32_t head = pageHead % heads;
    const int32_t page = pages.GetValue(pageHead / heads);
    if (page < 0 || static_cast<uint32_t>(page) >= numPages) {
        return {0, 0, 0, isValue, false};
    }
    const auto strides = isValue ? value : key;
    const uint64_t sourceRow = static_cast<uint64_t>(page) * kPageRows + row;
    return {
        page * strides.page + static_cast<uint64_t>(head) * strides.head + row * strides.token,
        sourceRow * heads + head,
        (static_cast<uint64_t>(pageHead) * kPageRows + row) * kHeadDim,
        isValue,
        true,
    };
}

__aicore__ inline void LoadTile(const StagingTile& tile, uint32_t heads, uint32_t tokenStride,
                               const AscendC::GlobalTensor<uint8_t>& cache,
                               const AscendC::GlobalTensor<float>& scale,
                               const AscendC::LocalTensor<uint8_t>& input,
                               const AscendC::LocalTensor<float>& scaleBuf)
{
    if (tokenStride == kHeadDim) {
        AscendC::DataCopy(input, cache[tile.inputOffset], kTileElems);
    } else {
        AscendC::DataCopyPad(input, cache[tile.inputOffset],
            AscendC::DataCopyExtParams{kTileRows, kHeadDim, tokenStride - kHeadDim, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    }
    if (heads == 1U) {
        AscendC::DataCopy(scaleBuf, scale[tile.scaleOffset], kTileRows);
    } else if (heads <= 4U) {
        const uint64_t rowStart = tile.scaleOffset - tile.scaleOffset % heads;
        AscendC::DataCopyPad(scaleBuf, scale[rowStart],
            AscendC::DataCopyExtParams{1, kTileRows * heads * uint32_t{sizeof(float)}, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    } else {
        AscendC::DataCopyPad(scaleBuf, scale[tile.scaleOffset],
            AscendC::DataCopyExtParams{kTileRows, sizeof(float), (heads - 1U) * uint32_t{sizeof(float)}, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    }
}
#endif

}  // namespace

extern "C" __global__ __aicore__ __vector__ void StageFP8KV(
    GM_ADDR key, GM_ADDR value, GM_ADDR keyScale, GM_ADDR valueScale, GM_ADDR sourcePages,
    GM_ADDR keyOut, GM_ADDR valueOut, CacheStrides keyStrides, CacheStrides valueStrides,
    uint32_t heads, uint32_t numPages, uint32_t tasks)
{
#ifdef __DAV_VEC__
    using namespace AscendC;
    const uint32_t block = GetBlockIdx();
    const uint32_t blocks = GetBlockNum();
    const uint32_t base = tasks / blocks;
    const uint32_t extra = tasks % blocks;
    const uint32_t start = block * base + (block < extra ? block : extra);
    const uint32_t count = base + (block < extra ? 1U : 0U);

    GlobalTensor<uint8_t> caches[2];
    GlobalTensor<float> scales[2];
    GlobalTensor<bfloat16_t> outputs[2];
    GlobalTensor<int32_t> pages;
    caches[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key));
    caches[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value));
    scales[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keyScale));
    scales[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(valueScale));
    outputs[0].SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(keyOut));
    outputs[1].SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(valueOut));
    pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourcePages));

    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
    constexpr uint32_t kScaleBytes = kTileRows * kScalePitch * sizeof(float);
    constexpr uint32_t kBankBytes = kTileElems * (1 + sizeof(bfloat16_t)) + kScaleBytes;
    static_assert(2 * kBankBytes <= Catlass::Arch::Ascend950::UB_SIZE);
    LocalTensor<uint8_t> inputs[2];
    LocalTensor<float> scaleBufs[2];
    LocalTensor<bfloat16_t> outBufs[2];
    StagingTile tiles[2];

    // Alternate two UB banks. Refill may overlap Vector and MTE3 on the
    // other bank. Wait for MTE3 before reusing a bank.
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const uint32_t offset = slot * kBankBytes;
        inputs[slot] = resource.ubBuf.GetBufferByByte<uint8_t>(offset);
        outBufs[slot] = resource.ubBuf.GetBufferByByte<bfloat16_t>(offset + kTileElems);
        scaleBufs[slot] = resource.ubBuf.GetBufferByByte<float>(offset + kBankBytes - kScaleBytes);
    }
    for (uint32_t step = 0; step <= count; ++step) {
        const uint32_t next = step % 2U;
        if (step < count) {
            if (step >= 2U) {
                WaitFlag<HardEvent::MTE3_MTE2>(next);
            }
            auto& tile = tiles[next];
            tile = GetTile(start + step, heads, numPages, pages, keyStrides, valueStrides);
            if (tile.valid) {
                const auto strides = tile.isValue ? valueStrides : keyStrides;
                LoadTile(tile, heads, strides.token, caches[tile.isValue], scales[tile.isValue],
                         inputs[next], scaleBufs[next]);
            }
            SetFlag<HardEvent::MTE2_V>(next);
        }
        if (step == 0U) {
            continue;
        }
        const uint32_t cur = 1U - next;
        const auto& tile = tiles[cur];
        WaitFlag<HardEvent::MTE2_V>(cur);
        if (tile.valid) {
            const uint32_t scalePitch = heads <= 4U ? heads : kScalePitch;
            const uint32_t scaleHead = heads <= 4U ? tile.scaleOffset % heads : 0U;
            ConvertRows(reinterpret_cast<__ubuf__ bfloat16_t*>(outBufs[cur].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint8_t*>(inputs[cur].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(scaleBufs[cur][scaleHead].GetPhyAddr()),
                        scalePitch);
            SetFlag<HardEvent::V_MTE3>(cur);
            WaitFlag<HardEvent::V_MTE3>(cur);
            DataCopy(outputs[tile.isValue][tile.outputOffset], outBufs[cur], kTileElems);
        }
        SetFlag<HardEvent::MTE3_MTE2>(cur);
    }
    for (uint32_t finished = count > 1U ? count - 2U : 0U; finished < count; ++finished) {
        WaitFlag<HardEvent::MTE3_MTE2>(finished % 2U);
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

void stage_fp8_kv(const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& keyScale, const at::Tensor& valueScale,
                  const at::Tensor& sourcePages, const at::Tensor& keyOut,
                  const at::Tensor& valueOut)
{
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1 && key.dim() == 4 &&
                key.size(1) == kPageRows && key.size(3) == kHeadDim && key.size(2) > 0,
                "FP8 KV staging expects NPU caches of shape [pages, 128, heads, 192]");
    for (const auto* cache : {&key, &value}) {
        TORCH_CHECK(cache->device() == key.device() && cache->scalar_type() == at::kFloat8_e4m3fn &&
                    cache->sizes() == key.sizes(), "K/V must be matching FP8 E4M3FN caches on the same NPU");
        const bool bbnd = cache->stride(1) == key.size(2) * kHeadDim && cache->stride(2) == kHeadDim;
        const bool bnbd = cache->stride(1) == kHeadDim && cache->stride(2) == kPageRows * kHeadDim;
        TORCH_CHECK((bbnd || bnbd) && cache->stride(3) == 1 &&
                    (cache->size(0) <= 1 || cache->stride(0) >= key.size(2) * kPageRows * kHeadDim),
                    "K/V must have packed BBND or BNBD inner strides and disjoint pages");
    }
    for (const auto* scale : {&keyScale, &valueScale}) {
        TORCH_CHECK(scale->device() == key.device() && scale->scalar_type() == at::kFloat &&
                    scale->sizes() == key.sizes().slice(0, 3) && scale->is_contiguous(),
                    "K/V descales must be contiguous FP32 [pages, 128, heads]");
    }
    TORCH_CHECK(sourcePages.device() == key.device() && sourcePages.scalar_type() == at::kInt &&
                sourcePages.dim() == 1 && sourcePages.is_contiguous(),
                "Source pages must be a contiguous NPU int32 vector");
    TORCH_CHECK(key.size(0) > 0 && key.size(0) <= std::numeric_limits<int32_t>::max() &&
                key.size(2) <= std::numeric_limits<uint32_t>::max() / kHeadDim &&
                sourcePages.numel() <= std::numeric_limits<uint32_t>::max() / (key.size(2) * 2U * kTilesPerHead),
                "FP8 KV staging dimensions exceed kernel limits");
    const auto inputs = {&key, &value, &keyScale, &valueScale, &sourcePages};
    for (const auto* out : {&keyOut, &valueOut}) {
        TORCH_CHECK(out->device() == key.device() && out->scalar_type() == at::kBFloat16 &&
                    out->dim() == 4 && out->size(0) == std::max<int64_t>(sourcePages.numel(), 1) &&
                    out->sizes().slice(1) == key.sizes().slice(1) && out->stride(3) == 1 &&
                    out->stride(1) == kHeadDim &&
                    (key.size(2) == 1 || out->stride(2) == kPageRows * kHeadDim) &&
                    (out->size(0) == 1 || out->stride(0) == key.size(2) * kPageRows * kHeadDim),
                    "Staging outputs must be packed BNBD BF16 [compact_pages, 128, heads, 192]");
        for (const auto* input : inputs) {
            TORCH_CHECK(!out->is_alias_of(*input), "Staging outputs must not alias the inputs");
        }
    }
    TORCH_CHECK(!keyOut.is_alias_of(valueOut), "K and V staging outputs must have separate storage");
    const auto tasks = static_cast<uint32_t>(sourcePages.numel() * key.size(2) * 2 * kTilesPerHead);
    if (tasks == 0) {
        return;
    }
    const c10_npu::NPUGuard guard(key.device());
    const auto stream = c10_npu::getCurrentNPUStream();
    const auto handle = stream.stream(false);
    const uint32_t blocks = std::min(tasks, platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    TORCH_CHECK(blocks > 0, "FP8 KV staging requires an available AIV core");
    const auto ptr = [](const at::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    const auto strides = [](const at::Tensor& t) {
        return CacheStrides{static_cast<uint64_t>(t.stride(0)), static_cast<uint32_t>(t.stride(1)),
                            static_cast<uint32_t>(t.stride(2))};
    };
    const auto keyPtr = ptr(key), valuePtr = ptr(value), keyScalePtr = ptr(keyScale), valueScalePtr = ptr(valueScale);
    const auto pagePtr = ptr(sourcePages), keyOutPtr = ptr(keyOut), valueOutPtr = ptr(valueOut);
    const auto keyStrides = strides(key), valueStrides = strides(value);
    const auto heads = static_cast<uint32_t>(key.size(2));
    const auto numPages = static_cast<uint32_t>(key.size(0));
    const auto launch = [=]() -> int {
        StageFP8KV<<<blocks, nullptr, handle>>>(keyPtr, valuePtr, keyScalePtr, valueScalePtr, pagePtr,
            keyOutPtr, valueOutPtr, keyStrides, valueStrides, heads, numPages, tasks);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_stage_fp8_kv", launch);
    for (const auto* t : {&key, &value, &keyScale, &valueScale, &sourcePages, &keyOut, &valueOut}) {
        c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(), stream);
    }
}
