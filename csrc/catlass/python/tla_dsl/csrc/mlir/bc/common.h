#pragma once

#include <cstddef>
#include <cstdint>

#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define CEIL_FACTOR(x, y) (((x) + ((y) - 1)) / (y) * (y))

constexpr int32_t UB_ALIGN_BYTES = 32;

template <typename T, size_t Dim>
struct memref_t {
    T* allocated;
    T* aligned;
    int64_t offset;
    int64_t sizes[Dim];
    int64_t strides[Dim];
};
