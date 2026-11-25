#ifndef OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_TLA_H
#define OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_TLA_H

#include <type_traits>
#include "tla/tensor.hpp"
#include "common/tile_shape_scaler.h"

namespace CatlassKernel {

template <typename Shape, uint32_t KScale>
struct ScaleKTileTLA {
    static_assert(KScale > 0, "KScale must be positive");
    using type = Shape;
};

template <uint32_t M, uint32_t N, uint32_t K, uint32_t KScale>
struct ScaleKTileTLA<tla::tuple<tla::C<M>, tla::C<N>, tla::C<K>>, KScale> {
    static_assert(K % KScale == 0, "K must be divisible by KScale");
    using type = tla::tuple<tla::C<M>, tla::C<N>, tla::C<K / KScale>>;
};

template <typename ActualDtype, typename BaselineDtype, typename BaseShape>
struct TileShapeScalerTLA {
    static constexpr uint32_t kScale = KScaleFactor<ActualDtype, BaselineDtype>::value;
    using type = typename ScaleKTileTLA<BaseShape, kScale>::type;
};

}  // namespace CatlassKernel

#endif  // OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_TLA_H
