// Copyright (c) 2023, Tri Dao.
// Modified by Minghua Shen, 2026
// Precompiled Header for Flash Attention NPU v3 (Ascend 2201)

#ifndef PCH_HPP
#define PCH_HPP

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <type_traits>
#include <memory>
#include <string>
#include <sstream>

#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "tiling/platform/platform_ascendc.h"
#include "fag_tiling.h"

#endif // PCH_HPP
