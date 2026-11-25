/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_OPTIONS_HPP
#define EXAMPLES_COMMON_OPTIONS_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef CATLASS_EXAMPLE_NAME
#define CATLASS_EXAMPLE_NAME catlass_example
#endif

/**
 * @struct GemmOptions
 * @brief Options structuture for gemm examples.
 * @brief Arguments: `example_name m n k [device_id]`
 */
struct GemmOptions {
    const std::string HELPER = "m n k [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    GemmOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

/**
 * @struct GemvOptions
 * @brief Options structuture for gemv examples.
 * @brief Arguments: `example_name m n [device_id]`
 */
struct GemvOptions {
    const std::string HELPER = "m n [device_id]";

    Catlass::GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    GemvOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

/**
 * @struct GroupedGemmOptions
 * @brief Options structuture for grouped/batched gemm examples.
 * @brief Arguments: `example_name problem_count m n k [device_id]`
 */
struct GroupedGemmOptions {
    const std::string HELPER = "problem_count m n k [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t problemCount{1};
    int32_t deviceId{0};

    GroupedGemmOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            GROUP_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

/**
 * @struct TrmmOptions
 * @brief Options structure for TRMM examples.
 * @brief Arguments: `example_name m n side uplo trans diag alpha [device_id]`
 */
struct TrmmOptions {
    const std::string HELPER = "m n side uplo trans diag alpha [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t side{0};
    uint32_t uplo{0};
    uint32_t trans{0};
    uint32_t diag{0};
    float alpha{1.0f};
    int32_t deviceId{0};

    TrmmOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            SIDE_INDEX,
            UPLO_INDEX,
            TRANS_INDEX,
            DIAG_INDEX,
            ALPHA_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::ALPHA_INDEX) + 1) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        side = static_cast<uint32_t>(std::atoi(argv[static_cast<uint32_t>(ArgsIndex::SIDE_INDEX)]));
        uplo = static_cast<uint32_t>(std::atoi(argv[static_cast<uint32_t>(ArgsIndex::UPLO_INDEX)]));
        trans = static_cast<uint32_t>(std::atoi(argv[static_cast<uint32_t>(ArgsIndex::TRANS_INDEX)]));
        diag = static_cast<uint32_t>(std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DIAG_INDEX)]));
        alpha = std::atof(argv[static_cast<uint32_t>(ArgsIndex::ALPHA_INDEX)]);

        problemShape.k() = (side == 0) ? problemShape.m() : problemShape.n();

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

#endif
