/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_LIBRARY_HELPER_H
#define CATLASS_TUNER_LIBRARY_HELPER_H

#include <unordered_map>
#include <functional>

#include "log.h"
#include "catlass/library/operation.h"
#include "catlass/layout/matrix.hpp"
#include "catlass/layout/vector.hpp"

namespace Catlass {

class LibraryHelper {
public:
    using DataType = Library::DataType;
    using LayoutType = Library::LayoutType;

    static size_t GetDataTypeSize(DataType dataType);
    static size_t GetLayoutSize(LayoutType layoutType);
    static std::string_view GetDataTypeStr(DataType dataType);
    static std::string_view GetLayoutStr(LayoutType layoutType);
    static DataType GetDataTypeEnum(std::string_view str);
    static LayoutType GetLayoutEnum(std::string_view str);
    static void ConstructLayout(LayoutType layoutType, DataType dataType, uint32_t a, uint32_t b, uint8_t *data);

    template <typename LayoutTag, typename Element>
    static size_t GetLayoutCapacity(uint32_t x, uint32_t y)
    {
        return LayoutTag::template MakeLayout<Element>(x, y).Capacity();
    }

    static size_t GetLayoutCapacity(LayoutType layout, DataType type,
        uint32_t x, uint32_t y)
    {
        static const std::unordered_map<LayoutType, 
            std::unordered_map<DataType, std::function<size_t(uint32_t, uint32_t)>>> dispatchTable = {
            {LayoutType::RowMajor,{
                {DataType::Fp32, &GetLayoutCapacity<layout::RowMajor, float>},
                {DataType::Int32, &GetLayoutCapacity<layout::RowMajor, int32_t>}
            }},
            {LayoutType::ColumnMajor, {
                {DataType::Fp32, &GetLayoutCapacity<layout::ColumnMajor, float>},
                {DataType::Int32, &GetLayoutCapacity<layout::ColumnMajor, int32_t>}
            }},
            {LayoutType::nZ, {
                {DataType::Fp32, &GetLayoutCapacity<layout::nZ, float>},
                {DataType::Int32, &GetLayoutCapacity<layout::nZ, int32_t>}
            }},
            {LayoutType::zN, {
                {DataType::Fp32, &GetLayoutCapacity<layout::zN, float>},
                {DataType::Int32, &GetLayoutCapacity<layout::zN, int32_t>}
            }}
        };
        auto it1 = dispatchTable.find(layout);
        if (it1 == dispatchTable.end()) {
            LOGE("Invalid layout");
            return 0U;
        }
        auto it2 = it1->second.find(type);
        if (it2 == it1->second.end()) {
            LOGE("Invalid datatype");
            return 0U;
        }
        return it2->second(x, y);
    }
};

}
#endif // CATLASS_TUNER_LIBRARY_HELPER_H
