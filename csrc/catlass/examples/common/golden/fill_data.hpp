/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <stack>
#include <vector>

#include "catlass/gemm_coord.hpp"

namespace Catlass::golden {

template <class Element, class ElementRandom>
void GenRandomData(Element& data, ElementRandom low, ElementRandom high)
{
    ElementRandom randomValue =
        low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low);
    data = static_cast<Element>(randomValue);
}

template <class Element, class ElementRandom>
void FillRandomData(std::vector<Element>& data, ElementRandom low, ElementRandom high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        ElementRandom randomValue =
            low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low);
        data[i] = static_cast<Element>(randomValue);
    }
}

template <>
void FillRandomData<int8_t, int>(std::vector<int8_t>& data, int low, int high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        int randomValue = low + rand() % (high - low + 1);
        data[i] = static_cast<int8_t>(randomValue);
    }
}

template <class Element, class ElementRandom>
void FillTriangularData(
    std::vector<Element>& data, uint32_t rows, uint32_t columns, uint32_t uplo, uint32_t diag, ElementRandom low,
    ElementRandom high, Element alpha = static_cast<Element>(1))
{
    data.resize(static_cast<std::size_t>(rows) * columns);
    FillRandomData(data, low, high);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
            bool keep = (uplo == 0) ? (j <= i) : (j >= i);
            auto index = static_cast<std::size_t>(i) * columns + j;
            if (!keep) {
                data[index] = static_cast<Element>(0);
                continue;
            }
            data[index] = static_cast<Element>(data[index] * alpha);
            if (diag != 0 && i == j) {
                data[index] = alpha;
            }
        }
    }
}

template <class Element, class ElementRandom>
void FillSymmRandomData(
    std::vector<Element>& data, uint32_t dim, bool upperStorage, ElementRandom low, ElementRandom high)
{
    FillRandomData(data, low, high);
    for (uint32_t r = 0; r < dim; ++r) {
        for (uint32_t c = 0; c < r; ++c) {
            if (upperStorage) {
                data[r * dim + c] = data[c * dim + r];
            } else {
                data[c * dim + r] = data[r * dim + c];
            }
        }
    }
}

template <typename T>
void QuickSort(std::vector<T>& arr, int left, int right)
{
    std::stack<std::pair<int, int>> stk;
    stk.push({left, right});

    while (!stk.empty()) {
        auto [l, r] = stk.top();
        stk.pop();

        if (l >= r) {
            continue;
        }

        T pivot = arr[(l + r) / 2];
        int i = l;
        int j = r;
        while (i <= j) {
            while (arr[i] < pivot) {
                i++;
            }
            while (arr[j] > pivot) {
                j--;
            }
            if (i <= j) {
                std::swap(arr[i], arr[j]);
                i++;
                j--;
            }
        }

        if (l < j) {
            stk.push({l, j});
        }
        if (i < r) {
            stk.push({i, r});
        }
    }
}

// Generate an ascending random sequence as grouplist
// If CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED is defined, convert to segmented form
// Otherwise, keep prefix-sum form
template <typename T = int32_t>
std::vector<T> GenerateGroupList(uint32_t m, uint32_t problemCount)
{
    std::vector<T> groupList(problemCount);
    std::srand(0);
    for (int i = 0; i < problemCount; ++i) {
        groupList[i] = rand() % (m + 1);
    }
    QuickSort(groupList, 0, groupList.size() - 1);

#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
    for (int i = problemCount - 1; i > 0; --i) {
        groupList[i] -= groupList[i - 1];
    }
#endif

    return groupList;
}

// Generate an ascending random sequence as grouplist
template <typename T = int32_t, bool isCumsum = true>
std::vector<T> GenerateAverageGroupList(uint32_t m, uint32_t problemCount)
{
    uint32_t groupSize = m / problemCount;
    std::vector<T> groupList(problemCount);
    for (int i = 0; i < problemCount; ++i) {
        if constexpr (isCumsum) {
            groupList[i] = groupSize * (i + 1);
        } else {
            groupList[i] = groupSize;
        }
    }

    return groupList;
}

template <class ElementC, class ElementGolden, class LayoutTagDst, class LayoutTagSrc>
void fillContigunousBatchedData(
    uint32_t batchCount, const GemmCoord& problemShape, std::vector<ElementGolden>& dstData, const LayoutTagDst& tagDst,
    const std::vector<ElementC>& srcData, const LayoutTagSrc& tagSrc, int64_t strideDst, int64_t strideSrc)
{
    if (strideDst == -1) { /* default to m*n if dst set to -1 */
        strideDst = static_cast<int64_t>(problemShape.m()) * problemShape.n();
    }
    size_t offsetBatchDst = 0;
    size_t offsetBatchSrc = 0;
    for (uint32_t b = 0; b < batchCount; ++b) {
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t srcOffset = offsetBatchSrc + tagSrc.GetOffset(MakeCoord(i, j));
                size_t dstOffset = offsetBatchDst + tagDst.GetOffset(MakeCoord(i, j));
                dstData[dstOffset] = static_cast<ElementGolden>(srcData[srcOffset]);
            }
        }
        offsetBatchDst += strideDst;
        offsetBatchSrc += strideSrc;
    }
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
