/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_TEST_FIXTURE_H
#define ASCENDC_STUB_TEST_FIXTURE_H

#include <gtest/gtest.h>
#include "stub/ascendc_logger.h"
#include "stub/kernel_tensor.h"
#include "stub/kernel_struct_mm.h"
#include "stub/kernel_struct_unary.h"
#include "stub/kernel_operator_data_copy_ext.h"
#include "stub/kernel_operator_vec_vconv_intf.h"

class AscendCTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASCENDC_CLEAR_LOGS();
    }

    void TearDown() override
    {
        ASCENDC_CLEAR_LOGS();
    }
};

/// TileCopyTest, for AIC related data copy tile-level ut
class TileCopyTest : public AscendCTest {
protected:
    template <class Element, bool isTrans = false>
    void _setShape(uint32_t row, uint32_t col)
    {
        constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
        constexpr uint32_t ELE_NUM_PER_C0 = 32 / sizeof(Element);

        _row = row;
        _col = col;
        if constexpr (isTrans) {
            _row_round = (row + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _col_round = (col + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _rows_by_fractal = _row_round / ELE_NUM_PER_C0;
            _cols_by_fractal = _col_round / C0_NUM_PER_FRACTAL;
        } else {
            _row_round = (row + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _col_round = (col + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _rows_by_fractal = _row_round / C0_NUM_PER_FRACTAL;
            _cols_by_fractal = _col_round / ELE_NUM_PER_C0;
        }
    }

protected:
    uint32_t _row = 128;
    uint32_t _col = 256;
    uint32_t _row_round = 0;
    uint32_t _col_round = 0;
    uint32_t _rows_by_fractal = 0;
    uint32_t _cols_by_fractal = 0;
    // call 'setShape' firstly in every testcase
};

class TileCopyTlaTest : public AscendCTest {
protected:
    template <class Element, bool isSrcTrans = false, bool isDstTrans = false>
    void _setShape(uint32_t row, uint32_t col)
    {
        constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
        constexpr uint32_t ELE_NUM_PER_C0 = 32 / sizeof(Element);

        if constexpr (isSrcTrans) {
            _row_round = (row + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _col_round = (col + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _rows_by_fractal = _row_round / ELE_NUM_PER_C0;
            _cols_by_fractal = _col_round / C0_NUM_PER_FRACTAL;
        } else {
            _row_round = (row + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _col_round = (col + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _rows_by_fractal = _row_round / C0_NUM_PER_FRACTAL;
            _cols_by_fractal = _col_round / ELE_NUM_PER_C0;
        }

        if constexpr (isDstTrans) {
            _dst_row_round = (_dst_row + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _dst_col_round = (_dst_col + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _dst_row_by_fractal = _dst_row_round / ELE_NUM_PER_C0;
            _dst_cols_by_fractal = _dst_col_round / C0_NUM_PER_FRACTAL;
        } else {
            _dst_row_round = (_dst_row + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
            _dst_col_round = (_dst_col + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0 * ELE_NUM_PER_C0;
            _dst_row_by_fractal = _dst_row_round / C0_NUM_PER_FRACTAL;
            _dst_cols_by_fractal = _dst_col_round / ELE_NUM_PER_C0;
        }
    }

    bool isValidDataCopy() const
    {
        return (_row_coord + _dst_row < _row) && (_col_coord + _dst_col < _col);
    }

protected:
    uint32_t _row = 128;
    uint32_t _col = 256;
    uint32_t _row_round = 128;
    uint32_t _col_round = 256;
    uint32_t _rows_by_fractal = 16;
    uint32_t _cols_by_fractal = 32;
    // coord
    uint32_t _row_coord = 16;
    uint32_t _col_coord = 32;
    uint32_t _dst_row = 67;
    uint32_t _dst_col = 128;
    uint32_t _dst_row_round = 80;
    uint32_t _dst_col_round = 128;
    uint32_t _dst_row_by_fractal = 80;
    uint32_t _dst_cols_by_fractal = 128;
};

/// TileMmadTest, for AIC related mmad tile-level ut
class TileMmadTest : public AscendCTest {
protected:
    void _setShape(uint32_t m, uint32_t n, uint32_t k)
    {
        _m = m;
        _n = n;
        _k = k;
    }

protected:
    uint32_t _m = 128;
    uint32_t _n = 256;
    uint32_t _k = 128;
};

/// UBTileCopyTest, for AIV related data copy tile-level ut
class UBTileCopyTest : public AscendCTest {
protected:
    void _setShape(const uint32_t blkLen, const uint16_t blkCnt)
    {
        _blkLen = blkLen;
        _blkCnt = blkCnt;
        _totalLen = blkLen * blkCnt;
    }

protected:
    uint32_t _blkLen = 128;
    uint16_t _blkCnt = 1;
    uint32_t _totalLen = 128;
};

#endif // ASCENDC_STUB_TEST_BASE_H
