
#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"

/// ${include_header}
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 2201 // Arch guard

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyGmToL1Test : public AscendCTest {
protected:
    void SetUp() override { AscendCTest::SetUp(); }

    template <class Element>
    void setShape(uint32_t row, uint32_t col) {
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
        _row = row; _col = col;
        _row_round = RoundUp<C0_NUM_PER_FRACTAL>(_row);
        _col_round = RoundUp<ELE_NUM_PER_C0>(_col);
    }

protected:
    uint32_t _row = 128, _col = 256;
    uint32_t _row_round = _0, _col_round = _0;
};

// Put your unittest case here

#endif // CATLASS_ARCH == 2201