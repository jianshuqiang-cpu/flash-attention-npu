#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#define GET_TYPEDEF_CLASSES
#include "tla/Types.h.inc"

namespace tla {
::mlir::LogicalResult getTlaIndexTreeLeaves(::llvm::ArrayRef<int64_t> tree, ::llvm::SmallVectorImpl<int64_t>& leaves);

/// The single source of truth for layouts whose shape and stride each have
/// four physical leaves in tensor descriptors.
inline bool isNZFamilyLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::zN || layoutTag == ::LayoutTag::nZ || layoutTag == ::LayoutTag::zZ ||
           layoutTag == ::LayoutTag::L0Clayout || layoutTag == ::LayoutTag::zNUnAlign;
}

inline int64_t getByteSizeOfFixedWidthScalarType(::mlir::Type type)
{
    if (type.isBF16() || type.isF16())
        return 2;
    if (type.isF32())
        return 4;
    if (type.isF64())
        return 8;
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
        if (intTy.getWidth() % 8 == 0)
            return intTy.getWidth() / 8;
    }
    return 0;
}

using coord = ::mlir::Type;
using cross_flag = ::mlir::Type;
using flag = ::mlir::Type;
using index = ::mlir::IndexType;
using memref = ::mlir::Type;
using mutex = ::mlir::Type;
using range = ::mlir::Type;
using shape = ::mlir::Type;
using stride = ::mlir::Type;
using layout = ::mlir::Type;
using tensor = ::mlir::Type;
using tile = ::mlir::Type;
// PtrType is defined in Types.h.inc (TableGen TypeDef).
} // namespace tla
