#pragma once

// Shared representation and codec for materialized tensor descriptors.
//
// `tla-lower-tensor-desc` owns producer-chain analysis. Downstream passes may
// decode and validate `tla.tensor_desc`, but must not recover descriptor
// metadata by walking raw tensor producers.

#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <string>

namespace tla {

/// Discardable metadata carried by lowered alloc addresses until tensor views
/// have recovered the static allocation capacity. Removed by finalize.
inline constexpr llvm::StringLiteral kAllocSizeBytesMetadataAttrName = "tla.alloc_size_bytes";

/// Bridge a structured `!tla.tensor` type to a builtin memref type.
mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType);

enum class TensorLayoutTag
{
    Unknown,
    RowMajor,
    ColumnMajor,
    zN,
    zZ,
    nZ,
    L0C,
    zNUnAlign
};

bool isNZFamilyLayout(TensorLayoutTag layoutTag);
bool isLinearLayout(TensorLayoutTag layoutTag);
llvm::StringRef stringifyTensorLayoutTag(TensorLayoutTag layoutTag);
mlir::FailureOr<TensorLayoutTag> convertTlaLayoutTag(::LayoutTag layoutTag);
mlir::FailureOr<TensorLayoutTag> parseTensorLayoutTagAttr(llvm::StringRef layouttag);
mlir::FailureOr<TensorLayoutTag> getExplicitTensorLayoutTagAttr(mlir::Operation* op);

/// Static metadata decoded from a structured `!tla.tensor` type.
struct TileTypeInfo {
    llvm::SmallVector<int64_t, 4> shapeDims;
    llvm::SmallVector<int64_t, 4> strideDims;
    llvm::SmallVector<int64_t, 4> coordDims;
    llvm::SmallVector<int64_t, 4> originShapeDims;
    std::string addressSpace;
    std::string elementType;
    mlir::Type mlirElementType;
    ::AddressSpace tlaAddressSpace = ::AddressSpace::gm;
    TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
    int64_t rank = 0;
};

mlir::FailureOr<TileTypeInfo> decodeTileTypeInfo(mlir::Type tileType);

/// Raw, rank-preserving decode used by vector lowering.
struct ParsedTensorInfo {
    llvm::SmallVector<int64_t, 2> shape;
    llvm::SmallVector<int64_t, 2> originShape;
    llvm::SmallVector<int64_t, 2> coord;
    llvm::SmallVector<int64_t, 2> strides;
    ::AddressSpace addressSpace;
    mlir::Type elementType;
    std::string layoutTag;
};

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType);

/// SSA-valued descriptor represented by `tla.tensor_desc`.
struct TensorDescriptor {
    mlir::Value base;
    mlir::Type bridgedBaseMemrefType;
    std::array<mlir::Value, 4> shape;
    std::array<mlir::Value, 4> stride;
    std::array<mlir::Value, 2> originShape;
    std::array<mlir::Value, 2> coord;
    TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
    std::string addrspace;
    std::string elementType;
};

/// Fully dynamic shape/stride form used at structural joins and runtime calls.
mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType);

bool validateTensorDescriptor(mlir::Operation* op, const TensorDescriptor& desc, llvm::StringRef errorMessage);

/// Recover optional allocation capacity preserved by pointer lowering.
mlir::FailureOr<int64_t> getStaticAllocationElementCount(mlir::Value ptr);

/// Decode an already materialized `tla.tensor_desc`.
mlir::FailureOr<TensorDescriptor> descriptorFromTensorDescOp(::tla::TensorDescOp op);

} // namespace tla
