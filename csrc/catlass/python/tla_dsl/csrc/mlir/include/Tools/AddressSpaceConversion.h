#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace tla {

inline mlir::FailureOr<int64_t> mapTlaAddressSpaceToMlirMemRefSpaceValue(AddressSpace addressSpace)
{
    switch (addressSpace) {
        case AddressSpace::generic:
            return 0;
        case AddressSpace::gm:
            return 1;
        case AddressSpace::l1:
            return 2;
        case AddressSpace::l0a:
            return 3;
        case AddressSpace::l0b:
            return 4;
        case AddressSpace::l0c:
            return 5;
        case AddressSpace::ub:
            return 6;
    }
    return mlir::failure();
}

/// True when memref carries GM memory space (hivm GM attr or legacy int 1).
inline bool isGmMemRef(mlir::MemRefType memrefType)
{
    mlir::Attribute memorySpace = memrefType.getMemorySpace();
    if (!memorySpace)
        return false;
    if (auto hivmSpace = mlir::dyn_cast<mlir::hivm::AddressSpaceAttr>(memorySpace))
        return hivmSpace.getAddressSpace() == mlir::hivm::AddressSpace::GM;
    if (auto intSpace = mlir::dyn_cast<mlir::IntegerAttr>(memorySpace))
        return intSpace.getInt() == 1;
    return false;
}

} // namespace tla
