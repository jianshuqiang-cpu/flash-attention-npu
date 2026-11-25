#pragma once

#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/IR/Attributes.h"
// Enums.h.inc forward-declares mlir::FieldParser with two template parameters and no default;
// that hides the real primary template from DialectImplementation.h and breaks ODS-generated
// FieldParser<T>::parse (one argument). Include the full FieldParser API first.
#include "mlir/IR/DialectImplementation.h"

#include "tla/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tla/Attrs.h.inc"
