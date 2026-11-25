#pragma once

#include "mlir/IR/Dialect.h"

namespace tla {

class TlaDialect : public mlir::Dialect {
public:
    explicit TlaDialect(mlir::MLIRContext* context);
    static llvm::StringRef getDialectNamespace()
    {
        return "tla";
    }
    void initialize();
    mlir::Type parseType(mlir::DialectAsmParser& parser) const override;
    void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
    mlir::Attribute parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const override;
    void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const override;
};

} // namespace tla

MLIR_DECLARE_EXPLICIT_TYPE_ID(tla::TlaDialect)
