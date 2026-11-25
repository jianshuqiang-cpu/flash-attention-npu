#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace tla::tools {

void registerTlaCompileDialectsAndTranslations(mlir::DialectRegistry& registry);

void registerTlaCompileTranslationsAndInterfaces(mlir::DialectRegistry& registry);

void loadTlaCompileDialects(mlir::MLIRContext& context);

void buildTlaCompilePassManagers(mlir::MLIRContext& context, mlir::PassManager& tlaPm, mlir::PassManager& llvmPm);

bool runTlaCompilePipelinesWithManagers(
    mlir::ModuleOp module, llvm::StringRef emitMode, mlir::PassManager& tlaPm, mlir::PassManager& llvmPm,
    std::string& output, std::string& error, bool rewriteTileSignaturesToLLVMPointer);

bool runTlaCompilePipelines(mlir::ModuleOp module, llvm::StringRef emitMode, std::string& output, std::string& error);

} // namespace tla::tools
