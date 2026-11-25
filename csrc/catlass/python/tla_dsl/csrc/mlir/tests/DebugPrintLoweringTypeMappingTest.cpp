#include <cstdio>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Passes/PassesInternal.h"

namespace {

bool check(mlir::Type type, tla::DebugPrintPlacement placement,
           llvm::StringRef expected)
{
    std::string diagnostic;
    auto spec = tla::getDebugPrintHelperSpec(type, placement, diagnostic);
    if (succeeded(spec) && spec->callee == expected && diagnostic.empty())
        return true;
    std::fprintf(stderr, "debug_print helper mapping failed for %s\n",
                 expected.str().c_str());
    return false;
}

} // namespace

int main()
{
    mlir::MLIRContext context;
    using Signedness = mlir::IntegerType::SignednessSemantics;
    const std::pair<mlir::Type, llvm::StringRef> supported[] = {
        {mlir::IntegerType::get(&context, 8),
         "_mlir_ciface_tla_printf_x_i8"},
        {mlir::IntegerType::get(&context, 16),
         "_mlir_ciface_tla_printf_x_i16"},
        {mlir::IntegerType::get(&context, 32),
         "_mlir_ciface_tla_printf_x_i32"},
        {mlir::IntegerType::get(&context, 8, Signedness::Unsigned),
         "_mlir_ciface_tla_printf_x_u8"},
        {mlir::IntegerType::get(&context, 16, Signedness::Unsigned),
         "_mlir_ciface_tla_printf_x_u16"},
        {mlir::IntegerType::get(&context, 32, Signedness::Unsigned),
         "_mlir_ciface_tla_printf_x_u32"},
        {mlir::Float16Type::get(&context),
         "_mlir_ciface_tla_printf_v_f16"},
        {mlir::Float32Type::get(&context),
         "_mlir_ciface_tla_printf_v_f32"},
    };
    for (auto [type, callee] : supported) {
        if (!check(type, tla::DebugPrintPlacement::Cube, callee) ||
            !check(type, tla::DebugPrintPlacement::Vector, callee))
            return 1;
    }

    std::string diagnostic;
    auto invalid = tla::getDebugPrintHelperSpec(
        mlir::IntegerType::get(&context, 32),
        tla::DebugPrintPlacement::Invalid, diagnostic);
    if (succeeded(invalid) ||
        diagnostic !=
            "unsupported tla.debug_print placement invalid for dtype i32; "
            "expected cube or vector") {
        std::fprintf(stderr, "debug_print invalid placement diagnostic failed\n");
        return 1;
    }
    return 0;
}
