#include <cstdio>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Passes/PassesInternal.h"

namespace {
constexpr llvm::StringLiteral kSupported =
    "f16, f32, i8, i16, i32, u8, u16, u32";

bool check(mlir::Type type, llvm::StringRef expected, bool accepted)
{
    std::string diagnostic;
    auto suffix = tla::getPrintTensorHelperSuffix(type, diagnostic);
    bool valid = accepted
        ? succeeded(suffix) && *suffix == expected && diagnostic.empty()
        : failed(suffix) && diagnostic ==
              "unsupported dtype " + expected.str() +
                  "; supported dtypes: " + kSupported.str();
    if (!valid)
        std::fprintf(stderr, "print_tensor type mapping failed for %s\n",
                     expected.str().c_str());
    return valid;
}
} // namespace

int main()
{
    mlir::MLIRContext context;
    using Signedness = mlir::IntegerType::SignednessSemantics;
    const std::pair<mlir::Type, llvm::StringRef> supported[] = {
        {mlir::Float16Type::get(&context), "f16"},
        {mlir::Float32Type::get(&context), "f32"},
        {mlir::IntegerType::get(&context, 8), "i8"},
        {mlir::IntegerType::get(&context, 16), "i16"},
        {mlir::IntegerType::get(&context, 32), "i32"},
        {mlir::IntegerType::get(&context, 8, Signedness::Unsigned), "u8"},
        {mlir::IntegerType::get(&context, 16, Signedness::Unsigned), "u16"},
        {mlir::IntegerType::get(&context, 32, Signedness::Unsigned), "u32"},
    };
    const std::pair<mlir::Type, llvm::StringRef> rejected[] = {
        {mlir::BFloat16Type::get(&context), "bf16"},
        {mlir::IntegerType::get(&context, 1), "i1"},
        {mlir::IntegerType::get(&context, 64), "i64"},
        {mlir::IntegerType::get(&context, 64, Signedness::Unsigned), "u64"},
        {mlir::IntegerType::get(&context, 8, Signedness::Signed), "si8"},
        {mlir::IntegerType::get(&context, 16, Signedness::Signed), "si16"},
        {mlir::IntegerType::get(&context, 32, Signedness::Signed), "si32"},
        {mlir::IndexType::get(&context), "index"},
        {mlir::NoneType::get(&context), "none"},
    };
    for (auto [type, suffix] : supported)
        if (!check(type, suffix, true))
            return 1;
    for (auto [type, spelling] : rejected)
        if (!check(type, spelling, false))
            return 1;
    return 0;
}
