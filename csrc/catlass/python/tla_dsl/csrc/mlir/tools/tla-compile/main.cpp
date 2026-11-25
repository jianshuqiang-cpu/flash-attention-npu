#include "Passes.h"
#include "Tools/CompilePipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input mlir>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output MLIR file"), llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> allowUnregistered(
    "allow-unregistered-dialect", llvm::cl::desc("Allow parsing unregistered dialects"), llvm::cl::init(true));

static llvm::cl::opt<std::string> emitAction(
    "emit", llvm::cl::desc("Output format"), llvm::cl::value_desc("tlair|tla-mlir|tla-mlir-no-verify|mlir|llvm"),
    llvm::cl::init("mlir"));

static llvm::cl::opt<std::string> printPipeline(
    "print-pipeline", llvm::cl::desc("Print the pass pipeline and exit"), llvm::cl::value_desc("none|llvm|mlir|all"),
    llvm::cl::init("none"));

int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);

    DialectRegistry registry;
    tla::tools::registerTlaCompileDialectsAndTranslations(registry);
    tla::registerTlaPasses();

    mlir::registerPassManagerCLOptions();
    llvm::cl::ParseCommandLineOptions(argc, argv, "tla-compile\n");

    MLIRContext context(registry);
    context.allowUnregisteredDialects(allowUnregistered);
    tla::tools::loadTlaCompileDialects(context);

    PassManager tlaPm(&context);
    PassManager llvmPm(&context);
    tla::tools::buildTlaCompilePassManagers(context, tlaPm, llvmPm);
    auto _ignoreTla = mlir::applyPassManagerCLOptions(tlaPm);
    auto _ignoreLlvm = mlir::applyPassManagerCLOptions(llvmPm);

    if (printPipeline == "all" || printPipeline == "mlir" || printPipeline == "llvm") {
        if (printPipeline == "all" || printPipeline == "mlir") {
            tlaPm.printAsTextualPipeline(llvm::outs());
            llvm::outs() << "\n";
        }
        if (printPipeline == "all" || printPipeline == "llvm") {
            llvmPm.printAsTextualPipeline(llvm::outs());
            llvm::outs() << "\n";
        }
        return 0;
    } else if (printPipeline != "none") {
        llvm::errs() << "--print-pipeline flag not recognized. Available values: {none, mlir, llvm, all}.\n";
        return 1;
    }

    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << "Failed to open input: " << errorMessage << "\n";
        return 1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    ParserConfig parseConfig(&context);
    OwningOpRef<ModuleOp> moduleRef = parseSourceFile<ModuleOp>(sourceMgr, parseConfig);
    if (!moduleRef) {
        llvm::errs() << "Failed to parse input MLIR.\n";
        return 1;
    }

    if (emitAction == "tlair") {
        std::string errorMessage;
        std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile(outputFilename, &errorMessage);
        if (!outFile) {
            llvm::errs() << "Failed to open output: " << errorMessage << "\n";
            return 1;
        }
        moduleRef->print(outFile->os());
        outFile->keep();
        return 0;
    }

    if (emitAction == "tla-mlir" || emitAction == "tla-mlir-no-verify") {
        PassManager frontendPm(&context);
        ::tla::buildTlaPipeline(frontendPm);
        if (emitAction == "tla-mlir-no-verify")
            frontendPm.enableVerifier(false);
        auto _ignoreFrontend = mlir::applyPassManagerCLOptions(frontendPm);
        if (failed(frontendPm.run(moduleRef.get()))) {
            llvm::errs() << "Failed to run Tla frontend pipeline.\n";
            return 1;
        }
        std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile(outputFilename, &errorMessage);
        if (!outFile) {
            llvm::errs() << "Failed to open output: " << errorMessage << "\n";
            return 1;
        }
        moduleRef->print(outFile->os());
        outFile->keep();
        return 0;
    }

    std::string outputText;
    std::string pipelineError;
    if (!tla::tools::runTlaCompilePipelinesWithManagers(
            moduleRef.get(), emitAction, tlaPm, llvmPm, outputText, pipelineError,
            /*rewriteTileSignaturesToLLVMPointer=*/true)) {
        llvm::errs() << pipelineError << "\n";
        return 1;
    }

    std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!outFile) {
        llvm::errs() << "Failed to open output: " << errorMessage << "\n";
        return 1;
    }
    outFile->os() << outputText;
    outFile->keep();
    return 0;
}
