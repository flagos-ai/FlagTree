#include "triton/Target/LLVMIR/KL3GroupedExp.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

// The standalone plugin and FlagTree's built-in pipeline run the same pass.
extern "C" LLVM_ATTRIBUTE_WEAK LLVM_EXTERNAL_VISIBILITY llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "KL3GroupedExp", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &builder) {
            builder.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::ModulePassManager &passes,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (name != "kl3-grouped-exp")
                    return false;
                  passes.addPass(llvm::KL3GroupedExpPass());
                  return true;
                });
            builder.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager &passes, llvm::OptimizationLevel) {
                  passes.addPass(llvm::KL3GroupedExpPass());
                });
          }};
}
