#include "TritonMUSALLVMIR/MTGPUTargetTransformInfo.h"
#include "TritonMUSALLVMIR/Passes.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace mlir::triton::musa {

bool registerMTGPUTargetIRAnalysis(FunctionAnalysisManager &FAM,
                                   StringRef TargetTriple,
                                   StringRef Processor) {
  return llvm::mtgpu::registerMTGPUTargetIRAnalysis(FAM, TargetTriple,
                                                    Processor);
}

void registerMTGPUPostSLPCallbacks(PassBuilder &PB) {
  PB.registerOptimizerLastEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) {
        FunctionPassManager FPM;
        FPM.addPass(MTGPUPostSLPRepairPass());
        FPM.addPass(MTGPUCanonicalizeVectorIntrinsicsPass());
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
}

} // namespace mlir::triton::musa
