#ifndef TRITON_THIRD_PARTY_MUSA_TRITONMUSALLVMIR_PASSES_H
#define TRITON_THIRD_PARTY_MUSA_TRITONMUSALLVMIR_PASSES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class PassBuilder;
}

namespace mlir::triton::musa {

struct MTGPUPostSLPRepairPass : llvm::PassInfoMixin<MTGPUPostSLPRepairPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "MTGPUPostSLPRepairPass"; }
};

struct MTGPUCanonicalizeVectorIntrinsicsPass
    : llvm::PassInfoMixin<MTGPUCanonicalizeVectorIntrinsicsPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() {
    return "MTGPUCanonicalizeVectorIntrinsicsPass";
  }
};

bool registerMTGPUTargetIRAnalysis(llvm::FunctionAnalysisManager &FAM,
                                   llvm::StringRef TargetTriple,
                                   llvm::StringRef Processor);

void registerMTGPUPostSLPCallbacks(llvm::PassBuilder &PB);

} // namespace mlir::triton::musa
#endif
