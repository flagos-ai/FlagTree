#ifndef LLVM_LIB_TARGET_MTGPU_MTGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_MTGPU_MTGPUTARGETTRANSFORMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace mtgpu {

bool registerMTGPUTargetIRAnalysis(FunctionAnalysisManager &FAM,
                                   StringRef TargetTriple, StringRef Processor);

}
} // namespace llvm
#endif
