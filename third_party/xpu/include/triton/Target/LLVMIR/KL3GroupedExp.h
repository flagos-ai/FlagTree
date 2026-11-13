#pragma once

#include "llvm/IR/PassManager.h"

namespace llvm {
class KL3GroupedExpPass : public PassInfoMixin<KL3GroupedExpPass> {
public:
  PreservedAnalyses run(Module &module, ModuleAnalysisManager &);
};
} // namespace llvm
