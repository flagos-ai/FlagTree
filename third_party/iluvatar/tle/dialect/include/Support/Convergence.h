// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#ifndef ILUVATAR_TLE_CONVERGENCE_H
#define ILUVATAR_TLE_CONVERGENCE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace mlir::triton::iluvatar_tle {

// Collective intrinsics carry convergence constraints. Their device-function
// wrappers and callers must retain them too: sinking a wrapper into the two
// sides of a lane predicate changes which lanes rendezvous at each call site.
inline void propagateConvergence(llvm::Module &module) {
  llvm::DenseMap<llvm::Function *, llvm::SmallVector<llvm::Function *, 4>> callers;
  llvm::SmallVector<llvm::Function *> worklist;
  for (llvm::Function &function : module) {
    if (function.hasFnAttribute(llvm::Attribute::Convergent))
      worklist.push_back(&function);
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call)
        continue;
      if (auto *callee = call->getCalledFunction())
        callers[callee].push_back(&function);
      if (call->isConvergent() &&
          !function.hasFnAttribute(llvm::Attribute::Convergent)) {
        function.addFnAttr(llvm::Attribute::Convergent);
        worklist.push_back(&function);
      }
    }
  }
  while (!worklist.empty()) {
    llvm::Function *callee = worklist.pop_back_val();
    for (llvm::Function *caller : callers[callee]) {
      if (caller->hasFnAttribute(llvm::Attribute::Convergent))
        continue;
      caller->addFnAttr(llvm::Attribute::Convergent);
      worklist.push_back(caller);
    }
  }
  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function))
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
        if (auto *callee = call->getCalledFunction())
          if (callee->hasFnAttribute(llvm::Attribute::Convergent))
            call->setConvergent();
}

} // namespace mlir::triton::iluvatar_tle
#endif
