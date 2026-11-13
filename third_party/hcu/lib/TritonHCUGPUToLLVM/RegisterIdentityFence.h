#ifndef FLAGTREE_HCU_REGISTER_IDENTITY_FENCE_H
#define FLAGTREE_HCU_REGISTER_IDENTITY_FENCE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include <string>

namespace flagtree::hcu {

// These exact instructions preserve every explicitly tied VGPR and do not
// access either async queue or LDS. Keep the original call, memory clobber,
// priority instruction and SSA dependencies when scheduling its register group.
// Arbitrary asm and nonidentity constraints remain opaque memory events.
inline bool isRegisterIdentityFence(const llvm::Instruction *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  auto *assembly = call ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand())
                        : nullptr;
  if (!assembly || !assembly->hasSideEffects() || assembly->isAlignStack() ||
      assembly->canThrow() || call->hasOperandBundles() ||
      call->isMustTailCall() || call->getAttributes().hasRetAttrs())
    return false;
  auto code = llvm::StringRef(assembly->getAsmString()).trim();
  if (!code.empty() && code != "s_setprio 0" && code != "s_setprio 1")
    return false;
  llvm::SmallVector<llvm::Type *> results;
  if (auto *structure = llvm::dyn_cast<llvm::StructType>(call->getType())) {
    if (structure->isPacked() || structure->isOpaque())
      return false;
    results.append(structure->element_begin(), structure->element_end());
  } else {
    results.push_back(call->getType());
  }
  if (results.empty() || results.size() != call->arg_size())
    return false;
  std::string constraints;
  for (unsigned i = 0; i < results.size(); ++i) {
    auto *type = results[i];
    auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type);
    if (!(type->isFloatTy() ||
          (vector && vector->getNumElements() == 4 &&
           vector->getElementType()->isFloatTy())) ||
        call->getArgOperand(i)->getType() != type ||
        call->getAttributes().hasParamAttrs(i))
      return false;
    constraints += "=v,";
  }
  for (unsigned i = 0; i < results.size(); ++i)
    constraints += std::to_string(i) + ",";
  constraints += "~{memory}";
  return assembly->getConstraintString() == constraints;
}

} // namespace flagtree::hcu
#endif
