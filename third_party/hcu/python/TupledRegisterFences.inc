// Preserve MMAC-sized register tuples across register-preserving inline fences.
// No instruction, memory effect, dependency, or scheduling boundary is removed.
#include "llvm/IR/Verifier.h"

namespace {
struct TupledRegisterFenceStats {
  unsigned sites = 0, scalarLanes = 0, rejected = 0;
};

static TupledRegisterFenceStats
tupleRegisterFences(llvm::Module *module, const std::string &arch) {
  TupledRegisterFenceStats stats;
  if (!module || arch != "gfx936")
    return stats;
  llvm::SmallVector<llvm::CallInst *> calls;
  for (auto &fn : *module) {
    if (fn.isDeclaration() || llvm::verifyFunction(fn))
      continue;
    for (auto &bb : fn)
      for (auto &inst : bb)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (call->isInlineAsm())
            calls.push_back(call);
  }
  for (auto *call : calls) {
    auto *assembly = llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
    auto *result = llvm::dyn_cast<llvm::StructType>(call->getType());
    if (!result || result->isPacked() || result->getNumElements() < 4)
      continue;
    const unsigned count = result->getNumElements();
    auto code = llvm::StringRef(assembly->getAsmString()).trim();
    // These scalar instructions cannot write any of the tied vector operands.
    // Operand-referencing asm, fixed clobbers, and arbitrary asm stay unchanged.
    bool eligible = (code.empty() || code == "s_setprio 0" ||
                     code == "s_setprio 1") &&
                    assembly->hasSideEffects() && !assembly->isAlignStack() &&
                    !assembly->canThrow() && count % 4 == 0 &&
                    call->arg_size() == count && !call->getNumOperandBundles() &&
                    !call->isMustTailCall() &&
                    !call->getAttributes().hasRetAttrs();
    for (unsigned i = 0; i < count && eligible; ++i)
      eligible = result->getElementType(i)->isFloatTy() &&
                 call->getArgOperand(i)->getType()->isFloatTy() &&
                 !call->getAttributes().hasParamAttrs(i);
    std::string expected;
    for (unsigned i = 0; i < count; ++i)
      expected += "=v,";
    for (unsigned i = 0; i < count; ++i)
      expected += std::to_string(i) + ",";
    expected += "~{memory}";
    eligible &= assembly->getConstraintString() == expected;
    llvm::SmallVector<llvm::ExtractValueInst *> extracts;
    for (auto *user : call->users()) {
      auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(user);
      if (!extract || extract->getNumIndices() != 1 ||
          *extract->idx_begin() >= count) {
        eligible = false;
        break;
      }
      extracts.push_back(extract);
    }
    if (!eligible) {
      ++stats.rejected;
      continue;
    }
    auto &context = module->getContext();
    auto *vector = llvm::FixedVectorType::get(llvm::Type::getFloatTy(context), 4);
    const unsigned groups = count / 4;
    llvm::SmallVector<llvm::Type *> types(groups, vector);
    // LLVM inline asm requires a direct result for a single output constraint.
    llvm::Type *returnType =
        groups == 1 ? static_cast<llvm::Type *>(vector)
                    : llvm::StructType::get(context, types);
    auto *functionType = llvm::FunctionType::get(returnType, types, false);
    std::string constraints;
    for (unsigned i = 0; i < groups; ++i)
      constraints += "=v,";
    for (unsigned i = 0; i < groups; ++i)
      constraints += std::to_string(i) + ",";
    constraints += "~{memory}";
    auto *tupleAsm = llvm::InlineAsm::get(
        functionType, assembly->getAsmString(), constraints,
        assembly->hasSideEffects(), assembly->isAlignStack(),
        assembly->getDialect(), assembly->canThrow());
    llvm::IRBuilder<> builder(call);
    llvm::SmallVector<llvm::Value *> arguments;
    for (unsigned group = 0; group < groups; ++group) {
      llvm::Value *value = llvm::PoisonValue::get(vector);
      for (unsigned lane = 0; lane < 4; ++lane)
        value = builder.CreateInsertElement(
            value, call->getArgOperand(4 * group + lane), builder.getInt32(lane));
      arguments.push_back(value);
    }
    auto *tupleCall = builder.CreateCall(functionType, tupleAsm, arguments);
    tupleCall->setCallingConv(call->getCallingConv());
    tupleCall->setAttributes(call->getAttributes());
    tupleCall->setTailCallKind(call->getTailCallKind());
    tupleCall->copyMetadata(*call);
    tupleCall->setDebugLoc(call->getDebugLoc());
    llvm::SmallVector<llvm::Value *> vectors;
    for (unsigned group = 0; group < groups; ++group)
      vectors.push_back(groups == 1
                            ? tupleCall
                            : builder.CreateExtractValue(tupleCall, {group}));
    for (auto *extract : extracts) {
      const unsigned index = *extract->idx_begin();
      llvm::IRBuilder<> use(extract);
      auto *value = use.CreateExtractElement(vectors[index / 4], index % 4);
      extract->replaceAllUsesWith(value);
      extract->eraseFromParent();
    }
    call->eraseFromParent();
    ++stats.sites;
    stats.scalarLanes += count;
  }
  if (llvm::verifyModule(*module))
    throw std::runtime_error("invalid module after tupled register fences");
  return stats;
}
} // namespace
