#ifndef TRITON_HCU_SHIFTED_DISJOINT_OR_H
#define TRITON_HCU_SHIFTED_DISJOINT_OR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/KnownBits.h"
#include <string>

namespace flagtree::hcu {

// This is a sufficient, deliberately incomplete gfx936 wave-uniform proof.
// False means NOT PROVED UNIFORM, never "proved divergent". Unknowns keep the
// old v2 fold eligibility; this does not eliminate every possible waterfall.
class ProvedWaveUniformInteger {
  const llvm::Instruction *context;
  llvm::DenseMap<const llvm::Value *, bool> memo;
  llvm::SmallPtrSet<const llvm::Value *, 32> active;

  bool definedIntegerOperation(const llvm::Instruction *inst) const {
    // All operands are recursively proved defined/uniform before this call.
    // Still retain poison-generating flags: do not drop nsw/nuw/exact/nneg or
    // same-sign promises and do not treat freeze as evidence of uniformity.
    return !llvm::canCreateUndefOrPoison(llvm::cast<llvm::Operator>(inst), true) ||
           llvm::isGuaranteedNotToBeUndefOrPoison(
               inst, /*AC=*/nullptr, /*CtxI=*/context, /*DT=*/nullptr,
               /*Depth=*/0);
  }

  bool proveImpl(const llvm::Value *value, unsigned depth) {
    if (!value->getType()->isIntegerTy() ||
        llvm::isa<llvm::UndefValue, llvm::PoisonValue>(value))
      return false;
    if (llvm::isa<llvm::ConstantInt>(value))
      return true;
    if (depth >= 32)
      return false;
    if (auto found = memo.find(value); found != memo.end())
      return found->second;
    if (!active.insert(value).second)
      return false;

    bool uniform = false;
    if (auto *argument = llvm::dyn_cast<llvm::Argument>(value)) {
      // InReg alone is not a generic language-level uniformity promise. Use
      // its AMDGPU kernel ABI meaning only, and require a defined argument.
      uniform = argument->getParent()->getCallingConv() ==
                    llvm::CallingConv::AMDGPU_KERNEL &&
                argument->hasAttribute(llvm::Attribute::InReg) &&
                argument->hasAttribute(llvm::Attribute::NoUndef);
    } else if (auto *call = llvm::dyn_cast<llvm::CallInst>(value)) {
      auto *callee = call->getCalledFunction();
      if (callee && callee->isIntrinsic() &&
          !call->getNumOperandBundles() && call->getType()->isIntegerTy(32)) {
        auto name = callee->getName();
        if (call->arg_empty() &&
            (name == "llvm.amdgcn.workgroup.id.x" ||
             name == "llvm.amdgcn.workgroup.id.y" ||
             name == "llvm.amdgcn.workgroup.id.z")) {
          // A workgroup ID is defined and common to the active wave on gfx936.
          uniform = true;
        } else if ((name == "llvm.amdgcn.readfirstlane" ||
                    name == "llvm.amdgcn.readfirstlane.i32") &&
                   call->arg_size() == 1 &&
                   call->getArgOperand(0)->getType()->isIntegerTy(32)) {
          // readfirstlane may take a varying input. A poison/undef input is
          // not accepted as a proof; definedness is a separate LLVM22 query.
          uniform = llvm::isGuaranteedNotToBeUndefOrPoison(
              call->getArgOperand(0), /*AC=*/nullptr, /*CtxI=*/context,
              /*DT=*/nullptr, /*Depth=*/0);
        }
      }
    } else if (auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
      bool allowed = false;
      switch (binary->getOpcode()) {
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::And:
      case llvm::Instruction::Or:
      case llvm::Instruction::Xor:
        allowed = true;
        break;
      case llvm::Instruction::Shl:
      case llvm::Instruction::LShr:
      case llvm::Instruction::AShr:
        if (auto *amount =
                llvm::dyn_cast<llvm::ConstantInt>(binary->getOperand(1)))
          allowed = amount->getValue().ult(
              binary->getType()->getIntegerBitWidth());
        break;
      default:
        // In particular, div/rem are not admitted by a poison-only query:
        // division by zero can raise immediate UB without producing poison.
        break;
      }
      uniform = allowed && proveImpl(binary->getOperand(0), depth + 1) &&
                proveImpl(binary->getOperand(1), depth + 1) &&
                definedIntegerOperation(binary);
    } else if (auto *cast = llvm::dyn_cast<llvm::CastInst>(value)) {
      bool allowed = cast->getOperand(0)->getType()->isIntegerTy() &&
                     (cast->getOpcode() == llvm::Instruction::Trunc ||
                      cast->getOpcode() == llvm::Instruction::ZExt ||
                      cast->getOpcode() == llvm::Instruction::SExt);
      // No ptr/int or address-space casts: private addressing may have a
      // per-lane base even if an integer-looking offset is common.
      uniform = allowed && proveImpl(cast->getOperand(0), depth + 1) &&
                definedIntegerOperation(cast);
    } else if (auto *cmp = llvm::dyn_cast<llvm::ICmpInst>(value)) {
      uniform = cmp->getOperand(0)->getType()->isIntegerTy() &&
                proveImpl(cmp->getOperand(0), depth + 1) &&
                proveImpl(cmp->getOperand(1), depth + 1) &&
                definedIntegerOperation(cmp);
    } else if (auto *select = llvm::dyn_cast<llvm::SelectInst>(value)) {
      uniform = select->getCondition()->getType()->isIntegerTy(1) &&
                proveImpl(select->getCondition(), depth + 1) &&
                proveImpl(select->getTrueValue(), depth + 1) &&
                proveImpl(select->getFalseValue(), depth + 1) &&
                definedIntegerOperation(select);
    }
    // PHIs (including cycles), freeze, loads, arbitrary calls/inline asm,
    // vectors and all other operations deliberately remain unproved.
    active.erase(value);
    memo[value] = uniform;
    return uniform;
  }

public:
  explicit ProvedWaveUniformInteger(const llvm::Instruction *context)
      : context(context) {}
  bool prove(const llvm::Value *value) { return proveImpl(value, 0); }
};

struct ShiftedDisjointOrStats {
  unsigned combined = 0;
  unsigned skippedProvedUniform = 0;
  std::string reason = "no eligible one-use i32 shifted disjoint OR";
};

// Late instruction selection only: (A | B) << K == (A + B) << K modulo
// i32 when A and B cannot have a set bit in common. Never infer disjointness
// from a tensor shape, a launch bound, a name, or an unchecked OR spelling.
inline ShiftedDisjointOrStats
combineShiftedDisjointOr(llvm::Module &module, llvm::StringRef arch) {
  ShiftedDisjointOrStats stats;
  if (arch != "gfx936") {
    stats.reason = "unsupported architecture (requires gfx936)";
    return stats;
  }
  const llvm::DataLayout &layout = module.getDataLayout();
  llvm::SmallVector<llvm::BinaryOperator *, 32> shifts;
  for (auto &fn : module) {
    if (fn.isDeclaration())
      continue;
    for (auto &block : fn)
      for (auto &inst : block)
        if (auto *shift = llvm::dyn_cast<llvm::BinaryOperator>(&inst))
          if (shift->getOpcode() == llvm::Instruction::Shl &&
              shift->getType()->isIntegerTy(32) && shift->hasOneUse())
            shifts.push_back(shift);
  }
  for (auto *shift : shifts) {
    auto *amount = llvm::dyn_cast<llvm::ConstantInt>(shift->getOperand(1));
    auto *bits = llvm::dyn_cast<llvm::BinaryOperator>(shift->getOperand(0));
    if (!amount || amount->getZExtValue() < 1 ||
        amount->getZExtValue() > 31 || !bits ||
        bits->getOpcode() != llvm::Instruction::Or || !bits->hasOneUse() ||
        !bits->getType()->isIntegerTy(32))
      continue;

    // The actual use instruction and this module's DataLayout are passed to
    // ValueTracking. No AssumptionCache/DT is supplied: proofs requiring those
    // are conservatively missed rather than invented. A known-zero bit cannot
    // be set, so (~ZeroA & ~ZeroB)==0 proves every possible-one pair disjoint.
    const llvm::KnownBits a = llvm::computeKnownBits(
        bits->getOperand(0), layout, /*AC=*/nullptr, /*CxtI=*/shift,
        /*DT=*/nullptr, /*UseInstrInfo=*/true, /*Depth=*/0);
    const llvm::KnownBits b = llvm::computeKnownBits(
        bits->getOperand(1), layout, /*AC=*/nullptr, /*CxtI=*/shift,
        /*DT=*/nullptr, /*UseInstrInfo=*/true, /*Depth=*/0);
    if (!(~a.Zero & ~b.Zero).isZero())
      continue;

    // Keep proved scalar work in LLVM so normal SGPR selection is preserved.
    // A forced =v output would hide uniformity and can create descriptor EXEC
    // waterfalls. Never insert readfirstlane or guess unknown inputs uniform.
    ProvedWaveUniformInteger uniform(shift);
    if (uniform.prove(bits->getOperand(0)) &&
        uniform.prove(bits->getOperand(1))) {
      ++stats.skippedProvedUniform;
      continue;
    }

    auto *i32 = llvm::Type::getInt32Ty(module.getContext());
    auto *functionType = llvm::FunctionType::get(i32, {i32, i32}, false);
    const std::string code = "v_add_lshl_u32 $0, $1, $2, " +
                             std::to_string(amount->getZExtValue());
    auto *assembly = llvm::InlineAsm::get(
        functionType, code, "=v,v,v", /*hasSideEffects=*/false,
        /*isAlignStack=*/false, llvm::InlineAsm::AD_ATT, /*canThrow=*/false);
    llvm::IRBuilder<> builder(shift);
    builder.SetCurrentDebugLocation(shift->getDebugLoc());
    auto *combined = builder.CreateCall(
        functionType, assembly, {bits->getOperand(0), bits->getOperand(1)});
    combined->setDoesNotAccessMemory();
    combined->setDoesNotThrow();
    combined->addFnAttr(llvm::Attribute::WillReturn);
    combined->takeName(shift);
    shift->replaceAllUsesWith(combined);
    shift->eraseFromParent();
    // Only this now-dead OR is removed; operands, fences, and all memory and
    // control-flow operations keep their original dependency ordering.
    if (bits->use_empty())
      bits->eraseFromParent();
    ++stats.combined;
  }
  if (stats.combined)
    stats.reason = "combined proved disjoint one-use i32 OR+shift";
  if (stats.skippedProvedUniform)
    stats.reason += "; skipped proved wave-uniform OR: " +
                    std::to_string(stats.skippedProvedUniform);
  return stats;
}

} // namespace flagtree::hcu
#endif
