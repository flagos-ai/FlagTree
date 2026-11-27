#ifndef FLAGTREE_HCU_PACK_BF16_ROUNDS_H
#define FLAGTREE_HCU_PACK_BF16_ROUNDS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>
#include <utility>

namespace flagtree::hcu {
namespace detail {

struct Bf16Round {
  llvm::SelectInst *select;
  llvm::Value *rounded;
};

// Match the software RTNE lowering, including its choice of canonical NaN.
// The original integer sum and NaN test are reused without changing them.
inline std::optional<Bf16Round> matchBf16Round(llvm::Value *value) {
  using namespace llvm;
  using namespace llvm::PatternMatch;
  auto *select = dyn_cast<SelectInst>(value);
  if (!select || !select->hasOneUse())
    return std::nullopt;
  auto *nan = dyn_cast<ConstantFP>(select->getTrueValue());
  if (!nan || !nan->getType()->isBFloatTy() ||
      nan->getValueAPF().bitcastToAPInt() != 0x7fff)
    return std::nullopt;

  Value *shifted, *rounded;
  if (!match(select->getFalseValue(), m_BitCast(m_Trunc(m_Value(shifted)))) ||
      !match(shifted, m_LShr(m_Value(rounded), m_SpecificInt(16))) ||
      !rounded->getType()->isIntegerTy(32) ||
      cast<BinaryOperator>(shifted)->isExact())
    return std::nullopt;

  Value *biased, *parity, *bits, *parityShift, *input;
  if (!match(rounded, m_c_Add(m_Value(biased), m_c_And(m_Value(parityShift),
                                                       m_SpecificInt(1)))) ||
      !match(biased, m_c_Add(m_Value(bits), m_SpecificInt(0x7fff))) ||
      !match(parityShift, m_LShr(m_Value(parity), m_SpecificInt(16))) ||
      parity != bits || !match(bits, m_BitCast(m_Value(input))) ||
      !input->getType()->isFloatTy())
    return std::nullopt;

  auto *test = dyn_cast<FCmpInst>(select->getCondition());
  if (!test || test->getPredicate() != CmpInst::FCMP_UNO)
    return std::nullopt;
  Value *lhs = test->getOperand(0);
  Value *rhs = test->getOperand(1);
  if (rhs == input)
    std::swap(lhs, rhs);
  if (lhs != input)
    return std::nullopt;
  auto *zero = dyn_cast<ConstantFP>(rhs);
  if (!zero || !zero->isZero())
    return std::nullopt;
  return Bf16Round{select, rounded};
}
} // namespace detail

inline unsigned packBf16Rounds(llvm::Module &module) {
  using namespace llvm;
  using namespace llvm::PatternMatch;
  if (!module.getDataLayout().isLittleEndian())
    return 0;

  SmallVector<BitCastInst *> packs;
  for (Function &function : module)
    for (Instruction &instruction : instructions(function))
      if (auto *cast = dyn_cast<BitCastInst>(&instruction)) {
        auto *pair = dyn_cast<FixedVectorType>(cast->getSrcTy());
        if (cast->getDestTy()->isIntegerTy(32) && pair &&
            pair->getNumElements() == 2 && pair->getElementType()->isBFloatTy())
          packs.push_back(cast);
      }

  unsigned sites = 0;
  SmallVector<Instruction *> dead;
  for (BitCastInst *pack : packs) {
    auto *second = dyn_cast<InsertElementInst>(pack->getOperand(0));
    if (!second || !second->hasOneUse() ||
        !match(second->getOperand(2), m_SpecificInt(1)))
      continue;
    auto *first = dyn_cast<InsertElementInst>(second->getOperand(0));
    if (!first || !first->hasOneUse() ||
        !isa<UndefValue>(first->getOperand(0)) ||
        !match(first->getOperand(2), m_SpecificInt(0)))
      continue;
    auto lo = detail::matchBf16Round(first->getOperand(1));
    auto hi = detail::matchBf16Round(second->getOperand(1));
    if (!lo || !hi)
      continue;

    // Place each new select at the original conversion so that its predicate
    // and the unreduced FP32 operand do not stay live until the pack.
    auto selectHighWord = [](const detail::Bf16Round &round) {
      IRBuilder<> builder(round.select);
      return builder.CreateSelect(round.select->getCondition(),
                                  builder.getInt32(0x7fff0000), round.rounded);
    };
    Value *loWord = selectHighWord(*lo);
    Value *hiWord = selectHighWord(*hi);
    IRBuilder<> builder(pack);
    // The low result element takes bytes 2 and 3 of loWord. The high
    // result element takes bytes 2 and 3 of hiWord.
    Function *perm =
        Intrinsic::getOrInsertDeclaration(&module, Intrinsic::amdgcn_perm);
    Value *result = builder.CreateCall(
        perm, {hiWord, loWord, builder.getInt32(0x07060302)});
    pack->replaceAllUsesWith(result);
    dead.push_back(pack);
    ++sites;
  }
  for (Instruction *pack : dead)
    RecursivelyDeleteTriviallyDeadInstructions(pack);
  return sites;
}
} // namespace flagtree::hcu

#endif
