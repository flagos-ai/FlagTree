#ifndef TRITON_HCU_FACTOR_BUFFER_STORE_OFFSETS_H
#define TRITON_HCU_FACTOR_BUFFER_STORE_OFFSETS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

namespace flagtree::hcu {
namespace buffer_store_offsets {

using Terms = llvm::SmallVector<std::pair<llvm::Value *, uint32_t>, 4>;

struct Address {
  Terms terms;
  uint32_t constant = 0;
};

struct Store {
  llvm::CallInst *call;
  llvm::Value *mask;
  llvm::ConstantInt *sentinel;
  uint32_t constant;
};

struct Group {
  llvm::Value *descriptor;
  Terms terms;
  llvm::SmallVector<Store, 8> stores;
};

class DecomposeAddress {
  const llvm::DataLayout &layout;
  llvm::Instruction *context;
  llvm::DenseMap<llvm::Value *, unsigned> &ordinals;
  unsigned remaining = 256;

  bool disjoint(llvm::BinaryOperator *op) const {
    auto a = llvm::computeKnownBits(op->getOperand(0), layout, nullptr, context);
    auto b = llvm::computeKnownBits(op->getOperand(1), layout, nullptr, context);
    return (~a.Zero & ~b.Zero).isZero();
  }

  bool collect(llvm::Value *value, uint32_t scale, Address &out,
               unsigned depth) {
    if (!remaining || depth == 24 || !scale)
      return false;
    --remaining;
    if (auto *constant = llvm::dyn_cast<llvm::ConstantInt>(value)) {
      out.constant += scale * static_cast<uint32_t>(constant->getZExtValue());
      return true;
    }

    if (auto *op = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
      switch (op->getOpcode()) {
      case llvm::Instruction::Or:
      case llvm::Instruction::Xor:
        if (!disjoint(op))
          break;
        [[fallthrough]];
      case llvm::Instruction::Add:
        return collect(op->getOperand(0), scale, out, depth + 1) &&
               collect(op->getOperand(1), scale, out, depth + 1);
      case llvm::Instruction::Shl:
        if (auto *shift = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(1))) {
          uint64_t amount = shift->getZExtValue();
          if (amount < 32)
            return collect(op->getOperand(0), scale << amount, out, depth + 1);
        }
        break;
      default:
        break;
      }
    }
    // Multiplication and all other operations are kept as existing SSA
    // values. KnownBits can still prove that their low bits are zero.
    if (!ordinals.count(value))
      return false;
    for (auto &entry : out.terms) {
      if (entry.first == value) {
        entry.second += scale;
        return entry.second != 0;
      }
    }
    if (out.terms.size() == 8)
      return false;
    out.terms.emplace_back(value, scale);
    return true;
  }

public:
  DecomposeAddress(const llvm::DataLayout &layout, llvm::Instruction *context,
                   llvm::DenseMap<llvm::Value *, unsigned> &ordinals)
      : layout(layout), context(context), ordinals(ordinals) {}

  bool run(llvm::Value *value, Address &out) {
    if (!collect(value, 1, out, 0) || out.terms.empty())
      return false;
    llvm::sort(out.terms, [&](const auto &a, const auto &b) {
      return ordinals.lookup(a.first) < ordinals.lookup(b.first);
    });
    return true;
  }
};

inline bool isCandidate(llvm::CallInst *call) {
  auto *callee = call ? call->getCalledFunction() : nullptr;
  return callee && callee->isIntrinsic() && call->arg_size() == 5 &&
         !call->getNumOperandBundles() &&
         callee->getName() == "llvm.amdgcn.raw.ptr.buffer.store.i32" &&
         call->getArgOperand(2)->getType()->isIntegerTy(32) &&
         llvm::PatternMatch::match(call->getArgOperand(3), llvm::PatternMatch::m_Zero()) &&
         llvm::PatternMatch::match(call->getArgOperand(4), llvm::PatternMatch::m_Zero());
}

} // namespace buffer_store_offsets

// Keep descriptor, payload, SOFFSET, cache fields and store order. The new
// VOFFSET expression is the same i32 bit pattern for all defined inputs.
inline unsigned factorBufferStoreOffsets(llvm::Module &module,
                                         llvm::StringRef arch) {
  using namespace buffer_store_offsets;
  if (arch != "gfx936")
    return 0;
  unsigned changed = 0;
  for (auto &function : module) {
    if (function.isDeclaration() ||
        function.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    llvm::DenseMap<llvm::Value *, unsigned> ordinals;
    unsigned ordinal = 0;
    for (auto &arg : function.args())
      ordinals.try_emplace(&arg, ordinal++);
    for (auto &block : function)
      for (auto &inst : block)
        ordinals.try_emplace(&inst, ordinal++);

    llvm::DominatorTree dom(function);
    for (auto &block : function) {
      llvm::SmallVector<Group, 8> groups;
      unsigned candidates = 0;
      for (auto &inst : block) {
        auto *store = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!isCandidate(store))
          continue;
        if (++candidates > 256)
          break;
        llvm::Value *offset = store->getArgOperand(2);
        llvm::Value *mask = nullptr;
        llvm::ConstantInt *sentinel = nullptr;
        if (auto *select = llvm::dyn_cast<llvm::SelectInst>(offset)) {
          sentinel = llvm::dyn_cast<llvm::ConstantInt>(select->getFalseValue());
          if (!sentinel)
            continue;
          mask = select->getCondition();
          offset = select->getTrueValue();
        }
        Address address;
        if (!DecomposeAddress(module.getDataLayout(), store, ordinals)
                 .run(offset, address))
          continue;
        auto *descriptor = store->getArgOperand(1);
        auto found = llvm::find_if(groups, [&](const Group &group) {
          return group.descriptor == descriptor && group.terms == address.terms;
        });
        if (found == groups.end()) {
          groups.push_back({descriptor, std::move(address.terms), {}});
          found = std::prev(groups.end());
        }
        found->stores.push_back({store, mask, sentinel, address.constant});
      }
      if (candidates > 256)
        continue;

      llvm::SmallVector<llvm::WeakTrackingVH, 16> dead;
      for (const auto &group : groups) {
        if (group.stores.size() < 2 ||
            std::none_of(group.stores.begin(), group.stores.end(),
                         [](const Store &store) { return store.constant != 0; }))
          continue;
        auto *first = group.stores.front().call;
        if (llvm::any_of(group.terms, [&](const auto &term) {
              return !dom.dominates(term.first, first);
            }))
          continue;

        llvm::IRBuilder<> builder(first);
        builder.SetCurrentDebugLocation(first->getDebugLoc());
        llvm::Value *common = nullptr;
        for (const auto &[value, scale] : group.terms) {
          llvm::Value *term = value;
          if (scale != 1)
            term = builder.CreateMul(term, builder.getInt32(scale));
          common = common ? builder.CreateAdd(common, term) : term;
        }
        for (const auto &store : group.stores) {
          builder.SetInsertPoint(store.call);
          builder.SetCurrentDebugLocation(store.call->getDebugLoc());
          llvm::Value *offset = common;
          if (store.constant)
            offset = builder.CreateAdd(offset, builder.getInt32(store.constant));
          if (store.mask)
            offset = builder.CreateSelect(store.mask, offset, store.sentinel);
          dead.emplace_back(store.call->getArgOperand(2));
          store.call->setArgOperand(2, offset);
          ++changed;
        }
      }
      // Delete expressions only after every staged replacement has run.
      for (llvm::Value *value : dead)
        if (auto *inst = llvm::dyn_cast_or_null<llvm::Instruction>(value))
          llvm::RecursivelyDeleteTriviallyDeadInstructions(inst);
    }
  }
  return changed;
}

} // namespace flagtree::hcu
#endif
