#include "TritonHCUGPUToLLVM/MmaPhaseScheduling.h"
#include "RegisterIdentityFence.h"
#include <cstdint>
#include <functional>
#include <optional>
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

// This scheduling preparation only moves register computations. A caller must
// additionally validate the final target's explicit VM/LGKM operand readiness:
// SSA dominance alone does not prove completion of an asynchronous LDS load.
// Unsupported IR is unchanged. The native validation is part of enabling this
// preparation, not an optional debug check.
namespace flagtree::hcu {

namespace mma_phase_detail {
inline bool hasPrefix(llvm::StringRef text, llvm::StringRef prefix) {
  return text.take_front(prefix.size()) == prefix;
}
inline bool isMma(const llvm::Instruction *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  auto *fn = call ? call->getCalledFunction() : nullptr;
  return fn && hasPrefix(fn->getName(), "llvm.hcu.mmac.") &&
         !call->hasOperandBundles() && !call->getType()->isVoidTy() &&
         !call->mayReadOrWriteMemory() && call->doesNotThrow() &&
         call->hasFnAttr(llvm::Attribute::WillReturn);
}
inline bool isBarrier(const llvm::Instruction *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  auto *assembly = call ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand())
                        : nullptr;
  return assembly && llvm::StringRef(assembly->getAsmString()).trim() ==
                         "s_barrier" &&
         !call->hasOperandBundles() && call->getType()->isVoidTy() &&
         assembly->hasSideEffects();
}
inline std::optional<uint32_t> waitEncoding(const llvm::Instruction *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  auto *fn = call ? call->getCalledFunction() : nullptr;
  if (!fn || fn->getName() != "llvm.amdgcn.s.waitcnt" ||
      call->arg_size() != 1 || call->hasOperandBundles())
    return std::nullopt;
  auto *constant = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
  if (!constant || constant->getBitWidth() != 32)
    return std::nullopt;
  return uint32_t(constant->getZExtValue());
}
// gfx936 LLVM wait encoding: six VM bits and six LGKM bits. Require an
// unconstrained opposite queue and expcnt; never weaken either original wait.
inline bool isVmWait(const llvm::Instruction *inst) {
  auto value = waitEncoding(inst);
  return value && (*value & 0x3f00) == 0x3f00 &&
         (*value & 0x70) == 0x70 && (*value & 0xc00f) != 0xc00f;
}
inline bool isLgkmWait(const llvm::Instruction *inst) {
  auto value = waitEncoding(inst);
  return value && (*value & 0xc00f) == 0xc00f &&
         (*value & 0x70) == 0x70 && (*value & 0x3f00) != 0x3f00;
}

// A CTA barrier requires equal trip counts across all waves. Wave-uniform
// readfirstlane(workitem-id) is deliberately insufficient for this proof.
inline bool isCtaUniform(llvm::Value *value,
                         llvm::SmallPtrSetImpl<llvm::Value *> &active,
                         llvm::SmallPtrSetImpl<llvm::Value *> &proven,
                         unsigned &budget) {
  if (!budget--)
    return false;
  if (llvm::isa<llvm::ConstantInt, llvm::ConstantFP, llvm::ConstantPointerNull,
                llvm::GlobalValue, llvm::Argument>(value) || proven.contains(value))
    return true;
  if (active.contains(value))
    return true; // Inductive PHI: every external seed is still checked below.
  auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
  if (!inst || inst->mayReadOrWriteMemory())
    return false;
  active.insert(value);
  bool uniform = true;
  if (auto *call = llvm::dyn_cast<llvm::CallInst>(inst)) {
    auto *fn = call->getCalledFunction();
    uniform = fn && (fn->getName() == "llvm.amdgcn.workgroup.id.x" ||
                     fn->getName() == "llvm.amdgcn.workgroup.id.y" ||
                     fn->getName() == "llvm.amdgcn.workgroup.id.z");
    // Integer min/max preserve CTA uniformity. They commonly bound a peeled
    // induction trip count; arbitrary readnone calls remain unsupported.
    if (fn && (hasPrefix(fn->getName(), "llvm.smax.") ||
               hasPrefix(fn->getName(), "llvm.smin.") ||
               hasPrefix(fn->getName(), "llvm.umax.") ||
               hasPrefix(fn->getName(), "llvm.umin."))) {
      uniform = true;
      for (llvm::Value *operand : call->args())
        if (!isCtaUniform(operand, active, proven, budget)) {
          uniform = false;
          break;
        }
    }
  } else if (!llvm::isa<llvm::BinaryOperator, llvm::CastInst, llvm::CmpInst,
                        llvm::SelectInst, llvm::PHINode>(inst)) {
    uniform = false;
  } else {
    for (llvm::Value *operand : inst->operands())
      if (!isCtaUniform(operand, active, proven, budget)) {
        uniform = false;
        break;
      }
  }
  active.erase(value);
  if (uniform)
    proven.insert(value);
  return uniform;
}
struct Cluster {
  llvm::Instruction *vm, *lgkm;
  llvm::SmallVector<llvm::Instruction *> move;
};
inline bool version(llvm::BasicBlock &block, unsigned numWarps,
                    MmaPhaseStats &stats) {
  auto *term = llvm::dyn_cast<llvm::BranchInst>(block.getTerminator());
  if (!term || !term->isConditional() ||
      term->getMetadata("hcu.mma.phase.loop"))
    return false;
  llvm::BasicBlock *exit = nullptr;
  if (term->getSuccessor(0) == &block && term->getSuccessor(1) != &block)
    exit = term->getSuccessor(1);
  else if (term->getSuccessor(1) == &block && term->getSuccessor(0) != &block)
    exit = term->getSuccessor(0);
  if (!exit)
    return false;
  ++stats.considered;
  llvm::BasicBlock *entry = nullptr;
  unsigned predecessors = 0;
  for (llvm::BasicBlock *pred : llvm::predecessors(&block)) {
    if (pred == &block)
      continue;
    if (entry && pred != entry)
      return false;
    entry = pred;
    ++predecessors;
  }
  auto *entryBranch = entry ? llvm::dyn_cast<llvm::BranchInst>(entry->getTerminator())
                            : nullptr;
  if (predecessors != 1 || !entryBranch || !entryBranch->isUnconditional() ||
      entryBranch->getSuccessor(0) != &block)
    return false;
  llvm::SmallPtrSet<llvm::Value *, 32> active, proven;
  unsigned budget = 4096;
  if (!isCtaUniform(term->getCondition(), active, proven, budget))
    return false;

  llvm::SmallVector<llvm::Instruction *> instructions, waits;
  unsigned totalMmas = 0, totalBarriers = 0;
  for (auto &inst : block) {
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
      if (auto *callee = call->getCalledFunction())
        if (callee->getName() == "llvm.experimental.noalias.scope.decl")
          return false;
    instructions.push_back(&inst);
    if (isVmWait(&inst))
      waits.push_back(&inst);
    totalMmas += isMma(&inst);
    totalBarriers += isBarrier(&inst);
  }
  if (waits.empty() || totalMmas == 0 || totalBarriers != waits.size())
    return false;
  llvm::SmallVector<Cluster> clusters;
  llvm::SmallPtrSet<llvm::Instruction *, 32> claimed;
  for (unsigned group = 0; group < waits.size(); ++group) {
    llvm::Instruction *vm = waits[group];
    llvm::Instruction *lgkm = vm->getNextNode();
    while (lgkm && llvm::isa<llvm::DbgInfoIntrinsic>(lgkm))
      lgkm = lgkm->getNextNode();
    if (!lgkm || !isLgkmWait(lgkm))
      lgkm = nullptr;
    auto begin = llvm::find(instructions, lgkm ? lgkm : vm);
    auto end = group + 1 < waits.size()
                   ? llvm::find(instructions, waits[group + 1])
                   : instructions.end();
    llvm::Instruction *barrier = nullptr;
    llvm::SmallVector<llvm::Instruction *> mmas;
    for (auto at = begin + 1; at != end; ++at) {
      if (isBarrier(*at)) {
        if (barrier)
          return false;
        barrier = *at;
      }
      if (isMma(*at)) {
        if (!barrier)
          return false;
        mmas.push_back(*at);
      }
    }
    if (!barrier || mmas.empty())
      return false;
    llvm::SmallPtrSet<llvm::Instruction *, 32> allowed(mmas.begin(), mmas.end());
    llvm::SmallPtrSet<llvm::Instruction *, 32> scheduled, visiting;
    Cluster cluster{vm, lgkm, {}};
    std::function<bool(llvm::Instruction *)> append = [&](llvm::Instruction *inst) {
      if (inst->getParent() != &block || inst->comesBefore(vm))
        return true;
      if (scheduled.contains(inst))
        return true;
      if (!visiting.insert(inst).second || llvm::isa<llvm::PHINode>(inst))
        return false;
      if (isMma(inst)) {
        if (!allowed.contains(inst))
          return false;
      } else if (!isRegisterIdentityFence(inst) &&
                 (inst->mayReadOrWriteMemory() ||
                  !llvm::isSafeToSpeculativelyExecute(inst))) {
        return false;
      }
      for (llvm::Value *operand : inst->operands())
        if (auto *dependency = llvm::dyn_cast<llvm::Instruction>(operand))
          if (!append(dependency))
            return false;
      visiting.erase(inst);
      scheduled.insert(inst);
      cluster.move.push_back(inst);
      return true;
    };
    // Visit roots in source order, including closing tied fences with no later
    // MMA user. Every priority reset stays with the complete register group.
    for (auto at = begin + 1; at != end; ++at)
      if (isMma(*at)) {
        if (!append(*at) || !claimed.insert(*at).second)
          return false;
      } else if (isRegisterIdentityFence(*at) && !append(*at)) {
        return false;
      }
    clusters.push_back(std::move(cluster));
  }
  if (claimed.size() != totalMmas)
    return false;

  llvm::Function *fn = block.getParent();
  llvm::ValueToValueMapTy map;
  llvm::BasicBlock *high = llvm::CloneBasicBlock(&block, map, ".mma_high", fn);
  map[&block] = high;
  for (auto &inst : *high)
    llvm::RemapInstruction(&inst, map, llvm::RF_IgnoreMissingLocals |
                                           llvm::RF_NoModuleLevelChanges);
  for (const Cluster &cluster : clusters) {
    auto *vm = llvm::cast<llvm::Instruction>(map[cluster.vm]);
    if (cluster.lgkm) {
      auto *lgkm = llvm::cast<llvm::Instruction>(map[cluster.lgkm]);
      lgkm->moveBefore(*high, vm->getIterator());
    }
    for (auto *inst : cluster.move)
      llvm::cast<llvm::Instruction>(map[inst])->moveBefore(*high, vm->getIterator());
    // Preserve this register-only group through target instruction scheduling.
    // This is a scheduling pseudo and emits no hardware instruction.
    llvm::IRBuilder<> boundary(vm);
    auto fence = fn->getParent()->getOrInsertFunction(
        "llvm.amdgcn.sched.barrier",
        llvm::FunctionType::get(boundary.getVoidTy(), {boundary.getInt32Ty()}, false));
    boundary.CreateCall(fence, {boundary.getInt32(0)});
  }
  llvm::LLVMContext &ctx = fn->getContext();
  auto *selector = llvm::BasicBlock::Create(ctx, "mma_wave_group", fn, &block);
  llvm::IRBuilder<> builder(selector);
  auto i32 = builder.getInt32Ty();
  auto workitem = fn->getParent()->getOrInsertFunction(
      "llvm.amdgcn.workitem.id.x", llvm::FunctionType::get(i32, false));
  auto firstlane = fn->getParent()->getOrInsertFunction(
      "llvm.amdgcn.readfirstlane.i32", llvm::FunctionType::get(i32, {i32}, false));
  llvm::Value *tid = builder.CreateCall(workitem, {});
  llvm::Value *waveFirst = builder.CreateCall(firstlane, {tid});
  llvm::Value *low = builder.CreateICmpULT(
      waveFirst, builder.getInt32(64 * (numWarps / 2)));
  builder.CreateCondBr(low, &block, high);
  entryBranch->setSuccessor(0, selector);
  for (auto &phi : block.phis())
    phi.replaceIncomingBlockWith(entry, selector);
  for (auto &phi : high->phis())
    phi.replaceIncomingBlockWith(entry, selector);
  // Live-outs need SSA merges even when the original single-predecessor exit
  // used them directly rather than through an LCSSA PHI.
  for (auto &inst : block) {
    if (inst.getType()->isVoidTy())
      continue;
    llvm::SmallVector<llvm::Use *> externalUses;
    for (auto &use : inst.uses()) {
      auto *user = llvm::dyn_cast<llvm::Instruction>(use.getUser());
      if (user && user->getParent() != &block && user->getParent() != high)
        externalUses.push_back(&use);
    }
    if (externalUses.empty())
      continue;
    llvm::SSAUpdater updater;
    updater.Initialize(inst.getType(), "mma_phase_merge");
    updater.AddAvailableValue(&block, &inst);
    updater.AddAvailableValue(high, llvm::cast<llvm::Value>(map[&inst]));
    for (auto *use : externalUses)
      updater.RewriteUse(*use);
  }
  // Existing PHIs in the direct exit have a new predecessor as well. Their
  // original edge is already valid; add the corresponding cloned live-out.
  for (auto &phi : exit->phis()) {
    int index = phi.getBasicBlockIndex(&block);
    if (index >= 0 && phi.getBasicBlockIndex(high) < 0)
      phi.addIncoming(llvm::MapValue(phi.getIncomingValue(index), map,
                                    llvm::RF_IgnoreMissingLocals), high);
  }
  // Loop IDs are identities, not just scheduling hints. Each cloned natural
  // loop must own its self-referential ID while retaining the original hints.
  if (auto *oldID = term->getMetadata("llvm.loop")) {
    llvm::SmallVector<llvm::Metadata *> operands;
    for (const auto &operand : oldID->operands())
      operands.push_back(operand.get());
    if (!operands.empty()) {
      auto *highID = llvm::MDNode::getDistinct(ctx, operands);
      highID->replaceOperandWith(0, highID);
      high->getTerminator()->setMetadata("llvm.loop", highID);
    }
  }
  auto *mark = llvm::MDNode::get(ctx, {});
  term->setMetadata("hcu.mma.phase.loop", mark);
  high->getTerminator()->setMetadata("hcu.mma.phase.loop", mark);
  ++stats.versioned;
  stats.mmaCalls += totalMmas;
  return true;
}
} // namespace mma_phase_detail

MmaPhaseStats prepareMmaPhaseLoops(llvm::Module &module,
                                          llvm::StringRef arch,
                                          unsigned numWarps) {
  MmaPhaseStats stats;
  if (arch != "gfx936" || numWarps < 2 || numWarps > 32)
    return stats;
  for (auto &fn : module) {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    llvm::SmallVector<llvm::BasicBlock *> original;
    for (auto &block : fn)
      original.push_back(&block);
    for (auto *block : original)
      mma_phase_detail::version(*block, numWarps, stats);
  }
  return stats;
}
} // namespace flagtree::hcu
