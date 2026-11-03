// LLVM-only, post-optimization read/read interleaving. No DMA or wait rewrite.
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"

namespace {
struct ReadOnlyInterleaveStats {
  unsigned windows = 0, globalReads = 0, ldsReads = 0, fences = 0;
  unsigned pureDependents = 0;
  unsigned rejectedDependencies = 0, invalidFunctions = 0, iglpFunctions = 0;
};

static bool interleaveOrdinaryRead(llvm::Instruction *i, unsigned as) {
  auto *load = llvm::dyn_cast<llvm::LoadInst>(i);
  return load && load->getPointerAddressSpace() == as &&
         !load->isVolatile() && !load->isAtomic();
}

static bool interleaveGlobalRead(llvm::Instruction *i) {
  if (i->getMetadata("hcu.readonly.interleave.done"))
    return false; // Importing a marker can only disable work, never authorize it.
  if (interleaveOrdinaryRead(i, 1))
    return true;
  auto *call = llvm::dyn_cast<llvm::CallInst>(i);
  auto *fn = call ? call->getCalledFunction() : nullptr;
  if (!fn || !fn->isDeclaration() || call->getNumOperandBundles() || call->isMustTailCall() ||
      !call->doesNotThrow() || call->mayWriteToMemory())
    return false;
  auto id = fn->getIntrinsicID();
  bool structured = id == llvm::Intrinsic::amdgcn_struct_ptr_buffer_load;
  bool raw = id == llvm::Intrinsic::amdgcn_raw_ptr_buffer_load;
  if ((!structured && !raw) || call->arg_size() != (structured ? 5u : 4u))
    return false;
  auto *resource = llvm::dyn_cast<llvm::PointerType>(call->getArgOperand(0)->getType());
  if (!resource || resource->getAddressSpace() != 8 || call->getType()->isVoidTy())
    return false;
  for (unsigned n = 1; n < call->arg_size(); ++n)
    if (!call->getArgOperand(n)->getType()->isIntegerTy(32))
      return false;
  // Only the known ordinary, non-atomic/non-volatile cache auxiliary domain.
  auto *aux = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(call->arg_size()-1));
  return aux && aux->isZero();
}

static bool interleavePure(llvm::Instruction *i) {
  // Explicit register-only instructions. No arbitrary pure call, query,
  // inline asm, fence, PHI, load, store, allocation or terminator is crossed.
  return llvm::isa<llvm::BinaryOperator, llvm::CastInst, llvm::CmpInst,
                   llvm::SelectInst, llvm::GetElementPtrInst,
                   llvm::ExtractElementInst, llvm::InsertElementInst,
                   llvm::ShuffleVectorInst, llvm::ExtractValueInst,
                   llvm::InsertValueInst, llvm::FreezeInst>(i) &&
         !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects();
}

struct ReadOnlyInterleavePlan {
  llvm::SmallVector<llvm::Instruction *> global, lds, moving;
  llvm::Instruction *middle = nullptr;
};

static bool interleaveUseDominated(llvm::Instruction *position,
                                  llvm::Use &use, llvm::DominatorTree &dt) {
  auto *user = llvm::dyn_cast<llvm::Instruction>(use.getUser());
  if (!user)
    return false;
  if (auto *phi = llvm::dyn_cast<llvm::PHINode>(user)) {
    // A PHI consumes on its incoming edge, not at its lexical instruction.
    // In particular, a backedge use remains valid before that terminator.
    unsigned n = use.getOperandNo();
    return n < phi->getNumIncomingValues() &&
           dt.dominates(position, phi->getIncomingBlock(n)->getTerminator());
  }
  return user == position || dt.dominates(position, user);
}

static ReadOnlyInterleaveStats
interleaveGlobalLdsReads(llvm::Module *module, const std::string &arch,
                         bool addFences) {
  ReadOnlyInterleaveStats stats;
  if (arch != "gfx936") return stats;
  if (llvm::verifyModule(*module, &llvm::errs())) {
    stats.invalidFunctions = 1; return stats;
  }
  for (auto &fn : *module) {
    if (fn.isDeclaration() ||
        fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    bool iglp = false;
    for (auto &bb : fn)
      for (auto &inst : bb)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *f = call->getCalledFunction())
            iglp |= f->getIntrinsicID() == llvm::Intrinsic::amdgcn_iglp_opt;
    if (iglp) { ++stats.iglpFunctions; continue; }
    llvm::DominatorTree dt(fn);
    llvm::SmallVector<ReadOnlyInterleavePlan> plans;
    llvm::DenseSet<llvm::Instruction *> seenLds, claimedGlobal;
    for (auto &bb : fn) {
      for (auto &inst : bb) {
        if (!interleaveOrdinaryRead(&inst, 3) || seenLds.contains(&inst))
          continue;
        ReadOnlyInterleavePlan plan;
        for (auto *i = &inst; i; i = i->getNextNode()) {
          if (interleaveOrdinaryRead(i, 3)) {
            plan.lds.push_back(i); seenLds.insert(i);
          } else if (!interleavePure(i))
            break;
        }
        // The split is structural; no fixed tile/read count or wave geometry.
        if (plan.lds.size() < 2)
          continue;
        for (auto *i = inst.getPrevNode(); i; i = i->getPrevNode()) {
          if (interleaveGlobalRead(i))
            plan.global.push_back(i);
          else if (!interleavePure(i))
            break;
        }
        if (plan.global.empty())
          continue;
        std::reverse(plan.global.begin(), plan.global.end());
        plan.middle = plan.lds[plan.lds.size()/2];
        llvm::DenseSet<llvm::Instruction *> moving;
        for (auto *g : plan.global) moving.insert(g);
        bool legal = true;
        llvm::SmallVector<llvm::Instruction *> work(plan.global.begin(), plan.global.end());
        // Preserve SSA for pure unpack/cast/shuffle users materialized before
        // the new position. Do not duplicate them or cross any effect boundary.
        for (unsigned w = 0; w < work.size() && legal; ++w) {
          auto *value = work[w];
          for (auto &use : value->uses()) {
            auto *user = llvm::dyn_cast<llvm::Instruction>(use.getUser());
            if (user && moving.contains(user)) continue;
            if (user && !llvm::isa<llvm::PHINode>(user) &&
                user->getParent() == &bb && user->comesBefore(plan.middle)) {
              if (!interleavePure(user)) { legal = false; break; }
              if (moving.insert(user).second) work.push_back(user);
              continue;
            }
            legal &= interleaveUseDominated(plan.middle, use, dt);
          }
        }
        for (auto &candidate : bb)
          if (moving.contains(&candidate)) plan.moving.push_back(&candidate);
        for (auto *g : plan.global)
          legal &= !claimedGlobal.contains(g);
        for (auto *value : plan.moving)
          for (auto &operand : value->operands())
            if (auto *def = llvm::dyn_cast<llvm::Instruction>(operand.get()))
              legal &= moving.contains(def) || dt.dominates(def, plan.middle);
        if (!legal) { ++stats.rejectedDependencies; continue; }
        for (auto *g : plan.global) claimedGlobal.insert(g);
        plans.push_back(std::move(plan));
      }
    }
    if (plans.empty()) continue;
    struct Undo { llvm::Instruction *inst, *next; };
    llvm::SmallVector<Undo> undo;
    llvm::SmallVector<llvm::Instruction *> newFences;
    for (auto &plan : plans)
      for (auto *g : plan.moving)
        undo.push_back({g, g->getNextNode()});
    auto *previousFenceDecl = module->getFunction("llvm.amdgcn.sched.barrier");
    for (auto &plan : plans) {
      for (auto *g : plan.moving)
        g->moveBefore(plan.middle->getIterator()); // SDK22 moveBefore iterator; avoids moveAfter iterator ABI.
      if (addFences) {
        auto *f = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::amdgcn_sched_barrier);
        llvm::IRBuilder<> before(plan.global.front()), after(plan.middle);
        before.SetCurrentDebugLocation(plan.lds.front()->getDebugLoc());
        after.SetCurrentDebugLocation(plan.lds.front()->getDebugLoc());
        newFences.push_back(before.CreateCall(f, {before.getInt32(0)}));
        newFences.push_back(after.CreateCall(f, {after.getInt32(0)}));
      }
    }
    if (llvm::verifyFunction(fn, &llvm::errs())) {
      for (auto *f : newFences) f->eraseFromParent();
      for (auto it = undo.rbegin(); it != undo.rend(); ++it)
        it->inst->moveBefore(it->next->getIterator());
      if (!previousFenceDecl)
        if (auto *decl = module->getFunction("llvm.amdgcn.sched.barrier"))
          if (decl->isDeclaration() && decl->use_empty()) decl->eraseFromParent();
      ++stats.invalidFunctions;
      continue;
    }
    // Mark only the reads actually moved after successful verification.
    // No region/shape counts or imported assertions are proof inputs.
    auto *done = llvm::MDNode::get(module->getContext(),
        {llvm::MDString::get(module->getContext(), "v1")});
    for (auto &plan : plans)
      for (auto *g : plan.global)
        g->setMetadata("hcu.readonly.interleave.done", done);
    stats.windows += plans.size();
    stats.fences += newFences.size();
    for (auto &plan : plans) {
      stats.globalReads += plan.global.size();
      stats.pureDependents += plan.moving.size() - plan.global.size();
      stats.ldsReads += plan.lds.size();
    }
  }
  return stats;
}
} // namespace
