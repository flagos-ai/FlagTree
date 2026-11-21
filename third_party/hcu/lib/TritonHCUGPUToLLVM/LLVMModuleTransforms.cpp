#include "TritonHCUGPUToLLVM/LLVMModuleTransforms.h"
#include "TritonHCUGPUToLLVM/FactorBufferStoreOffsets.h"
#include "TritonHCUGPUToLLVM/PackBf16Rounds.h"
#include "TritonHCUGPUToLLVM/ShiftedDisjointOr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <array>
#include <optional>
using mlir::triton::HCU::ReadOnlyInterleaveStats;
using mlir::triton::HCU::TupledRegisterFenceStats;
#include "ReadOnlyLoadInterleave.inc"
#include "TupledRegisterFences.inc"
// Run after LLVM optimization and only for the affected native compiler.
static unsigned preserveLdsBarrierOrder(llvm::Module *module,
                                        const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), false);
  auto *fence = llvm::InlineAsm::get(type, "", "~{memory}", true);
  unsigned count = 0;
  for (auto &fn : *module)
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || callee->getName() != "llvm.amdgcn.s.barrier")
          continue;
        llvm::IRBuilder<> builder(call);
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        builder.CreateCall(fence);
        ++count;
      }
  return count;
}

// LocalBarrierOp synchronizes shared/register operations, not unrelated VMEM.
// DTK LLVM 18 conservatively drains VMEM at the generic barrier intrinsic.
// Preserve its provenance through optimization, then select the same hardware
// barrier with explicit convergence and memory effects. Global/other barriers
// keep their intrinsic. Explicit async waits and LDS waits are untouched.
// Keep synchronous functions on the established native barrier path. This
// optimization targets outstanding global-to-LDS copies, not GPR loads.
static llvm::Expected<unsigned>
stripSynchronousBarrierMarkers(llvm::Module *module, const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *marker = module->getFunction("__triton_hcu_local_barrier_marker");
  if (!marker)
    return 0;
  llvm::DenseMap<llvm::Function *, bool> hasAsyncCopy;
  for (auto &fn : *module)
    for (auto &block : fn)
      for (auto &inst : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *callee = call->getCalledFunction())
            if (callee->getName().starts_with(
                    "llvm.amdgcn.raw.ptr.buffer.load.lds") ||
                callee->getName().starts_with(
                    "llvm.amdgcn.struct.ptr.buffer.load.lds") ||
                callee->getName().starts_with("llvm.hcu.raw.buffer.load.lds"))
              hasAsyncCopy[&fn] = true;
  llvm::SmallVector<llvm::CallInst *, 8> remove;
  for (auto *user : marker->users()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(user);
    if (!call || call->getCalledFunction() != marker || call->arg_size() != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid HCU local barrier marker use");
    if (!hasAsyncCopy.lookup(call->getFunction()))
      remove.push_back(call);
  }
  for (auto *call : remove)
    call->eraseFromParent();
  if (marker->use_empty())
    marker->eraseFromParent();
  return remove.size();
}

static llvm::Expected<unsigned>
lowerTaggedLocalBarriers(llvm::Module *module, const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *marker = module->getFunction("__triton_hcu_local_barrier_marker");
  if (!marker)
    return 0;
  llvm::SmallVector<llvm::CallInst *, 8> tags;
  for (auto *user : marker->users()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(user);
    if (!call || call->getCalledFunction() != marker || call->arg_size() != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid HCU local barrier marker use");
    tags.push_back(call);
  }
  unsigned changed = 0;
  for (auto *tag : tags) {
    llvm::CallInst *barrier = nullptr;
    for (auto *inst = tag->getPrevNode(); inst; inst = inst->getPrevNode()) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(inst)) {
        auto *callee = call->getCalledFunction();
        if (callee && callee->getName() == "llvm.amdgcn.s.barrier") {
          barrier = call;
          break;
        }
        if (callee && callee->getName() == "llvm.amdgcn.sched.barrier")
          continue;
      }
      if (inst->mayReadOrWriteMemory() || inst->mayHaveSideEffects())
        break;
    }
    if (barrier && barrier->getNumOperandBundles() == 0) {
      auto isRange = [](llvm::Instruction *inst) {
        auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        return callee && callee->getName() == "llvm.amdgcn.sched.barrier" &&
               call->arg_size() == 1 &&
               llvm::isa<llvm::ConstantInt>(call->getArgOperand(0));
      };
      bool iglp = false;
      for (auto &inst : *barrier->getParent())
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *callee = call->getCalledFunction()) {
            auto name = callee->getName();
            iglp |= name == "llvm.amdgcn.iglp.opt";
          }
      bool paired =
          isRange(barrier->getPrevNode()) && isRange(barrier->getNextNode());
      // IGroupLP rebuilds native instruction range edges. Opaque barriers
      // require the existing AMD full-range contract; permissive MFMA masks
      // can detach their memory dependencies. Unsupported contracts keep the
      // conservative intrinsic, as do convergence/deoptimization bundles.
      if (iglp) {
        tag->eraseFromParent();
        continue;
      }
      if (!paired) {
        // Establish AMD's full-range contract even without prior scheduler
        // hints. Otherwise native scheduling can move publication waits
        // across preceding MMACs. Keep fences and IGroupLP fallback unchanged.
        auto *boundary = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::amdgcn_sched_barrier);
        llvm::IRBuilder<> before(barrier), after(barrier->getNextNode());
        before.SetCurrentDebugLocation(barrier->getDebugLoc());
        after.SetCurrentDebugLocation(barrier->getDebugLoc());
        before.CreateCall(boundary, {before.getInt32(0)});
        after.CreateCall(boundary, {after.getInt32(0)});
        paired = true;
      }
      if (paired) {
        auto *zero = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(module->getContext()), 0);
        llvm::cast<llvm::CallInst>(barrier->getPrevNode())
            ->setArgOperand(0, zero);
        llvm::cast<llvm::CallInst>(barrier->getNextNode())
            ->setArgOperand(0, zero);
        // Match AMD BlockPingpong's boundary before async_wait. The emitted
        // LocalBarrier DS0 wait belongs to this publication, not to preceding
        // MMACs; keep it inside the native scheduling range as well.
        auto *begin = llvm::cast<llvm::CallInst>(barrier->getPrevNode());
        auto getWait = [](llvm::Instruction *inst) -> llvm::CallInst * {
          auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(inst);
          auto *callee = call ? call->getCalledFunction() : nullptr;
          if (!callee || callee->getName() != "llvm.amdgcn.s.waitcnt" ||
              call->arg_size() != 1 || call->getNumOperandBundles() ||
              !call->getArgOperand(0)->getType()->isIntegerTy(32) ||
              !llvm::isa<llvm::ConstantInt>(call->getArgOperand(0)))
            return nullptr;
          return call;
        };
        auto *previous = begin->getPrevNode();
        if (auto *fence = llvm::dyn_cast_or_null<llvm::CallInst>(previous))
          if (auto *code =
                  llvm::dyn_cast<llvm::InlineAsm>(fence->getCalledOperand()))
            if (code->getAsmString().empty() &&
                code->getConstraintString() == "~{memory}" &&
                code->hasSideEffects() && fence->arg_empty() &&
                !fence->getNumOperandBundles())
              previous = previous->getPrevNode();
        auto *wait = getWait(previous);
        // Canonical gfx936 LocalBarrier wait: vmcnt(63), lgkmcnt(0).
        if (wait && llvm::cast<llvm::ConstantInt>(wait->getArgOperand(0))
                            ->getZExtValue() == 0xc07f) {
          if (auto *asyncWait = getWait(wait->getPrevNode()))
            wait = asyncWait;
          begin->moveBefore(wait->getIterator());
        }
      }
      llvm::IRBuilder<> builder(barrier);
      builder.SetCurrentDebugLocation(barrier->getDebugLoc());
      auto *code = llvm::InlineAsm::get(barrier->getFunctionType(), "s_barrier",
                                        "~{memory}", true);
      auto *call = builder.CreateCall(code);
      call->setConvergent();
      call->setDoesNotThrow();
      barrier->eraseFromParent();
      ++changed;
    }
    // An unrecognized ordering keeps the conservative intrinsic lowering.
    tag->eraseFromParent();
  }
  if (marker->use_empty())
    marker->eraseFromParent();
  return changed;
}

// Preserve independently proved scalar descriptor indices after LLVM O3.
// Do not scalarize pointer bits or infer uniformity from a tensor/launch shape.
// The AS1 GEP and resource fields keep their exact per-lane bit values; only
// defined, wave-uniform integers receive the target's scalar register hint.
static unsigned preserveUniformResourceIndices(llvm::Module *module,
                                               const std::string &arch) {
  if (arch != "gfx936")
    return 0; // The reused uniformity proof specifies this target ABI.
  unsigned changed = 0;
  for (auto &function : *module) {
    if (function.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    llvm::SmallVector<llvm::CallInst *, 8> resources;
    for (auto &block : function)
      for (auto &inst : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          auto *callee = call->getCalledFunction();
          if (callee && callee->isIntrinsic() &&
              callee->getName().starts_with("llvm.amdgcn.make.buffer.rsrc.") &&
              call->arg_size() == 4 && !call->getNumOperandBundles() &&
              !call->isMustTailCall())
            resources.push_back(call);
        }
    for (auto *call : resources) {
      // Imported markers can only disable this optimization, never prove it.
      if (call->getMetadata("hcu.uniform.resource.indices.done"))
        continue;
      auto *gep =
          llvm::dyn_cast<llvm::GetElementPtrInst>(call->getArgOperand(0));
      auto *pointer = llvm::dyn_cast<llvm::PointerType>(call->getType());
      auto *count = call->getArgOperand(2);
      if (!gep || gep->getPointerAddressSpace() != 1 || !pointer ||
          pointer->getAddressSpace() != 8 || !count->getType()->isIntegerTy(64))
        continue;
      flagtree::hcu::ProvedWaveUniformInteger proof(call);
      if (!proof.prove(count))
        continue;
      bool valid = true, nonconstant = !llvm::isa<llvm::ConstantInt>(count);
      for (llvm::Value *index : gep->indices()) {
        unsigned bits = index->getType()->getIntegerBitWidth();
        valid &= bits <= 64 && proof.prove(index);
        nonconstant |= !llvm::isa<llvm::ConstantInt>(index);
      }
      if (!valid || !nonconstant)
        continue;
      // The existing target resource count is u32. Keep its upper zero bits
      // explicit for the SDK ABI; never truncate an unproved wide count.
      auto known =
          llvm::computeKnownBits(count, module->getDataLayout(), nullptr, call);
      auto upper = llvm::APInt::getHighBitsSet(64, 32);
      if ((known.Zero & upper) != upper)
        continue;
      llvm::IRBuilder<> builder(call);
      builder.SetCurrentDebugLocation(call->getDebugLoc());
      auto *read = llvm::Intrinsic::getOrInsertDeclaration(
          module, llvm::Intrinsic::amdgcn_readfirstlane,
          {builder.getInt32Ty()});
      auto scalar = [&](llvm::Value *value) -> llvm::Value * {
        if (llvm::isa<llvm::ConstantInt>(value))
          return value;
        auto *type = value->getType();
        if (type->getIntegerBitWidth() <= 32) {
          auto *word = builder.CreateZExtOrTrunc(value, builder.getInt32Ty());
          return builder.CreateZExtOrTrunc(builder.CreateCall(read, {word}),
                                           type);
        }
        auto *low = builder.CreateTrunc(value, builder.getInt32Ty());
        auto *high = builder.CreateTrunc(builder.CreateLShr(value, 32),
                                         builder.getInt32Ty());
        low = builder.CreateCall(read, {low});
        high = builder.CreateCall(read, {high});
        return builder.CreateOr(
            builder.CreateZExt(low, type),
            builder.CreateShl(builder.CreateZExt(high, type), 32));
      };
      auto *copy = llvm::cast<llvm::GetElementPtrInst>(gep->clone());
      bool indexChanged = false;
      for (unsigned i = 1; i < copy->getNumOperands(); ++i) {
        auto *index = copy->getOperand(i);
        auto *replacement = scalar(index);
        indexChanged |= replacement != index;
        copy->setOperand(i, replacement);
      }
      if (indexChanged) {
        copy->insertBefore(
            call->getIterator()); // Retain all original GEP flags.
        call->setArgOperand(0, copy);
      } else {
        copy->deleteValue();
      }
      if (!llvm::isa<llvm::ConstantInt>(count)) {
        auto *word = builder.CreateTrunc(count, builder.getInt32Ty());
        call->setArgOperand(2,
                            builder.CreateZExt(scalar(word), count->getType()));
      }
      call->setMetadata("hcu.uniform.resource.indices.done",
                        llvm::MDNode::get(module->getContext(), {}));
      ++changed;
    }
  }
  return changed;
}

// DTK DAG selection can discard the x2 byte scale of a prefetched buffer
// load. Protect its element offset after LLVM optimization: earlier freezes
// are propagated away by LLVM 22 before the module reaches DTK.
static unsigned preserveBufferLoadByteOffsets(llvm::Module *module,
                                              const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  llvm::SmallSetVector<llvm::BinaryOperator *, 16> shifts;
  for (auto &fn : *module) {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee ||
            !callee->getName().starts_with(
                "llvm.amdgcn.raw.ptr.buffer.load.") ||
            callee->getName().starts_with(
                "llvm.amdgcn.raw.ptr.buffer.load.lds"))
          continue;
        llvm::SmallVector<llvm::Value *, 4> offsets{call->getArgOperand(1)};
        while (!offsets.empty()) {
          auto *offset = offsets.pop_back_val();
          if (auto *select = llvm::dyn_cast<llvm::SelectInst>(offset)) {
            offsets.push_back(select->getTrueValue());
            offsets.push_back(select->getFalseValue());
          } else if (auto *shift =
                         llvm::dyn_cast<llvm::BinaryOperator>(offset)) {
            auto *amount =
                llvm::dyn_cast<llvm::ConstantInt>(shift->getOperand(1));
            if (shift->getOpcode() == llvm::Instruction::Shl &&
                shift->getType()->isIntegerTy(32) && amount &&
                amount->isOne() &&
                !llvm::isa<llvm::FreezeInst>(shift->getOperand(0)))
              shifts.insert(shift);
          }
        }
      }
  }
  for (auto *shift : shifts) {
    llvm::IRBuilder<> builder(shift);
    builder.SetCurrentDebugLocation(shift->getDebugLoc());
    shift->setOperand(
        0, builder.CreateFreeze(shift->getOperand(0), "buffer.element.offset"));
  }
  return shifts.size();
}

// Keep ready LDS reads ahead of future direct-to-LDS copies. Older gfx936
// codegen otherwise schedules the copies first and drains them before the
// reads, even when the existing pipeline alias scopes prove that their slots
// differ. This runs after LLVM optimization, under the opt-in local-prefetch
// hint.
static unsigned preferReadySharedLoads(llvm::Module *module,
                                       const std::string &arch) {
  if (arch != "gfx936")
    return 0;

  auto isCopy = [](llvm::Instruction &inst) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
    auto *fn = call ? call->getCalledFunction() : nullptr;
    if (!fn || !call->getType()->isVoidTy())
      return false;
    return (fn->getName() == "llvm.amdgcn.raw.ptr.buffer.load.lds" &&
            call->arg_size() == 7) ||
           (fn->getName() == "llvm.amdgcn.struct.ptr.buffer.load.lds" &&
            call->arg_size() == 8);
  };
  auto isNativeRead = [](llvm::Instruction &inst) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
    auto *fn = call ? call->getCalledFunction() : nullptr;
    return fn && fn->getName().starts_with("llvm.hcu.ds.read.m32x16.b16.");
  };
  auto isRead = [&](llvm::Instruction &inst) {
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
      return load->getPointerAddressSpace() == 3 && !load->isVolatile() &&
             !load->isAtomic();
    return isNativeRead(inst);
  };
  auto disjoint = [](llvm::Instruction *copy, llvm::Instruction *read) {
    auto *scopes = copy->getMetadata(llvm::LLVMContext::MD_alias_scope);
    auto *noalias = read->getMetadata(llvm::LLVMContext::MD_noalias);
    if (!scopes || !scopes->getNumOperands() || !noalias)
      return false;
    // Require every existing copy scope to be excluded by every ready read.
    // Never manufacture alias information from pointer arithmetic or tile
    // sizes.
    for (const auto &scope : scopes->operands()) {
      bool found = false;
      for (const auto &excluded : noalias->operands())
        found |= scope.get() == excluded.get();
      if (!found)
        return false;
    }
    return true;
  };

  unsigned changed = 0;
  for (auto &fn : *module) {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto &block : fn) {
      auto *branch = llvm::dyn_cast<llvm::BranchInst>(block.getTerminator());
      if (!branch || !branch->isConditional() ||
          (branch->getSuccessor(0) != &block &&
           branch->getSuccessor(1) != &block))
        continue;
      // An unrolled loop has several publication/reuse epochs. Alias scopes
      // are valid for each epoch, not for arbitrary pairs across its barriers.
      // Partition at every other memory effect, synchronization or terminator;
      // never move a copy across such a boundary or alter a completion wait.
      struct Region {
        llvm::SmallVector<llvm::Instruction *> copies, reads;
        unsigned matrixOps = 0;
      };
      llvm::SmallVector<Region> regions;
      Region region;
      auto finish = [&] {
        if (!region.copies.empty() && !region.reads.empty())
          regions.push_back(std::move(region));
        region = Region{};
      };
      for (auto &inst : block) {
        if (isCopy(inst))
          region.copies.push_back(&inst);
        else if (isRead(inst))
          region.reads.push_back(&inst);
        // A true assumption carries storage facts but executes no memory
        // operation. Moving a result-free copy later keeps its SSA operands
        // defined and does not move the assumption or any operation it uses.
        else if (auto *assume = llvm::dyn_cast<llvm::AssumeInst>(&inst);
                 assume &&
                 llvm::isa<llvm::ConstantInt>(assume->getArgOperand(0)) &&
                 llvm::cast<llvm::ConstantInt>(assume->getArgOperand(0))
                     ->isOne())
          continue;
        else if (inst.isTerminator() || inst.mayReadOrWriteMemory() ||
                 inst.mayHaveSideEffects())
          finish();
        else if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          if (auto *callee = call->getCalledFunction();
              callee && callee->getName().starts_with("llvm.hcu.mmac."))
            ++region.matrixOps;
        }
      }
      finish();
      for (auto &region : regions) {
        auto &copies = region.copies;
        auto &reads = region.reads;
        if (!copies.back()->comesBefore(reads.back()))
          continue;
        bool legal = true;
        for (auto *copy : copies)
          for (auto *read : reads)
            legal &= disjoint(copy, read);
        if (!legal)
          continue;
        // The all-pairs disjoint proof above authorizes movement. The issue
        // policy below only decides how much of each producer group to keep
        // in flight; it must never substitute for that proof.
        auto copyScope = [](llvm::Instruction *copy) {
          return copy->getMetadata(llvm::LLVMContext::MD_alias_scope);
        };
        auto isWideCopy = [](llvm::Instruction *copy) {
          auto *size = llvm::dyn_cast<llvm::ConstantInt>(
              llvm::cast<llvm::CallInst>(copy)->getArgOperand(2));
          return size && size->equalsInt(16);
        };
        // Identify a singleton producer followed by a larger producer group,
        // regardless of how many instructions vectorization used for the
        // latter.
        auto companions = llvm::ArrayRef(copies).drop_front();
        bool keepFirstCopy =
            companions.size() >= 2 &&
            copies.front()->comesBefore(reads.front()) &&
            copyScope(copies.front()) != copyScope(companions.front()) &&
            llvm::all_of(companions,
                         [&](auto *copy) {
                           return copyScope(copy) ==
                                  copyScope(companions.front());
                         }) &&
            llvm::all_of(copies, isWideCopy);
        auto *schedulingBarrier = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::amdgcn_sched_barrier);
        if (keepFirstCopy) {
          auto isStructured = [](llvm::Instruction *copy) {
            return llvm::cast<llvm::CallInst>(copy)
                       ->getCalledFunction()
                       ->getName() == "llvm.amdgcn.struct.ptr.buffer.load.lds";
          };
          // Start the LDS read group before an indexed copy when the
          // companion copies use raw addressing. Keep the existing producer
          // order, wait counts and alias proofs. Both sides of the copy
          // need scheduling fences to preserve the ready-read split.
          if (reads.size() > 1 && isStructured(copies.front()) &&
              llvm::none_of(companions, isStructured)) {
            // This is a target cost heuristic, not a dependence requirement.
            // Preserve the validated short/long compute split; changing this
            // threshold needs performance evidence, not an alias proof.
            constexpr unsigned maxShortMatrixGroup = 16;
            bool shortGroup =
                region.matrixOps > 0 && region.matrixOps <= maxShortMatrixGroup;
            auto *read = shortGroup ? reads.back() : reads[reads.size() / 2];
            llvm::IRBuilder<> split(read);
            auto *beforeCopy =
                split.CreateCall(schedulingBarrier, {split.getInt32(0)});
            copies.front()->moveAfter(beforeCopy);
            split.CreateCall(schedulingBarrier, {split.getInt32(0)});
          } else {
            llvm::IRBuilder<> beforeRead(reads.front());
            beforeRead.CreateCall(schedulingBarrier, {beforeRead.getInt32(0)});
          }
        }
        // These copies have no SSA result. Moving them later retains all
        // operand definitions and their original relative order.
        auto *lastRead = reads.back();
        llvm::IRBuilder<> builder(lastRead->getNextNode());
        auto *fence =
            builder.CreateCall(schedulingBarrier, {builder.getInt32(0)});
        llvm::Instruction *after = fence;
        for (auto *copy : llvm::ArrayRef(copies).drop_front(keepFirstCopy)) {
          copy->moveAfter(after);
          after = copy;
        }
        ++changed;
      }
    }
  }
  return changed;
}

namespace {
// Use physical LDS objects rather than one scope for all DMA operations.
// LLVM's AMDGPU wait pass tracks representative stores by complete AAInfo;
// separate direct-G2L packets therefore need distinct scopes. This follows
// AMDGPULowerModuleLDS's object-scope convention without changing any waits.
struct LdsOffsetRange {
  uint64_t low;
  uint64_t high;
};

// Layout addresses combine masked bitfields and fragment additions. Recurse
// through a bounded OR/add/sub tree before generic ValueTracking exhausts its
// depth budget. KnownBits transfers model modular arithmetic; no no-wrap flags,
// launch assumptions or frontend witnesses are required. Unknown expressions
// and exhausted budgets retain their original conservative facts.
static llvm::KnownBits getLdsIndexKnownBitsImpl(
    llvm::Value *value, const llvm::DataLayout &layout, unsigned &budget,
    llvm::DenseMap<llvm::Value *, llvm::KnownBits> &cache) {
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  auto remember = [&](llvm::KnownBits bits) {
    cache.try_emplace(value, bits);
    return bits;
  };
  auto known = llvm::computeKnownBits(value, layout);
  // A bound inside all LDS can still cross a smaller physical object.
  // Refine nonconstant expressions within the node budget before partitioning.
  if (!budget || known.isConstant())
    return remember(known);
  --budget;
  // readfirstlane selects one lane's input without changing its bits. Bounds
  // known for every possible input therefore also hold for that uniform copy
  // address, before InstCombine propagates its demanded bits into the caller.
  if (auto *call = llvm::dyn_cast<llvm::CallBase>(value))
    if (call->getIntrinsicID() == llvm::Intrinsic::amdgcn_readfirstlane)
      return remember(known.unionWith(getLdsIndexKnownBitsImpl(
          call->getArgOperand(0), layout, budget, cache)));
  // A layout can materialize a coordinate bit through select even when
  // the address calculation has already constrained that bit. Use the same
  // bounded analysis on equality operands before falling back to both arms.
  if (auto *select = llvm::dyn_cast<llvm::SelectInst>(value)) {
    auto *compare = llvm::dyn_cast<llvm::ICmpInst>(select->getCondition());
    if (compare && compare->isEquality() &&
        compare->getOperand(0)->getType()->isIntegerTy()) {
      auto lhs = getLdsIndexKnownBitsImpl(compare->getOperand(0), layout,
                                          budget, cache);
      auto rhs = getLdsIndexKnownBitsImpl(compare->getOperand(1), layout,
                                          budget, cache);
      bool different =
          lhs.One.intersects(rhs.Zero) || rhs.One.intersects(lhs.Zero);
      bool equal = lhs.isConstant() && rhs.isConstant() && lhs.One == rhs.One;
      if (different || equal) {
        bool takeTrue =
            (compare->getPredicate() == llvm::ICmpInst::ICMP_EQ) == equal;
        auto *arm = takeTrue ? select->getTrueValue() : select->getFalseValue();
        return remember(known.unionWith(
            getLdsIndexKnownBitsImpl(arm, layout, budget, cache)));
      }
    }
    return remember(known);
  }
  auto *op = llvm::dyn_cast<llvm::Operator>(value);
  if (!op)
    return remember(known);
  unsigned code = op->getOpcode();
  if (code == llvm::Instruction::Shl || code == llvm::Instruction::LShr) {
    auto *shift = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(1));
    unsigned width = known.getBitWidth();
    if (!shift || shift->getLimitedValue(width) >= width)
      return remember(known);
    unsigned amount = shift->getZExtValue();
    auto input =
        getLdsIndexKnownBitsImpl(op->getOperand(0), layout, budget, cache);
    auto refined = input;
    if (code == llvm::Instruction::Shl) {
      refined.Zero =
          (input.Zero << amount) | llvm::APInt::getLowBitsSet(width, amount);
      refined.One = input.One << amount;
    } else {
      refined.Zero =
          input.Zero.lshr(amount) | llvm::APInt::getHighBitsSet(width, amount);
      refined.One = input.One.lshr(amount);
    }
    return remember(known.unionWith(refined));
  }
  if (code == llvm::Instruction::Mul) {
    // Multiplication by a constant can hide disjoint shifted bitfields.
    // Sum its set-bit shifts in modular arithmetic, without no-wrap flags.
    auto *constant = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(1));
    llvm::Value *input = op->getOperand(0);
    if (!constant) {
      constant = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(0));
      input = op->getOperand(1);
    }
    if (!constant)
      return remember(known);
    auto lhs = getLdsIndexKnownBitsImpl(input, layout, budget, cache);
    unsigned width = lhs.getBitWidth();
    llvm::SmallVector<unsigned, 8> shifts;
    for (unsigned bit = 0; bit < width; ++bit)
      if (constant->getValue()[bit]) {
        shifts.push_back(bit);
        if (shifts.size() > budget)
          return remember(known);
      }
    budget -= shifts.size();
    llvm::KnownBits product(width);
    product.Zero.setAllBits();
    for (unsigned shift : shifts) {
      auto term = lhs;
      term.Zero =
          (lhs.Zero << shift) | llvm::APInt::getLowBitsSet(width, shift);
      term.One = lhs.One << shift;
      product =
          llvm::KnownBits::computeForAddSub(true, false, false, product, term);
    }
    return remember(known.unionWith(product));
  }
  if (code != llvm::Instruction::Or && code != llvm::Instruction::And &&
      code != llvm::Instruction::Xor && code != llvm::Instruction::Add &&
      code != llvm::Instruction::Sub)
    return remember(known);
  auto lhs = getLdsIndexKnownBitsImpl(op->getOperand(0), layout, budget, cache);
  auto rhs = getLdsIndexKnownBitsImpl(op->getOperand(1), layout, budget, cache);
  auto refined = lhs;
  if (code == llvm::Instruction::Or)
    refined |= rhs;
  else if (code == llvm::Instruction::And) {
    refined.Zero = lhs.Zero | rhs.Zero;
    refined.One = lhs.One & rhs.One;
  } else if (code == llvm::Instruction::Xor) {
    refined.Zero = (lhs.Zero & rhs.Zero) | (lhs.One & rhs.One);
    refined.One = (lhs.Zero & rhs.One) | (lhs.One & rhs.Zero);
  } else
    refined = llvm::KnownBits::computeForAddSub(code == llvm::Instruction::Add,
                                                false, false, lhs, rhs);
  return remember(known.unionWith(refined));
}

static llvm::KnownBits getLdsIndexKnownBits(llvm::Value *value,
                                            const llvm::DataLayout &layout) {
  // Layout expressions are DAGs: shared lane/warp terms must consume the
  // bounded analysis budget only once. Cache only within this one query, so
  // no result survives an IR mutation or changes another pointer's budget.
  unsigned budget = 64;
  llvm::DenseMap<llvm::Value *, llvm::KnownBits> cache;
  return getLdsIndexKnownBitsImpl(value, layout, budget, cache);
}

static std::optional<LdsOffsetRange>
getBoundedLdsOffset(llvm::Value *ptr, const llvm::DataLayout &layout,
                    unsigned depth = 0) {
  if (depth > 16 || !ptr->getType()->isPointerTy() ||
      ptr->getType()->getPointerAddressSpace() != 3)
    return std::nullopt;
  if (auto *global = llvm::dyn_cast<llvm::GlobalVariable>(ptr)) {
    if (global->getName() == "global_smem")
      return LdsOffsetRange{0, 0};
    return std::nullopt;
  }
  auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
  if (!gep)
    return std::nullopt;
  auto base = getBoundedLdsOffset(gep->getPointerOperand(), layout, depth + 1);
  if (!base)
    return std::nullopt;
  llvm::SmallMapVector<llvm::Value *, llvm::APInt, 4> variables;
  unsigned indexBits = layout.getIndexSizeInBits(3);
  if (indexBits == 0 || indexBits > 64)
    return std::nullopt;
  llvm::APInt constant(indexBits, 0);
  if (!gep->collectOffset(layout, indexBits, variables, constant))
    return std::nullopt;
  int64_t low = int64_t(base->low) + constant.getSExtValue();
  int64_t high = int64_t(base->high) + constant.getSExtValue();
  for (auto &[value, scale] : variables) {
    if (!value->getType()->isIntegerTy() || scale.isNegative() ||
        scale.getLimitedValue(65537) > 65536)
      return std::nullopt;
    llvm::KnownBits bits = getLdsIndexKnownBits(value, layout);
    auto maximum = bits.getMaxValue();
    // GEP indices are signed. Reject negative or wrapping index ranges.
    if (!bits.Zero.isSignBitSet() || maximum.getActiveBits() > 16)
      return std::nullopt;
    low += bits.getMinValue().getZExtValue() * scale.getZExtValue();
    high += maximum.getZExtValue() * scale.getZExtValue();
    if (low < 0 || high >= 65536)
      return std::nullopt;
  }
  if (low < 0 || high < low || high >= 65536)
    return std::nullopt;
  return LdsOffsetRange{uint64_t(low), uint64_t(high)};
}

static bool hasHcuAliasScope(llvm::MDNode *list, llvm::StringRef name,
                             bool prefix = false) {
  if (!list)
    return false;
  for (auto &operand : list->operands()) {
    auto *scope = llvm::dyn_cast_or_null<llvm::MDNode>(operand.get());
    if (!scope)
      continue;
    for (auto &field : scope->operands())
      if (auto *text = llvm::dyn_cast_or_null<llvm::MDString>(field.get()))
        if (prefix ? text->getString().starts_with(name)
                   : text->getString() == name)
          return true;
  }
  return false;
}

static unsigned annotateLdsObjectScopes(llvm::Module *module,
                                        const std::string &arch,
                                        unsigned waveSize) {
  if (arch != "gfx936" || waveSize != 64)
    return 0;
  unsigned changed = 0;
  for (auto &fn : *module) {
    uint64_t objectBytes = 4096;
    struct Access {
      llvm::Instruction *inst;
      LdsOffsetRange range;
      bool structured = false;
    };
    llvm::SmallVector<Access, 32> copies, reads;
    llvm::SmallSetVector<uint64_t, 8> objects;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        llvm::Value *pointer = nullptr;
        uint64_t bytes = 0;
        bool raw = callee &&
                   callee->getName() == "llvm.amdgcn.raw.ptr.buffer.load.lds" &&
                   call->arg_size() == 7;
        bool structured =
            callee &&
            callee->getName() == "llvm.amdgcn.struct.ptr.buffer.load.lds" &&
            call->arg_size() == 8;
        bool copy = raw || structured;
        if (copy) {
          if (!hasHcuAliasScope(
                  inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                  "hcug.AsyncCopies") &&
              !hasHcuAliasScope(
                  inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                  "LDS bytes ", true))
            continue;
          auto *size =
              llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          if (!size ||
              (size->getZExtValue() != 8 && size->getZExtValue() != 16) ||
              (structured && size->getZExtValue() != 16))
            continue; // Structured hardware footprint is verified for16B only.
          pointer = call->getArgOperand(1);
          // Byte width is per lane, as in AMD BufferEmitter::emitLoadToLds.
          // The scalar dst selects a complete wave packet.
          bytes = waveSize * size->getZExtValue();
        } else {
          if ((!hasHcuAliasScope(
                   inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                   "hcug.LocalLoads") &&
               !hasHcuAliasScope(
                   inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                   "LDS reads")) ||
              (!hasHcuAliasScope(
                   inst.getMetadata(llvm::LLVMContext::MD_noalias),
                   "hcug.AsyncCopies") &&
               !hasHcuAliasScope(
                   inst.getMetadata(llvm::LLVMContext::MD_noalias),
                   "LDS bytes ", true)))
            continue;
          if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
            pointer = load->getPointerOperand();
            auto size =
                module->getDataLayout().getTypeStoreSize(load->getType());
            if (size.isScalable() || size.getFixedValue() > 16)
              continue;
            bytes = size.getFixedValue();
          } else if (callee &&
                     (callee->getName() ==
                          "llvm.hcu.ds.read.m32x16.b16.v4i32.p3" ||
                      callee->getName() ==
                          "llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3")) {
            if (call->arg_size() != 2)
              continue;
            auto *mode =
                llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
            if (!mode || !mode->isZero())
              continue;
            pointer = call->getArgOperand(0);
            // Normal and ALT both gather halfwords within an issuing
            // lane's 16-byte row (the finite-read planner proves the
            // distinct cross-lane permutations). Bound the union of all
            // issuing-lane addresses plus [0,14].
            // The range bounds all lanes, including native cross-lane reads.
            bytes = 16;
          }
        }
        if (!pointer || !bytes)
          continue;
        auto range = getBoundedLdsOffset(pointer, module->getDataLayout());
        if (!range || range->high + bytes > 65536)
          continue;
        range->high += bytes; // high is now an exclusive access bound.
        if (copy) {
          bool fits4KiB = range->low / 4096 == (range->high - 1) / 4096;
          bool full8KiB =
              range->low % 8192 == 0 && range->high - range->low == 8192;
          if (!fits4KiB && !full8KiB)
            continue;
          // Large tiles span a proven, aligned 8KiB region across all waves.
          // Coarsening retains aliases; arbitrary crossing ranges still bail.
          if (full8KiB)
            objectBytes = 8192;
          copies.push_back({&inst, *range, structured});
        } else {
          reads.push_back({&inst, *range});
        }
      }
    // Storage extents survive descriptor lowering as standard assume bundles.
    // They are granularity hints only: all exclusions below still come from
    // the actual physical access bounds. In particular, dereferenceable does
    // not mean noalias, and overlapping extents must share a scope.
    llvm::SmallVector<LdsOffsetRange, 8> extents;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *assume = llvm::dyn_cast<llvm::AssumeInst>(&inst);
        if (!assume)
          continue;
        for (unsigned i = 0; i < assume->getNumOperandBundles(); ++i) {
          auto bundle = assume->getOperandBundleAt(i);
          if (bundle.getTagName() != "dereferenceable" ||
              bundle.Inputs.size() != 2)
            continue;
          auto *size = llvm::dyn_cast<llvm::ConstantInt>(bundle.Inputs[1]);
          if (!size || size->getLimitedValue(65537) > 65536 || size->isZero())
            continue;
          auto range =
              getBoundedLdsOffset(bundle.Inputs[0], module->getDataLayout());
          if (!range || range->high + size->getZExtValue() > 65536)
            continue;
          range->high += size->getZExtValue();
          extents.push_back(*range);
        }
      }
    const bool hasNarrowStructuredCopy =
        objectBytes == 8192 && llvm::any_of(copies, [](const Access &copy) {
          return copy.structured &&
                 copy.range.low / 4096 == (copy.range.high - 1) / 4096;
        });
    if ((!extents.empty() || hasNarrowStructuredCopy) && !copies.empty() &&
        !reads.empty()) {
      llvm::SmallVector<LdsOffsetRange, 8> groups;
      for (const auto &copy : copies) {
        // Keep an independently bounded structured packet in its own slot
        // even when another operand needs a wider object. Wider groups that
        // overlap this slot still merge below.
        const bool narrow =
            hasNarrowStructuredCopy && copy.structured &&
            copy.range.low / 4096 == (copy.range.high - 1) / 4096;
        const uint64_t granularity = narrow ? 4096 : objectBytes;
        const uint64_t base = copy.range.low / granularity * granularity;
        LdsOffsetRange group{base, base + granularity};
        const LdsOffsetRange *best = nullptr;
        for (const auto &extent : extents)
          if (extent.low <= copy.range.low && extent.high >= copy.range.high &&
              (!best || extent.high - extent.low < best->high - best->low))
            best = &extent;
        if (best &&
            (!narrow || (group.low <= best->low && best->high <= group.high)))
          group = *best;
        groups.push_back(group);
      }
      llvm::sort(groups, [](const auto &a, const auto &b) {
        return a.low < b.low || (a.low == b.low && a.high < b.high);
      });
      llvm::SmallVector<LdsOffsetRange, 8> merged;
      for (const auto &group : groups) {
        if (!merged.empty() && group.low < merged.back().high)
          merged.back().high = std::max(group.high, merged.back().high);
        else
          merged.push_back(group);
      }
      // Exceeding the hardware budget or having no useful distinction takes
      // the original conservative path below. Never truncate a scope list.
      if (merged.size() >= 2 && merged.size() <= 8) {
        llvm::MDBuilder metadata(module->getContext());
        auto *domain =
            metadata.createAnonymousAliasScopeDomain("gfx936 LDS objects");
        llvm::SmallVector<llvm::MDNode *, 8> scopes;
        for (const auto &group : merged)
          scopes.push_back(metadata.createAnonymousAliasScope(
              domain, "LDS bytes " + std::to_string(group.low) + ".." +
                          std::to_string(group.high)));
        auto *localScope =
            metadata.createAnonymousAliasScope(domain, "LDS reads");
        auto *localList = llvm::MDNode::get(module->getContext(), {localScope});
        for (auto &copy : copies) {
          for (unsigned i = 0; i < merged.size(); ++i)
            if (merged[i].low <= copy.range.low &&
                merged[i].high >= copy.range.high) {
              copy.inst->setMetadata(
                  llvm::LLVMContext::MD_alias_scope,
                  llvm::MDNode::get(module->getContext(), {scopes[i]}));
              ++changed;
              break;
            }
        }
        for (auto &read : reads) {
          llvm::SmallVector<llvm::Metadata *, 8> disjoint;
          for (unsigned i = 0; i < merged.size(); ++i)
            if (merged[i].high <= read.range.low ||
                merged[i].low >= read.range.high)
              disjoint.push_back(scopes[i]);
          if (disjoint.empty())
            continue;
          read.inst->setMetadata(llvm::LLVMContext::MD_alias_scope, localList);
          read.inst->setMetadata(
              llvm::LLVMContext::MD_noalias,
              llvm::MDNode::get(module->getContext(), disjoint));
          ++changed;
        }
        continue;
      }
    }
    for (const auto &copy : copies)
      objects.insert(copy.range.low / objectBytes);
    // LLVM18 reserves eight object-score slots. Coarsen proven byte ranges
    // when 4KiB partitions exceed that budget; merging regions only reduces
    // alias precision. Unsupported dynamic addresses retain conservative
    // scopes.
    llvm::DenseMap<uint64_t, uint64_t> representatives;
    unsigned scopeCount = objects.size();
    if (objects.size() > 8) {
      // Objects indistinguishable to every proven read can share a scoreboard
      // slot without losing any of that read-disjointness information.
      llvm::SmallVector<llvm::SmallVector<uint8_t, 32>, 8> patterns;
      llvm::SmallVector<uint64_t, 8> leaders;
      for (uint64_t object : objects) {
        llvm::SmallVector<uint8_t, 32> pattern;
        for (const auto &read : reads)
          pattern.push_back((object + 1) * objectBytes <= read.range.low ||
                            object * objectBytes >= read.range.high);
        unsigned group = 0;
        while (group < patterns.size() && patterns[group] != pattern)
          ++group;
        if (group == patterns.size()) {
          patterns.push_back(std::move(pattern));
          leaders.push_back(object);
        }
        representatives[object] = leaders[group];
      }
      scopeCount = leaders.size();
    }
    if (scopeCount > 8) {
      representatives.clear();
      objectBytes = 8192;
      objects.clear();
      for (const auto &copy : copies)
        objects.insert(copy.range.low / objectBytes);
      scopeCount = objects.size();
    }
    if (scopeCount < 2 || scopeCount > 8 || reads.empty())
      continue;
    bool useful = llvm::any_of(reads, [&](const Access &read) {
      return llvm::any_of(objects, [&](uint64_t object) {
        return (object + 1) * objectBytes <= read.range.low ||
               object * objectBytes >= read.range.high;
      });
    });
    if (!useful)
      continue;
    llvm::MDBuilder metadata(module->getContext());
    auto *domain =
        metadata.createAnonymousAliasScopeDomain("gfx936 LDS objects");
    llvm::DenseMap<uint64_t, llvm::MDNode *> scopes;
    llvm::DenseMap<uint64_t, llvm::MDNode *> groupScopes;
    for (auto object : objects) {
      uint64_t leader = representatives.count(object)
                            ? representatives.lookup(object)
                            : object;
      auto *&scope = groupScopes[leader];
      if (!scope)
        scope = metadata.createAnonymousAliasScope(
            domain,
            "LDS group led by bytes " + std::to_string(leader * objectBytes));
      scopes[object] = scope;
    }
    auto *localScope = metadata.createAnonymousAliasScope(domain, "LDS reads");
    auto *localList = llvm::MDNode::get(module->getContext(), {localScope});
    for (auto &copy : copies) {
      auto *scope = scopes.lookup(copy.range.low / objectBytes);
      copy.inst->setMetadata(llvm::LLVMContext::MD_alias_scope,
                             llvm::MDNode::get(module->getContext(), {scope}));
      ++changed;
    }
    for (auto &read : reads) {
      llvm::SmallVector<llvm::Metadata *, 8> disjoint;
      for (auto object : objects)
        if ((object + 1) * objectBytes <= read.range.low ||
            object * objectBytes >= read.range.high)
          if (!llvm::is_contained(disjoint, scopes.lookup(object)))
            disjoint.push_back(scopes.lookup(object));
      if (disjoint.empty())
        continue;
      read.inst->setMetadata(llvm::LLVMContext::MD_alias_scope, localList);
      read.inst->setMetadata(llvm::LLVMContext::MD_noalias,
                             llvm::MDNode::get(module->getContext(), disjoint));
      ++changed;
    }
  }
  return changed;
}

} // namespace

// DTK groups LDS DMA writes by AAMDNodes but keeps one pointer per group.
// BasicAA can disambiguate a read from that representative even when it reads
// another fragment of the same group. The wait pass then falls back to its
// general LDS slot and unnecessarily drains future buffers. Forget only the
// first DMA pointer's provenance, retaining the scoped-noalias contract and
// BasicAA everywhere else. A tied empty asm preserves all 32 pointer bits;
// it does not assert non-aliasing or remove any explicit synchronization.
static unsigned preserveLdsDmaAliasScopes(llvm::Module *module,
                                          const std::string &arch) {
  if (arch != "gfx936" || module->getDataLayout().getPointerSizeInBits(3) != 32)
    return 0;
  auto *i32 = llvm::Type::getInt32Ty(module->getContext());
  auto *type = llvm::FunctionType::get(i32, {i32}, false);
  auto *identity = llvm::InlineAsm::get(type, "", "=v,0", false);
  unsigned changed = 0;
  for (auto &fn : *module) {
    llvm::SmallVector<llvm::AAMDNodes, 8> seen;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || call->getNumOperandBundles())
          continue;
        auto name = callee->getName();
        bool raw = (name == "llvm.amdgcn.raw.ptr.buffer.load.lds" ||
                    name.starts_with("llvm.amdgcn.raw.ptr.buffer.load.lds.")) &&
                   call->arg_size() == 7;
        bool structured =
            (name == "llvm.amdgcn.struct.ptr.buffer.load.lds" ||
             name.starts_with("llvm.amdgcn.struct.ptr.buffer.load.lds.")) &&
            call->arg_size() == 8;
        // Both typed DMA forms carry the exact same 32-bit LDS operand.
        // Preserve the existing scope contract and all raw-path behavior.
        if (!raw && !structured)
          continue;
        auto *ptr = call->getArgOperand(1);
        auto *ptrType = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
        auto aa = call->getAAMetadata();
        if (!ptrType || ptrType->getAddressSpace() != 3 || !aa.Scope ||
            std::find(seen.begin(), seen.end(), aa) != seen.end())
          continue;
        seen.push_back(aa);
        // Avoid adding another identity if this module is visited again.
        if (auto *cast = llvm::dyn_cast<llvm::IntToPtrInst>(ptr))
          if (auto *opaque =
                  llvm::dyn_cast<llvm::CallInst>(cast->getOperand(0)))
            if (auto *code =
                    llvm::dyn_cast<llvm::InlineAsm>(opaque->getCalledOperand()))
              if (code->getAsmString().empty() &&
                  code->getConstraintString() == "=v,0" &&
                  !code->hasSideEffects() && !opaque->getNumOperandBundles() &&
                  opaque->arg_size() == 1 && opaque->getType() == i32 &&
                  opaque->getArgOperand(0)->getType() == i32)
                continue;
        llvm::IRBuilder<> builder(call);
        builder.SetCurrentDebugLocation(llvm::DebugLoc());
        auto *address =
            builder.CreatePtrToInt(ptr, i32, "lds.dma.alias.address");
        auto *opaque =
            builder.CreateCall(identity, {address}, "lds.dma.alias.opaque");
        call->setArgOperand(
            1, builder.CreateIntToPtr(opaque, ptrType, "lds.dma.alias.ptr"));
        ++changed;
      }
  }
  return changed;
}

namespace mlir::triton::HCU {
unsigned preserveLdsBarrierOrder(llvm::Module *module,
                                 const std::string &arch) {
  return ::preserveLdsBarrierOrder(module, arch);
}
llvm::Expected<unsigned>
stripSynchronousBarrierMarkers(llvm::Module *module, const std::string &arch) {
  return ::stripSynchronousBarrierMarkers(module, arch);
}
llvm::Expected<unsigned> lowerTaggedLocalBarriers(llvm::Module *module,
                                                  const std::string &arch) {
  return ::lowerTaggedLocalBarriers(module, arch);
}
unsigned preserveUniformResourceIndices(llvm::Module *module,
                                        const std::string &arch) {
  return ::preserveUniformResourceIndices(module, arch);
}
unsigned preserveBufferLoadByteOffsets(llvm::Module *module,
                                       const std::string &arch) {
  return ::preserveBufferLoadByteOffsets(module, arch);
}
unsigned preferReadySharedLoads(llvm::Module *module, const std::string &arch) {
  return ::preferReadySharedLoads(module, arch);
}
unsigned annotateLdsObjectScopes(llvm::Module *module, const std::string &arch,
                                 unsigned waveSize) {
  return ::annotateLdsObjectScopes(module, arch, waveSize);
}
unsigned preserveLdsDmaAliasScopes(llvm::Module *module,
                                   const std::string &arch) {
  return ::preserveLdsDmaAliasScopes(module, arch);
}
ReadOnlyInterleaveStats interleaveGlobalLdsReads(llvm::Module *module,
                                                 const std::string &arch,
                                                 bool addFences) {
  return ::interleaveGlobalLdsReads(module, arch, addFences);
}
llvm::Expected<TupledRegisterFenceStats>
tupleRegisterFences(llvm::Module *module, const std::string &arch) {
  return ::tupleRegisterFences(module, arch);
}
llvm::Expected<NativeMemoryOptimizationResult>
optimizeMemoryIR(const std::string &source, const std::string &arch,
                 bool factorStores, bool packRounds, bool combineShifts) {
  NativeMemoryOptimizationResult result;
  result.source = source;
  if (!factorStores && !packRounds && !combineShifts)
    return result;
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIR(llvm::MemoryBufferRef(source, "hcu-memory"),
                              diagnostic, context);
  if (!module)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Invalid LLVM IR for HCU memory optimizations");
  std::string errors;
  llvm::raw_string_ostream errorStream(errors);
  if (llvm::verifyModule(*module, &errorStream))
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Input LLVM verification failed: " + errors);
  if (factorStores)
    result.factoredStores =
        flagtree::hcu::factorBufferStoreOffsets(*module, arch);
  if (packRounds)
    result.packedBf16Rounds = flagtree::hcu::packBf16Rounds(*module);
  if (combineShifts) {
    auto stats = flagtree::hcu::combineShiftedDisjointOr(*module, arch);
    result.combinedShifts = stats.combined;
    result.shiftedReason = stats.reason;
  }
  if (!result.factoredStores && !result.packedBf16Rounds &&
      !result.combinedShifts)
    return result;
  errors.clear();
  if (llvm::verifyModule(*module, &errorStream))
    return llvm::createStringError(
        std::make_error_code(std::errc::state_not_recoverable),
        "HCU memory optimization verification failed: " + errors);
  result.source.clear();
  llvm::raw_string_ostream stream(result.source);
  module->print(stream, nullptr);
  return result;
}
} // namespace mlir::triton::HCU
