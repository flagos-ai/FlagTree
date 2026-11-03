#ifndef TRITON_HCU_LOOP_CARRIED_STORE_REFILL_H
#define TRITON_HCU_LOOP_CARRIED_STORE_REFILL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace flagtree::hcu {

struct LoopCarriedStoreRefillStats {
  unsigned functionsVisited = 0;
  unsigned functionsChanged = 0;
  unsigned loops = 0; // Committed loops; binding-friendly short spelling.
  unsigned loopsChanged = 0;
  unsigned pairs = 0;
  unsigned movedLoads = 0;
  unsigned movedPureDependencies = 0;
  unsigned schedulerBarriers = 0;
  unsigned rejectedFunctions = 0;
  unsigned invalidFunctions = 0;
  std::string reason = "no eligible closed loop-carried store/refill plan";
};

namespace loop_carried_refill_detail {

// A late LLVM scheduling transform, not a software-pipeline construction.
// No instruction is duplicated or deleted. Only already-executing readonly
// global loads and explicit pure dependencies move earlier in the SAME BB.
// Stores, LDS reads, MMACs, existing waits/barriers and CFG keep their order.
enum class Effect { Pure, Phi, Query, RawRead, LDS, Wave, LGKM, MMAC, Flow };

struct OriginBit {
  const llvm::CallInst *load = nullptr;
  unsigned bit = 0;
  unsigned backedges = 0;
};
using Bits = std::vector<OriginBit>;

struct PairPlan {
  llvm::StoreInst *store = nullptr;
  llvm::CallInst *load = nullptr;
  llvm::SmallVector<llvm::Instruction *, 16> moving;
};

class ClosedPlan {
  // Resource budgets bound proof cost, not a shape/geometry/transaction list.
  static constexpr unsigned MaxInstructions = 50000;
  static constexpr unsigned MaxDepth = 512;
  static constexpr unsigned MaxValueBits = 131072;
  static constexpr unsigned MaxMemoBits = 2097152;

  llvm::Function &fn;
  const llvm::DataLayout &dl;
  llvm::DominatorTree dt;
  llvm::BasicBlock *loop = nullptr;
  llvm::BasicBlock *preheader = nullptr;
  llvm::DenseMap<llvm::Instruction *, unsigned> order;
  llvm::DenseMap<llvm::Instruction *, Effect> effects;
  llvm::SmallPtrSet<llvm::CallInst *, 32> loopLoads;
  llvm::DenseMap<llvm::Value *, Bits> bitMemo;
  llvm::SmallPtrSet<llvm::Value *, 32> bitActive;
  unsigned memoBits = 0;
  std::string error;

  bool fail(llvm::StringRef why) {
    if (error.empty())
      error = why.str();
    return false;
  }

  static unsigned addressSpace(llvm::Type *ty) {
    auto *pointer = llvm::dyn_cast<llvm::PointerType>(ty);
    return pointer ? pointer->getAddressSpace() : ~0u;
  }

  unsigned payloadWidth(llvm::Type *ty) const {
    llvm::Type *element = ty;
    uint64_t count = 1;
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(ty)) {
      count = vector->getNumElements();
      element = vector->getElementType();
    } else if (ty->isVectorTy()) {
      return 0; // No scalable or nested vectors.
    }
    if (!element->isIntegerTy() && !element->isFloatingPointTy())
      return 0;
    auto scalarBits = element->getPrimitiveSizeInBits();
    if (scalarBits.isScalable())
      return 0;
    uint64_t width = count * scalarBits.getFixedValue();
    if (!width || width > MaxValueBits || width % 8)
      return 0;
    // Do not confuse an i1-vector's logical bits with its memory footprint.
    auto storage = dl.getTypeStoreSize(ty);
    if (storage.isScalable() || storage.getFixedValue() * 8 != width)
      return 0;
    return static_cast<unsigned>(width);
  }

  static bool pureSSA(llvm::Instruction *inst) {
    // Explicitly exclude division/remainder, pointer/integer/address-space
    // casts, alloca, arbitrary memory(none) calls, EH and constrained FP.
    switch (inst->getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor:
    case llvm::Instruction::Shl:
    case llvm::Instruction::LShr:
    case llvm::Instruction::AShr:
    case llvm::Instruction::FAdd:
    case llvm::Instruction::FSub:
    case llvm::Instruction::FMul:
    case llvm::Instruction::FNeg:
    case llvm::Instruction::ICmp:
    case llvm::Instruction::FCmp:
    case llvm::Instruction::Select:
    case llvm::Instruction::ZExt:
    case llvm::Instruction::SExt:
    case llvm::Instruction::Trunc:
    case llvm::Instruction::BitCast:
    case llvm::Instruction::FPExt:
    case llvm::Instruction::FPTrunc:
    case llvm::Instruction::SIToFP:
    case llvm::Instruction::UIToFP:
    case llvm::Instruction::FPToSI:
    case llvm::Instruction::FPToUI:
    case llvm::Instruction::GetElementPtr:
    case llvm::Instruction::ExtractElement:
    case llvm::Instruction::InsertElement:
    case llvm::Instruction::ShuffleVector:
    case llvm::Instruction::ExtractValue:
    case llvm::Instruction::InsertValue:
    case llvm::Instruction::Freeze:
      return !inst->mayReadOrWriteMemory() && !inst->mayHaveSideEffects();
    default:
      return false;
    }
  }

  static llvm::Function *directDeclaration(llvm::CallInst *call) {
    auto *callee = call->getCalledFunction();
    if (!callee || !callee->isDeclaration() || call->isInlineAsm() ||
        call->getNumOperandBundles() || call->isMustTailCall() ||
        !call->doesNotThrow() ||
        !callee->hasFnAttribute(llvm::Attribute::WillReturn))
      return nullptr;
    return callee;
  }

  bool readonlyGlobalPointer(llvm::Value *value, unsigned depth = 0) {
    if (depth >= MaxDepth || addressSpace(value->getType()) != 1)
      return fail("resource base has no bounded AS1 readonly provenance");
    if (auto *argument = llvm::dyn_cast<llvm::Argument>(value))
      return (argument->getParent() == &fn &&
              argument->hasAttribute(llvm::Attribute::ReadOnly)) ||
             fail("resource does not root in a readonly AS1 kernel argument");
    auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value);
    if (!gep || addressSpace(gep->getPointerOperand()->getType()) != 1)
      return fail("resource pointer cast/load/PHI/unknown root");
    return readonlyGlobalPointer(gep->getPointerOperand(), depth + 1);
  }

  bool makeResource(llvm::CallInst *call) {
    auto *callee = directDeclaration(call);
    if (!callee || !callee->isIntrinsic() ||
        !callee->getName().starts_with("llvm.amdgcn.make.buffer.rsrc.") ||
        !call->doesNotAccessMemory() || call->arg_size() != 4 ||
        addressSpace(call->getType()) != 8 ||
        addressSpace(call->getArgOperand(0)->getType()) != 1 ||
        !call->getArgOperand(1)->getType()->isIntegerTy(16) ||
        !call->getArgOperand(2)->getType()->isIntegerTy(32) ||
        !call->getArgOperand(3)->getType()->isIntegerTy(32))
      return fail("unknown resource construction/effects/type");
    auto *stride = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
    if (!stride || !stride->isZero())
      return fail("nonraw resource stride");
    return readonlyGlobalPointer(call->getArgOperand(0));
  }

  bool rawRead(llvm::CallInst *call) {
    auto *callee = directDeclaration(call);
    if (!callee ||
        callee->getIntrinsicID() !=
            llvm::Intrinsic::amdgcn_raw_ptr_buffer_load ||
        !call->onlyReadsMemory() || call->arg_size() != 4 ||
        addressSpace(call->getArgOperand(0)->getType()) != 8 ||
        !payloadWidth(call->getType()))
      return fail("unknown/nonordinary raw-buffer read");
    for (unsigned i = 1; i != 4; ++i)
      if (!call->getArgOperand(i)->getType()->isIntegerTy(32))
        return fail("raw-buffer offset/cache type");
    auto *aux = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(3));
    if (!aux || !aux->isZero())
      return fail("raw-buffer volatile/cache auxiliary domain not zero");
    auto *resource = llvm::dyn_cast<llvm::CallInst>(call->getArgOperand(0));
    return resource && makeResource(resource);
  }

  bool classify(llvm::Instruction *inst, Effect &effect) {
    if (pureSSA(inst)) {
      effect = Effect::Pure;
      return true;
    }
    if (llvm::isa<llvm::PHINode>(inst)) {
      effect = Effect::Phi;
      return true;
    }
    if (llvm::isa<llvm::BranchInst, llvm::ReturnInst,
                  llvm::UnreachableInst>(inst)) {
      effect = Effect::Flow;
      return true;
    }
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(inst)) {
      if (load->isVolatile() || load->isAtomic() ||
          load->getPointerAddressSpace() != 3)
        return fail("non-AS3/atomic/volatile ordinary load");
      effect = Effect::LDS;
      return true;
    }
    if (auto *store = llvm::dyn_cast<llvm::StoreInst>(inst)) {
      // Check the ACTUAL destination, not the stored value's pointer type.
      if (store->isVolatile() || store->isAtomic() ||
          store->getPointerAddressSpace() != 3)
        return fail("global/unknown/atomic/volatile publisher");
      effect = Effect::LDS;
      return true;
    }
    auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
    auto *callee = call ? directDeclaration(call) : nullptr;
    if (!callee)
      return fail("unknown effect/call/asm/fence/terminator");
    auto name = callee->getName();
    if (callee->getIntrinsicID() ==
        llvm::Intrinsic::amdgcn_raw_ptr_buffer_load) {
      if (!rawRead(call))
        return false;
      effect = Effect::RawRead;
      return true;
    }
    if (name.starts_with("llvm.amdgcn.make.buffer.rsrc.")) {
      if (!makeResource(call))
        return false;
      effect = Effect::Pure;
      return true;
    }
    if (name == "llvm.amdgcn.workgroup.id.x" ||
        name == "llvm.amdgcn.workgroup.id.y" ||
        name == "llvm.amdgcn.workgroup.id.z" ||
        name == "llvm.amdgcn.workitem.id.x" ||
        name == "llvm.amdgcn.workitem.id.y" ||
        name == "llvm.amdgcn.workitem.id.z") {
      if (!callee->isIntrinsic() || !call->arg_empty() ||
          !call->doesNotAccessMemory() || !call->getType()->isIntegerTy(32))
        return fail("unknown ID query");
      effect = Effect::Query; // May be crossed, never newly moved.
      return true;
    }
    if (name == "llvm.amdgcn.wave.barrier") {
      if (!callee->isIntrinsic() || !call->arg_empty() ||
          !call->getType()->isVoidTy())
        return fail("unknown wave barrier signature");
      effect = Effect::Wave;
      return true;
    }
    if (name == "llvm.amdgcn.s.waitcnt") {
      auto *imm = call->arg_size() == 1
                      ? llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0))
                      : nullptr;
      if (!callee->isIntrinsic() || !call->getType()->isVoidTy() || !imm ||
          !imm->getType()->isIntegerTy(32) || imm->getValue().ugt(65535))
        return fail("unknown wait encoding");
      uint64_t n = imm->getZExtValue();
      if ((n & 0xc00f) != 0xc00f || (n & 0x70) != 0x70)
        return fail("VM/EXP-constraining wait");
      effect = Effect::LGKM;
      return true;
    }
    if (name.starts_with("llvm.hcu.mmac.") &&
        call->doesNotAccessMemory()) {
      effect = Effect::MMAC;
      return true;
    }
    // CTA barriers, memory-clobber asm, scheduling/IGLP intrinsics, all
    // stores/atomics/fences/unknown calls remain closed-proof rejections.
    return fail("unknown call/publisher/barrier (not a wave/LGKM boundary)");
  }

  const Bits *traceBits(llvm::Value *value, unsigned depth = 0) {
    unsigned width = payloadWidth(value->getType());
    if (!width || depth >= MaxDepth) {
      fail("payload type/depth proof budget");
      return nullptr;
    }
    if (auto it = bitMemo.find(value); it != bitMemo.end())
      return &it->second;
    if (!bitActive.insert(value).second) {
      fail("cyclic payload chain");
      return nullptr;
    }
    Bits result;
    if (llvm::isa<llvm::PoisonValue, llvm::UndefValue>(value)) {
      result.resize(width); // Unknown may be overwritten, never stored.
    } else if (auto *call = llvm::dyn_cast<llvm::CallInst>(value);
               call && loopLoads.contains(call)) {
      for (unsigned b = 0; b != width; ++b)
        result.push_back({call, b, 0});
    } else if (auto *phi = llvm::dyn_cast<llvm::PHINode>(value)) {
      int edge = phi->getBasicBlockIndex(loop);
      if (phi->getParent() != loop || phi->getNumIncomingValues() != 2 ||
          edge < 0 || phi->getBasicBlockIndex(preheader) < 0) {
        fail("payload PHI is not the unique loop-carried edge");
        return nullptr;
      }
      auto *incoming = traceBits(phi->getIncomingValue(edge), depth + 1);
      if (!incoming)
        return nullptr;
      result = *incoming;
      for (auto &origin : result) {
        if (origin.load && origin.backedges != 0) {
          fail("payload crosses more than one iteration");
          return nullptr;
        }
        if (origin.load)
          ++origin.backedges;
      }
    } else if (auto *cast = llvm::dyn_cast<llvm::BitCastInst>(value)) {
      auto *source = traceBits(cast->getOperand(0), depth + 1);
      if (!source || source->size() != width)
        return nullptr;
      result = *source; // DataLayout is little-endian; no bit reordering.
    } else if (auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(value)) {
      auto *ty = llvm::dyn_cast<llvm::FixedVectorType>(shuffle->getType());
      auto *inputTy = llvm::dyn_cast<llvm::FixedVectorType>(
          shuffle->getOperand(0)->getType());
      if (!ty || !inputTy ||
          shuffle->getOperand(1)->getType() != inputTy) {
        fail("nonfixed shuffle payload");
        return nullptr;
      }
      unsigned elementBits = width / ty->getNumElements();
      // Copy before tracing B: DenseMap insertions may invalidate references.
      auto *ap = traceBits(shuffle->getOperand(0), depth + 1);
      if (!ap)
        return nullptr;
      Bits a = *ap;
      auto *bp = traceBits(shuffle->getOperand(1), depth + 1);
      if (!bp)
        return nullptr;
      a.insert(a.end(), bp->begin(), bp->end());
      for (int index : shuffle->getShuffleMask()) {
        if (index < 0) {
          result.resize(result.size() + elementBits);
        } else {
          uint64_t begin = uint64_t(index) * elementBits;
          if (begin + elementBits > a.size()) {
            fail("out-of-range shuffle payload");
            return nullptr;
          }
          result.insert(result.end(), a.begin() + begin,
                        a.begin() + begin + elementBits);
        }
      }
    } else if (auto *extract = llvm::dyn_cast<llvm::ExtractElementInst>(value)) {
      auto *ty = llvm::dyn_cast<llvm::FixedVectorType>(
          extract->getVectorOperand()->getType());
      auto *index = llvm::dyn_cast<llvm::ConstantInt>(extract->getIndexOperand());
      if (!ty || !index || index->getValue().uge(ty->getNumElements())) {
        fail("dynamic/out-of-range payload extract");
        return nullptr;
      }
      auto *vector = traceBits(extract->getVectorOperand(), depth + 1);
      if (!vector)
        return nullptr;
      uint64_t begin = index->getZExtValue() * uint64_t(width);
      result.assign(vector->begin() + begin, vector->begin() + begin + width);
    } else if (auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(value)) {
      auto *ty = llvm::dyn_cast<llvm::FixedVectorType>(insert->getType());
      auto *index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
      if (!ty || !index || index->getValue().uge(ty->getNumElements())) {
        fail("dynamic/out-of-range payload insert");
        return nullptr;
      }
      auto *vector = traceBits(insert->getOperand(0), depth + 1);
      if (!vector)
        return nullptr;
      result = *vector;
      auto *scalar = traceBits(insert->getOperand(1), depth + 1);
      if (!scalar)
        return nullptr;
      uint64_t begin = index->getZExtValue() * uint64_t(scalar->size());
      std::copy(scalar->begin(), scalar->end(), result.begin() + begin);
    } else {
      fail("unsupported/constant/computed payload bit source");
      return nullptr;
    }
    if (result.size() != width || width > MaxMemoBits - memoBits) {
      fail("payload size/memo proof budget");
      return nullptr;
    }
    memoBits += width;
    bitActive.erase(value);
    return &bitMemo.try_emplace(value, std::move(result)).first->second;
  }

  bool dependencies(llvm::Instruction *inst, PairPlan &pair,
                    const llvm::SmallPtrSetImpl<llvm::Instruction *> &scheduled,
                    llvm::SmallPtrSetImpl<llvm::Instruction *> &visiting,
                    llvm::SmallPtrSetImpl<llvm::Instruction *> &done,
                    unsigned depth = 0) {
    if (depth >= MaxDepth)
      return fail("dependency depth proof budget");
    if (inst->getParent() != loop)
      return dt.dominates(inst->getParent(), loop) ||
             fail("external dependency does not dominate loop");
    if (llvm::isa<llvm::PHINode>(inst) ||
        order.lookup(inst) < order.lookup(pair.store) ||
        scheduled.contains(inst) || done.contains(inst))
      return true;
    if ((inst != pair.load &&
         order.lookup(inst) >= order.lookup(pair.load)) ||
        !visiting.insert(inst).second)
      return fail("forward/cyclic moved dependency");
    if (inst != pair.load && effects.lookup(inst) != Effect::Pure)
      return fail("moving dependency is not explicit pure SSA/resource");
    for (auto &operand : inst->operands())
      if (auto *definition = llvm::dyn_cast<llvm::Instruction>(operand.get()))
        if (!dependencies(definition, pair, scheduled, visiting, done,
                          depth + 1))
          return false;
    visiting.erase(inst);
    done.insert(inst);
    pair.moving.push_back(inst);
    return true;
  }

public:
  llvm::SmallVector<llvm::Instruction *, 128> original;
  llvm::SmallVector<PairPlan, 16> pairs;
  llvm::Instruction *firstDrainAnchor = nullptr;

  explicit ClosedPlan(llvm::Function &function)
      : fn(function), dl(fn.getParent()->getDataLayout()), dt(fn) {}

  llvm::StringRef failure() const { return error; }
  llvm::BasicBlock *body() const { return loop; }

  bool prove() {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL ||
        fn.getParent()->getDataLayoutStr().empty() || !dl.isLittleEndian() ||
        fn.hasFnAttribute(llvm::Attribute::StrictFP))
      return fail("requires little-endian non-strictfp AMDGPU kernel");
    unsigned instructionCount = 0;
    for (auto &bb : fn) {
      if (!dt.isReachableFromEntry(&bb))
        return fail("unreachable basic block");
      instructionCount += bb.size();
    }
    if (instructionCount > MaxInstructions)
      return fail("instruction proof budget");
    llvm::LoopInfo loops(dt);
    if (loops.empty() || std::next(loops.begin()) != loops.end())
      return fail("requires one unique natural loop");
    auto *natural = *loops.begin();
    if (!natural->getSubLoops().empty() || natural->getNumBlocks() != 1)
      return fail("requires one basic-block self-edge loop");
    loop = natural->getHeader();
    auto *branch = llvm::dyn_cast<llvm::BranchInst>(loop->getTerminator());
    if (!branch || !branch->isConditional() ||
        branch->getSuccessor(0) == branch->getSuccessor(1) ||
        (branch->getSuccessor(0) != loop && branch->getSuccessor(1) != loop))
      return fail("nonclosed self-loop terminator");
    unsigned predecessors = 0;
    for (auto *pred : llvm::predecessors(loop)) {
      ++predecessors;
      if (pred != loop) {
        if (preheader)
          return fail("nonunique loop preheader");
        preheader = pred;
      }
    }
    auto *preBranch = preheader
                          ? llvm::dyn_cast<llvm::BranchInst>(preheader->getTerminator())
                          : nullptr;
    if (predecessors != 2 || !preBranch || !preBranch->isUnconditional() ||
        preBranch->getSuccessor(0) != loop)
      return fail("requires unconditional unique loop preheader");

    // Across a wave barrier the AS1 publication proof is FUNCTION provenance,
    // not an imported metadata assertion. Scan EVERY entry-to-loop ancestor.
    llvm::SmallPtrSet<llvm::BasicBlock *, 32> ancestors;
    llvm::SmallVector<llvm::BasicBlock *, 16> work{loop};
    while (!work.empty()) {
      auto *bb = work.pop_back_val();
      if (!ancestors.insert(bb).second)
        continue;
      for (auto *pred : llvm::predecessors(bb))
        if (pred != bb)
          work.push_back(pred);
    }
    // The set is only membership: effects/plan order is always actual IR order.
    for (auto &bb : fn)
      if (ancestors.contains(&bb))
        for (auto &inst : bb) {
          Effect effect;
          if (!classify(&inst, effect))
            return false;
          effects[&inst] = effect;
        }
    llvm::SmallVector<llvm::StoreInst *, 16> stores;
    for (auto &inst : *loop) {
      order[&inst] = original.size();
      original.push_back(&inst);
      if (effects.lookup(&inst) == Effect::RawRead)
        loopLoads.insert(llvm::cast<llvm::CallInst>(&inst));
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
        stores.push_back(store);
    }
    if (stores.empty() || stores.size() != loopLoads.size())
      return fail("loop raw-load/AS3-store transaction counts do not pair");
    llvm::SmallPtrSet<llvm::CallInst *, 32> used;
    for (auto *store : stores) {
      if (!llvm::isa<llvm::FixedVectorType>(store->getValueOperand()->getType()))
        return fail("requires fixed-vector AS3 store payload");
      auto *mapping = traceBits(store->getValueOperand());
      if (!mapping || mapping->empty())
        return false;
      auto *load = const_cast<llvm::CallInst *>(mapping->front().load);
      if (!load || !loopLoads.contains(load) || !used.insert(load).second ||
          order.lookup(load) <= order.lookup(store) ||
          payloadWidth(load->getType()) != mapping->size())
        return fail("refill pair not unique/forward/full-width");
      for (unsigned b = 0; b != mapping->size(); ++b)
        if ((*mapping)[b].load != load || (*mapping)[b].bit != b ||
            (*mapping)[b].backedges != 1)
          return fail("store is not one complete ordered previous-iteration load");
      PairPlan pair;
      pair.store = store;
      pair.load = load;
      pairs.push_back(std::move(pair));
    }
    if (used.size() != loopLoads.size())
      return fail("unpaired raw-buffer load");

    llvm::SmallPtrSet<llvm::Instruction *, 32> scheduled;
    for (auto &pair : pairs) {
      llvm::SmallPtrSet<llvm::Instruction *, 32> visiting, done;
      if (!dependencies(pair.load, pair, scheduled, visiting, done))
        return false;
      for (unsigned n = order.lookup(pair.store) + 1;
           n < order.lookup(pair.load); ++n) {
        auto *crossed = original[n];
        if (scheduled.contains(crossed))
          continue;
        auto effect = effects.lookup(crossed);
        if (effect != Effect::Pure && effect != Effect::MMAC &&
            effect != Effect::LDS && effect != Effect::Wave &&
            effect != Effect::LGKM && effect != Effect::RawRead)
          return fail("crossed nonclosed effect/control/query");
      }
      for (auto *inst : pair.moving)
        if (!scheduled.insert(inst).second)
          return fail("dependency scheduled twice");
    }

    // Verify the COMPLETE virtual order before touching IR; includes uses of
    // moved address dependencies by unchanged instructions and PHI edge uses.
    llvm::DenseMap<llvm::Instruction *, PairPlan *> after;
    for (auto &pair : pairs)
      after[pair.store] = &pair;
    llvm::SmallVector<llvm::Instruction *, 128> proposed;
    for (auto *inst : original)
      if (!scheduled.contains(inst)) {
        proposed.push_back(inst);
        if (auto it = after.find(inst); it != after.end())
          proposed.append(it->second->moving.begin(), it->second->moving.end());
      }
    if (proposed.size() != original.size())
      return fail("instruction bijection failed");
    llvm::DenseMap<llvm::Instruction *, unsigned> position;
    for (unsigned n = 0; n != proposed.size(); ++n)
      if (!position.try_emplace(proposed[n], n).second)
        return fail("instruction duplicated in virtual order");
    for (auto *inst : proposed) {
      if (auto *phi = llvm::dyn_cast<llvm::PHINode>(inst)) {
        for (unsigned n = 0; n != phi->getNumIncomingValues(); ++n)
          if (auto *def = llvm::dyn_cast<llvm::Instruction>(phi->getIncomingValue(n))) {
            auto *edge = phi->getIncomingBlock(n);
            if (!dt.dominates(def->getParent(), edge) ||
                (def->getParent() == loop &&
                 (edge != loop || position.lookup(def) >= position.lookup(branch))))
              return fail("virtual PHI-edge SSA dominance");
          }
        continue;
      }
      for (auto &operand : inst->operands())
        if (auto *def = llvm::dyn_cast<llvm::Instruction>(operand.get())) {
          if (def->getParent() == loop) {
            if (position.lookup(def) >= position.lookup(inst))
              return fail("virtual same-block SSA dominance");
          } else if (!dt.dominates(def->getParent(), loop)) {
            return fail("virtual outside-block SSA dominance");
          }
        }
    }
    // Every instruction stays in its original BB. Users in other BBs retain
    // block dominance; the loop terminator remains last for all outgoing PHIs.
    firstDrainAnchor = stores.front();
    for (unsigned n = order.lookup(stores.front()); n != 0;) {
      auto *inst = original[--n];
      auto effect = effects.lookup(inst);
      if (effect == Effect::Pure)
        continue;
      if (effect == Effect::Wave || effect == Effect::LGKM) {
        firstDrainAnchor = inst;
        continue;
      }
      break; // In particular stop AFTER the previous MMAC, never before it.
    }
    return true;
  }
};

inline LoopCarriedStoreRefillStats run(llvm::Function &fn,
                                      llvm::StringRef arch) {
  LoopCarriedStoreRefillStats stats;
  if (arch != "gfx936") {
    stats.reason = "unsupported architecture (requires gfx936)";
    return stats;
  }
  if (fn.isDeclaration() ||
      fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
    return stats;
  stats.functionsVisited = 1;
  llvm::raw_null_ostream sink;
  if (llvm::verifyFunction(fn, &sink)) {
    stats.invalidFunctions = 1;
    stats.reason = "input LLVM function verification failed; unchanged";
    return stats;
  }
  ClosedPlan plan(fn);
  if (!plan.prove()) {
    stats.rejectedFunctions = 1;
    stats.reason = plan.failure().str();
    return stats; // No insertion, metadata write, reordering or declaration.
  }
  auto *module = fn.getParent();
  auto *oldDeclaration = module->getFunction("llvm.amdgcn.sched.barrier");
  auto *declaration = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::amdgcn_sched_barrier);
  llvm::SmallVector<llvm::Instruction *, 32> fences;
  auto addFence = [&](llvm::Instruction *before, llvm::DebugLoc location) {
    llvm::IRBuilder<> builder(before);
    builder.SetCurrentDebugLocation(location);
    auto *fence = builder.CreateCall(declaration, {builder.getInt32(0)});
    fences.push_back(fence);
    return fence;
  };
  addFence(plan.firstDrainAnchor, plan.firstDrainAnchor->getDebugLoc());
  for (auto &pair : plan.pairs) {
    // Stable fence anchors avoid iterator hazards when an original next-node
    // is itself a moving dependency. Keep one boundary on each side of load.
    auto *before = addFence(pair.store->getNextNode(), pair.store->getDebugLoc());
    auto *after = addFence(before->getNextNode(), pair.store->getDebugLoc());
    for (auto *inst : pair.moving)
      inst->moveBefore(after->getIterator()); // LLVM22 iterator API.
  }
  if (llvm::verifyFunction(fn, &sink)) {
    for (auto *fence : fences)
      fence->eraseFromParent();
    auto *terminator = plan.body()->getTerminator();
    for (auto *inst : plan.original)
      if (inst != terminator)
        inst->moveBefore(terminator->getIterator());
    if (!oldDeclaration && declaration->use_empty())
      declaration->eraseFromParent();
    stats.invalidFunctions = 1;
    stats.reason = "output verifier failed; original instruction order restored";
    return stats;
  }
  stats.functionsChanged = stats.loops = stats.loopsChanged = 1;
  stats.pairs = stats.movedLoads = plan.pairs.size();
  stats.schedulerBarriers = fences.size();
  for (auto &pair : plan.pairs)
    stats.movedPureDependencies += pair.moving.size() - 1;
  stats.reason = "closed loop-carried readonly AS1 refill after AS3 store; "
                 "original effects/MMAC/CFG retained, LLVM verifier passed";
  return stats;
}

} // namespace loop_carried_refill_detail

// Intended binding: verifyModule -> this function -> verifyModule -> normal
// llc. Architecture must be the actual backend target, never a metadata claim.
// Do not run the old text hook and this structured transform simultaneously.
inline LoopCarriedStoreRefillStats
interleaveLoopCarriedStoreRefill(llvm::Function &fn, llvm::StringRef arch) {
  return loop_carried_refill_detail::run(fn, arch);
}

inline LoopCarriedStoreRefillStats
interleaveLoopCarriedStoreRefill(llvm::Module &module, llvm::StringRef arch) {
  LoopCarriedStoreRefillStats total;
  if (arch != "gfx936") {
    total.reason = "unsupported architecture (requires gfx936)";
    return total;
  }
  for (auto &fn : module) {
    auto stats = loop_carried_refill_detail::run(fn, arch);
    total.functionsVisited += stats.functionsVisited;
    total.functionsChanged += stats.functionsChanged;
    total.loops += stats.loops;
    total.loopsChanged += stats.loopsChanged;
    total.pairs += stats.pairs;
    total.movedLoads += stats.movedLoads;
    total.movedPureDependencies += stats.movedPureDependencies;
    total.schedulerBarriers += stats.schedulerBarriers;
    total.rejectedFunctions += stats.rejectedFunctions;
    total.invalidFunctions += stats.invalidFunctions;
    if (stats.functionsVisited)
      total.reason = stats.reason;
  }
  if (total.functionsChanged)
    total.reason = "committed closed loop-carried store/refill plans; "
                   "per-function LLVM verification passed";
  return total;
}

} // namespace flagtree::hcu

#endif
