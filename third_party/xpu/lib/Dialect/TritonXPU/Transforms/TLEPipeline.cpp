//===- TLEPipeline.cpp - software-pipeline TLE GM->LM/SM copies -----------===//
//
// Overlap a TLE loop's DMA fill with its compute by rotating the destination
// buffer through the loop's iter_args. See
// third_party/xpu/docs/XPUTLEPipelineDesignDoc.md.
//
// The XPU cluster ISA has no counting wait -- `mfence` is a bulk drain per
// memory space -- so depth is capped at one transfer per staged space and the
// body order must be `tle_wait ; issue prefetch(i+1) ; compute(i)`. Issuing
// before draining is a structural no-op that still looks correct in the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <string>

#define DEBUG_TYPE "tritonxpu-tle-pipeline"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTLEPIPELINE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

using ::mlir::triton::gpu::LocalAllocOp;
using ::mlir::triton::gpu::MemDescType;

/// The memory spaces a staged half may use, in the order depth 2 alternates
/// them. The stage ceiling is DERIVED from this array: one bulk-drain channel
/// per space means depth 1 per space, hence `maxStages = spaces + 1`. Appending
/// a third space here is the only way to raise it -- never write the number.
constexpr llvm::StringLiteral kStagedSpaces[] = {"lm", "sm"};
constexpr unsigned kNumStagedSpaces =
    sizeof(kStagedSpaces) / sizeof(kStagedSpaces[0]);
constexpr unsigned kMaxStages = kNumStagedSpaces + 1;

/// mfence mask for the cluster-shared SM channel. The masks are not monotonic
/// in their bits (MB0 K5), so this is the bare bit 2, not SM|GM.
constexpr unsigned kSharedMemWaitMask = 2;

constexpr llvm::StringLiteral kPipelinedAttrName = "triton_xpu.tle_pipelined";
constexpr llvm::StringLiteral kNumStagesAttrName = "tt.num_stages";

/// Which space staged half `h` lives in. At depth 1 there is a single half and
/// it stays where the kernel put the buffer; from depth 2 on the halves
/// alternate through `kStagedSpaces` so that each one gets its own drain
/// channel.
bool slotIsShared(unsigned h, unsigned H, bool originIsShared) {
  if (H < 2)
    return originIsShared;
  return kStagedSpaces[h % kNumStagedSpaces] == "sm";
}

std::string localMemBudgetMessage(unsigned bytes, unsigned limit) {
  return (llvm::Twine("per-core local memory ") + llvm::Twine(bytes) +
          " bytes exceeds the " + llvm::Twine(limit) +
          " byte budget (min(XTDK stack limit, 8KB physical))")
      .str();
}

std::string sharedMemBudgetMessage(unsigned perCoreBytes) {
  unsigned total = perCoreBytes * kCoresPerCluster;
  return (llvm::Twine("per-cluster shared memory ") + llvm::Twine(total) +
          " bytes (" + llvm::Twine(perCoreBytes) + " per core x " +
          llvm::Twine(kCoresPerCluster) + " cores) exceeds the " +
          llvm::Twine(kSharedMemBudgetBytes) + " byte budget (" +
          llvm::Twine(kSharedMemTotalBytes) + " physical minus " +
          llvm::Twine(kSharedMemAmoReserveBytes) +
          " reserved at the top for .sh.amo)")
      .str();
}

/// `t` moved to LM or SM. Only the memory space changes -- shape, encoding and
/// allocShape have to survive verbatim or the allocator re-cuts the tile.
MemDescType memDescWithSpace(MemDescType t, bool shared) {
  MLIRContext *ctx = t.getContext();
  Attribute space =
      shared ? Attribute(xpu::SharedMemorySpaceAttr::get(ctx))
             : Attribute(triton::gpu::SharedMemorySpaceAttr::get(ctx));
  return MemDescType::get(t.getShape(), t.getElementType(), t.getEncoding(),
                          space, t.getMutableMemory(), t.getAllocShape());
}

/// A `tle_local_ptr` cloned onto a slot in a different space still carries the
/// origin's pointer address space in its result type. Fix it up, but only on an
/// actual space change: an unannotated kernel may legitimately use a generic
/// `!tt.ptr<f32>` here and rewriting that would be a gratuitous type change.
void retypeLocalPtr(Operation *orig, Operation *clone) {
  auto lp = dyn_cast<TLELocalPtrOp>(clone);
  if (!lp)
    return;
  bool wasShared = isSharedMemDesc(orig->getOperand(0).getType());
  bool nowShared = isSharedMemDesc(lp.getBuffer().getType());
  if (wasShared == nowShared)
    return;
  auto tensorTy = dyn_cast<RankedTensorType>(lp.getResult().getType());
  if (!tensorTy)
    return;
  auto ptrTy = dyn_cast<triton::PointerType>(tensorTy.getElementType());
  if (!ptrTy)
    return;
  unsigned as = nowShared ? kSharedMemAddrSpace : kLocalMemAddrSpace;
  lp.getResult().setType(RankedTensorType::get(
      tensorTy.getShape(), triton::PointerType::get(ptrTy.getPointeeType(), as),
      tensorTy.getEncoding()));
}

/// The in-body ops `root` transitively depends on, in body order, or failure if
/// the address is not a pure function of the induction variable and of values
/// defined outside the loop. Anything loop-carried disqualifies the copy: the
/// prefetch runs one iteration early, when the carried value does not exist
/// yet.
LogicalResult collectAddressSlice(ArrayRef<Value> roots, Block *body, Value iv,
                                  SmallVectorImpl<Operation *> &slice) {
  DenseSet<Operation *> inSlice;
  DenseSet<Value> seen;
  SmallVector<Value> worklist(roots.begin(), roots.end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second || v == iv)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() == body)
        return failure(); // loop-carried
      continue;           // defined outside
    }
    Operation *def = v.getDefiningOp();
    if (def->getBlock() != body)
      continue; // loop invariant, reuse as is
    if (!isMemoryEffectFree(def))
      return failure();
    inSlice.insert(def);
    worklist.append(def->getOperands().begin(), def->getOperands().end());
  }
  for (Operation &op : *body)
    if (inSlice.contains(&op))
      slice.push_back(&op);
  return success();
}

/// True if anything other than the staged fill writes `buf`, or if its contents
/// leave through a copy-out. Rotating a buffer that is also a destination would
/// silently send the write to the wrong half.
bool bufferIsWrittenOrCopiedOut(Value buf) {
  for (Operation *user : buf.getUsers()) {
    if (isa<TLENormCopyGlobalToLocalOp, TLECopyGlobalToLocalOp>(user))
      continue; // the fill itself
    if (auto lp = dyn_cast<TLELocalPtrOp>(user)) {
      for (Operation *u : lp.getResult().getUsers())
        if (!isa<triton::LoadOp>(u))
          return true;
      continue;
    }
    if (isa<TLEVLoadOp, triton::gpu::LocalLoadOp>(user))
      continue;
    return true;
  }
  return false;
}

/// The GM address inputs of a staged fill: the pointer tensor for a gather
/// (`tle_normcopy_g2l`) or the descriptor offsets for a TMA copy
/// (`tle_copy_g2l`). These are what must be a pure function of the induction
/// variable and of loop invariants.
SmallVector<Value> copyAddressRoots(Operation *op) {
  if (auto nc = dyn_cast<TLENormCopyGlobalToLocalOp>(op)) {
    SmallVector<Value> roots;
    roots.push_back(nc.getSrcPtrs());
    return roots;
  }
  auto cc = cast<TLECopyGlobalToLocalOp>(op);
  return {cc.getOffsets().begin(), cc.getOffsets().end()};
}

/// Operand index of the destination buffer. The gather copy keeps it fixed at
/// 1; the TMA copy has the variadic `offsets` in front of it.
unsigned copyDstOperandIdx(Operation *op) {
  if (isa<TLENormCopyGlobalToLocalOp>(op))
    return 1;
  return 1 + cast<TLECopyGlobalToLocalOp>(op).getOffsets().size();
}

Value copyDstBuffer(Operation *op) {
  if (auto nc = dyn_cast<TLENormCopyGlobalToLocalOp>(op))
    return nc.getDstBuffer();
  return cast<TLECopyGlobalToLocalOp>(op).getDstBuffer();
}

void copySetIsSync(Operation *op, bool value) {
  if (auto nc = dyn_cast<TLENormCopyGlobalToLocalOp>(op))
    nc.setIsSync(value);
  else
    cast<TLECopyGlobalToLocalOp>(op).setIsSync(value);
}

/// One GM->LM/SM fill selected for staging, plus everything needed to replay it
/// an iteration early.
struct Candidate {
  Operation *copyOp;
  LocalAllocOp allocOp;
  bool originIsShared = false;
  unsigned perCoreBytes = 0;
  /// Address computation inside the body, in body order.
  SmallVector<Operation *> addressSlice;
  /// `tle_local_ptr` ops on this buffer defined outside the loop. They have to
  /// be rematerialized per half, since each half reads a different slot (C8).
  SmallVector<Operation *> outerPtrOps;
  /// `2H` buffer handles: `[cur_0..cur_{H-1}, nxt_0..nxt_{H-1}]`.
  SmallVector<Value> slots;
};

} // namespace

struct TritonXPUTLEPipelinePass
    : public impl::TritonXPUTLEPipelineBase<TritonXPUTLEPipelinePass> {
  using impl::TritonXPUTLEPipelineBase<
      TritonXPUTLEPipelinePass>::TritonXPUTLEPipelineBase;

  TritonXPUTLEPipelinePass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    for (auto func : mod.getOps<triton::FuncOp>())
      runOnFunc(func, groupSize);
  }

  /// Baseline footprint of `func`, split by space. LM is replicated per core so
  /// it is charged per core; an SM region is a single cluster-wide block, so it
  /// is charged 64x. Mixing the two units up mis-budgets by 64x.
  void collectBaseFootprint(triton::FuncOp func, unsigned groupSize,
                            unsigned &lmBytes, unsigned &smPerCoreBytes) {
    lmBytes = 0;
    smPerCoreBytes = 0;
    func.walk([&](LocalAllocOp alloc) {
      unsigned bytes = getPerCoreBytes(alloc, groupSize);
      if (isSharedMemDesc(alloc.getType()))
        smPerCoreBytes += bytes;
      else
        lmBytes += bytes;
    });
  }

  void runOnFunc(triton::FuncOp func, unsigned groupSize) {
    // Baseline footprint is computed once here and threaded into fitBudget;
    // recomputing it per loop is a full-function walk on every iteration.
    unsigned lmBase = 0, smBase = 0;
    collectBaseFootprint(func, groupSize, lmBase, smBase);

    // Budget checks run even with `enable=false`: this is the XRE/allocator's
    // hard limit on the kernel as written, not a pipeline artifact. A kernel
    // whose baseline does not fit cannot be rescued by staging (the rotated
    // slots only ADD memory), so fail loudly regardless of the pass being off.
    if (lmBase > lmBytesLimit) {
      func.emitError() << localMemBudgetMessage(lmBase, lmBytesLimit);
      signalPassFailure();
      return;
    }
    if (smBase * kCoresPerCluster > kSharedMemBudgetBytes) {
      func.emitError() << sharedMemBudgetMessage(smBase);
      signalPassFailure();
      return;
    }

    if (!enable)
      return;

    // Post-order: inner loops first, so an outer loop still sees valid bodies.
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      pipelineLoop(forOp, groupSize, lmBase, smBase);
  }

  void pipelineLoop(scf::ForOp forOp, unsigned groupSize, unsigned lmBase,
                    unsigned smBase) {
    if (forOp->hasAttr(kPipelinedAttrName))
      return;

    unsigned stages = numStages;
    if (auto attr = forOp->getAttrOfType<IntegerAttr>(kNumStagesAttrName))
      stages = attr.getInt();
    if (stages < 2)
      return; // pipelining off for this loop, and that is not worth a remark

    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!step || *step <= 0) {
      forOp.emitRemark() << "loop step is not a positive constant";
      return;
    }

    Block *body = forOp.getBody();
    bool nested = false, hasRaw = false;
    body->walk([&](Operation *op) {
      if (isa<LoopLikeOpInterface>(op))
        nested = true;
      if (isa<RawOp>(op))
        hasRaw = true;
    });
    if (nested) {
      forOp.emitRemark() << "loop body contains a nested loop";
      return;
    }
    if (hasRaw) {
      forOp.emitRemark() << "loop body contains triton_xpu.raw, whose memory "
                            "effects are unknown (C21)";
      return;
    }

    if (stages > kMaxStages) {
      forOp.emitRemark() << "num_stages " << stages << " exceeds the "
                         << kMaxStages
                         << " stages this pass can honour: mfence is a bulk "
                            "drain, so each of the "
                         << kNumStagedSpaces
                         << " staged memory space(s) supports depth 1; "
                            "clamping to "
                         << kMaxStages;
      stages = kMaxStages;
    }

    SmallVector<Candidate> cands;
    if (failed(collectCandidates(forOp, groupSize, cands)))
      return;

    unsigned H = stages - 1;
    if (H >= 2 && llvm::any_of(cands, [](const Candidate &c) {
          return c.originIsShared;
        })) {
      forOp.emitRemark()
          << "depth 2 alternates memory spaces, so every staged "
             "buffer must start in lm; this loop stages one from "
             "sm, falling back to depth 1";
      H = 1;
    }

    if (failed(fitBudget(forOp, cands, H, lmBase, smBase)))
      return;

    rewriteLoop(forOp, cands, H, *step);
    LLVM_DEBUG(llvm::dbgs()
               << "[tle-pipeline] pipelined loop @" << forOp << ": H=" << H
               << ", staged=" << cands.size() << "\n");
  }

  /// Every GM->LM/SM fill in `forOp` that can be moved an iteration early, with
  /// a remark on each one that cannot. Fails only when nothing is stageable.
  LogicalResult collectCandidates(scf::ForOp forOp, unsigned groupSize,
                                  SmallVectorImpl<Candidate> &cands) {
    Block *body = forOp.getBody();
    Value iv = forOp.getInductionVar();

    auto isStagedFill = [](Operation *op) {
      return isa<TLENormCopyGlobalToLocalOp, TLECopyGlobalToLocalOp>(op);
    };

    DenseMap<Value, unsigned> fillCount;
    body->walk([&](Operation *op) {
      if (isStagedFill(op))
        ++fillCount[copyDstBuffer(op)];
    });

    body->walk([&](Operation *op) {
      if (!isStagedFill(op))
        return;
      if (op->getBlock() != body) {
        op->emitRemark() << "copy is nested inside a region (predicated copies "
                            "are not staged)";
        return;
      }
      Value dst = copyDstBuffer(op);
      auto alloc = dst.getDefiningOp<LocalAllocOp>();
      if (!alloc || forOp->isAncestor(alloc)) {
        op->emitRemark()
            << "destination buffer is not a ttg.local_alloc outside the loop";
        return;
      }
      if (fillCount.lookup(dst) > 1) {
        op->emitRemark()
            << "destination buffer is filled more than once in the loop body";
        return;
      }
      if (bufferIsWrittenOrCopiedOut(dst)) {
        op->emitRemark() << "destination buffer is also written or copied out";
        return;
      }
      Candidate c;
      if (failed(collectAddressSlice(copyAddressRoots(op), body, iv,
                                     c.addressSlice))) {
        op->emitRemark() << "copy address does not depend only on the "
                            "induction variable and loop invariants";
        return;
      }
      for (Operation *user : dst.getUsers()) {
        if (user == op || forOp->isAncestor(user))
          continue;
        auto lp = dyn_cast<TLELocalPtrOp>(user);
        if (!lp || llvm::any_of(lp.getResult().getUsers(), [&](Operation *u) {
              return !forOp->isAncestor(u);
            })) {
          op->emitRemark() << "destination buffer is used outside the loop";
          return;
        }
        c.outerPtrOps.push_back(lp);
      }
      c.copyOp = op;
      c.allocOp = alloc;
      c.originIsShared = isSharedMemDesc(alloc.getType());
      c.perCoreBytes = getPerCoreBytes(alloc, groupSize);
      cands.push_back(std::move(c));
    });

    if (cands.empty()) {
      forOp.emitRemark() << "no stageable GM->LM copy in the loop";
      return failure();
    }
    return success();
  }

  /// Shrink `H` until the staged slots fit both budgets. Fails when even depth
  /// 1 does not fit, in which case the loop is left alone. `lmBase`/`smBase`
  /// are the function's baseline footprint, computed once in runOnFunc.
  LogicalResult fitBudget(scf::ForOp forOp, ArrayRef<Candidate> cands,
                          unsigned &H, unsigned lmBase, unsigned smBase) {
    while (true) {
      unsigned lm = lmBase, sm = smBase;
      for (const Candidate &c : cands) {
        (c.originIsShared ? sm : lm) -= c.perCoreBytes;
        for (unsigned k = 0, e = 2 * H; k < e; ++k)
          (slotIsShared(k % H, H, c.originIsShared) ? sm : lm) +=
              c.perCoreBytes;
      }
      if (lm <= lmBytesLimit && sm * kCoresPerCluster <= kSharedMemBudgetBytes)
        return success();

      bool canDowngrade = H > 1;
      std::string msg = lm > lmBytesLimit
                            ? localMemBudgetMessage(lm, lmBytesLimit)
                            : sharedMemBudgetMessage(sm);
      msg += " after double buffering";
      if (canDowngrade)
        msg += "; falling back to depth " + std::to_string(H - 1);
      forOp.emitRemark() << msg;
      if (!canDowngrade)
        return failure();
      --H;
    }
  }

  void rewriteLoop(scf::ForOp forOp, SmallVectorImpl<Candidate> &cands,
                   unsigned H, int64_t stepConst) {
    MLIRContext *ctx = forOp.getContext();
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound();
    Value origStep = forOp.getStep();
    Value origIV = forOp.getInductionVar();
    Type ivType = origIV.getType();
    // The prologue's arith::AddIOp and the scaled-step arith::constant below
    // both require an integer induction variable. scf.for also permits index,
    // which would trip a cast deep inside Builder::getIntegerAttr -- fail
    // early and mention what actually broke.
    assert(isa<IntegerType>(ivType) &&
           "TLE pipeline requires an integer scf.for induction variable");
    Block *body = forOp.getBody();
    unsigned nSlots = 2 * H;
    unsigned numOrig = forOp.getInitArgs().size();

    OpBuilder b(forOp);

    // Slots. Slot 0 reuses the kernel's own alloc; the rest are clones, not
    // fresh `create<>`s, so that the discardable `xpu.tile_layout` survives
    // (C18) -- losing it silently falls back to the legacy flat cut.
    for (Candidate &c : cands) {
      c.slots.assign({c.allocOp.getResult()});
      Operation *prev = c.allocOp;
      for (unsigned k = 1; k < nSlots; ++k) {
        Operation *clone = c.allocOp->clone();
        b.setInsertionPointAfter(prev);
        b.insert(clone);
        bool shared = slotIsShared(k % H, H, c.originIsShared);
        if (shared != c.originIsShared) {
          auto t = cast<MemDescType>(clone->getResult(0).getType());
          clone->getResult(0).setType(memDescWithSpace(t, shared));
        }
        c.slots.push_back(clone->getResult(0));
        prev = clone;
      }
    }
    b.setInsertionPoint(forOp);

    // Address computations feeding only a staged copy would land in the compute
    // half as dead code; the prefetch clones them itself with a shifted offset.
    DenseSet<Operation *> skipInCompute;
    for (Candidate &c : cands)
      skipInCompute.insert(c.copyOp);
    for (Operation &op : llvm::reverse(*body)) {
      if (skipInCompute.contains(&op) || op.getNumResults() == 0 ||
          op.use_empty() || !isMemoryEffectFree(&op))
        continue;
      if (llvm::all_of(op.getUsers(),
                       [&](Operation *u) { return skipInCompute.contains(u); }))
        skipInCompute.insert(&op);
    }

    // Replay one round of fills at offset `off` into `dsts`, asynchronously.
    auto issue = [&](OpBuilder &ib, Value off, ArrayRef<Value> dsts) {
      IRMapping map;
      map.map(origIV, off);
      for (unsigned ci = 0; ci < cands.size(); ++ci) {
        for (Operation *op : cands[ci].addressSlice)
          if (!map.lookupOrNull(op->getResult(0)))
            ib.clone(*op, map);
        Operation *nc = ib.clone(*cands[ci].copyOp, map);
        nc->setOperand(copyDstOperandIdx(cands[ci].copyOp), dsts[ci]);
        copySetIsSync(nc, false);
      }
    };

    // Prologue: fill `cur_h` for every half, each guarded so a short trip count
    // cannot read past the end.
    for (unsigned h = 0; h < H; ++h) {
      Value off = h == 0 ? lb : b.create<arith::AddIOp>(loc, lb, origStep);
      Value guard =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, off, ub);
      auto ifOp = b.create<scf::IfOp>(loc, guard, /*withElseRegion=*/false);
      OpBuilder ib(ifOp.thenYield());
      SmallVector<Value> dsts;
      for (Candidate &c : cands)
        dsts.push_back(c.slots[h]);
      issue(ib, off, dsts);
    }

    // Step multiples the body needs. Only `H <= 2` is reachable, so the
    // prologue never needs anything past 1x.
    std::set<unsigned> mults;
    for (unsigned h = 1; h < H; ++h)
      mults.insert(h);
    mults.insert(H);
    for (unsigned h = 0; h < H; ++h)
      mults.insert(H + h);
    DenseMap<unsigned, Value> scaled;
    for (unsigned m : mults) {
      if (m == 1)
        continue;
      scaled[m] = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(ivType, (int64_t)m * stepConst));
    }
    auto scaledStep = [&](unsigned m) -> Value {
      return m == 1 ? origStep : scaled.lookup(m);
    };

    SmallVector<Value> initArgs(forOp.getInitArgs());
    for (Candidate &c : cands)
      initArgs.append(c.slots.begin(), c.slots.end());
    auto newFor = b.create<scf::ForOp>(loc, lb, ub, scaledStep(H), initArgs);
    Block *nb = newFor.getBody();
    Value newIV = newFor.getInductionVar();

    SmallVector<SmallVector<Value>> slotArgs(cands.size());
    unsigned argIdx = 1 + numOrig;
    for (unsigned ci = 0; ci < cands.size(); ++ci)
      for (unsigned k = 0; k < nSlots; ++k)
        slotArgs[ci].push_back(nb->getArgument(argIdx++));

    // One half: drain, then prefetch, then compute. Draining first is not a
    // style choice -- `tle_wait` is a bulk drain, so a prefetch issued before
    // it would be swallowed by the very wait meant to cover the previous fill.
    auto emitHalf = [&](OpBuilder &hb, unsigned h, Value computeOff,
                        ArrayRef<Value> in) -> SmallVector<Value> {
      std::set<unsigned> masks;
      for (Candidate &c : cands)
        masks.insert(slotIsShared(h, H, c.originIsShared)
                         ? kSharedMemWaitMask
                         : static_cast<unsigned>(waitMask));
      for (unsigned m : masks)
        hb.create<TLEWaitOp>(loc, hb.getI32IntegerAttr(m));

      Value poff = hb.create<arith::AddIOp>(loc, newIV, scaledStep(H + h));
      Value pguard =
          hb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, poff, ub);
      auto pif = hb.create<scf::IfOp>(loc, pguard, /*withElseRegion=*/false);
      SmallVector<Value> dsts;
      for (unsigned ci = 0; ci < cands.size(); ++ci)
        dsts.push_back(slotArgs[ci][H + h]);
      OpBuilder ib(pif.thenYield());
      issue(ib, poff, dsts);

      IRMapping map;
      map.map(origIV, computeOff);
      for (unsigned i = 0; i < in.size(); ++i)
        map.map(body->getArgument(1 + i), in[i]);
      for (unsigned ci = 0; ci < cands.size(); ++ci)
        map.map(copyDstBuffer(cands[ci].copyOp), slotArgs[ci][h]);
      // C8: `tle_local_ptr` discards its indices and lowers to a constant gep,
      // so there is one per slot and the loop-invariant ones must come back in.
      for (Candidate &c : cands)
        for (Operation *lp : c.outerPtrOps)
          retypeLocalPtr(lp, hb.clone(*lp, map));
      for (Operation &op : body->without_terminator())
        if (!skipInCompute.contains(&op))
          retypeLocalPtr(&op, hb.clone(op, map));

      SmallVector<Value> out;
      for (Value v : body->getTerminator()->getOperands())
        out.push_back(map.lookupOrDefault(v));
      return out;
    };

    // Halves 1..H-1 run one original iteration further along, so they need
    // their own guard. When the loop carries values the guard has to forward
    // them through an else branch.
    OpBuilder bb = OpBuilder::atBlockBegin(nb);
    SmallVector<Value> carried(nb->getArguments().begin() + 1,
                               nb->getArguments().begin() + 1 + numOrig);
    SmallVector<Type> carriedTypes(forOp.getResultTypes());
    for (unsigned h = 0; h < H; ++h) {
      if (h == 0) {
        carried = emitHalf(bb, 0, newIV, carried);
        continue;
      }
      Value off = bb.create<arith::AddIOp>(loc, newIV, scaledStep(h));
      Value guard =
          bb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, off, ub);
      if (carried.empty()) {
        auto ifOp = bb.create<scf::IfOp>(loc, guard, /*withElseRegion=*/false);
        OpBuilder ib(ifOp.thenYield());
        emitHalf(ib, h, off, carried);
        continue;
      }
      // Non-empty result types means scf::IfOp::build creates no terminators.
      auto ifOp = bb.create<scf::IfOp>(loc, carriedTypes, guard,
                                       /*withElseRegion=*/true);
      OpBuilder ib = OpBuilder::atBlockEnd(ifOp.thenBlock());
      SmallVector<Value> inner = emitHalf(ib, h, off, carried);
      ib.create<scf::YieldOp>(loc, inner);
      OpBuilder eb = OpBuilder::atBlockEnd(ifOp.elseBlock());
      eb.create<scf::YieldOp>(loc, carried);
      carried.assign(ifOp.getResults().begin(), ifOp.getResults().end());
    }

    SmallVector<Value> yieldVals(carried);
    for (unsigned ci = 0; ci < cands.size(); ++ci)
      for (unsigned k = 0; k < nSlots; ++k)
        yieldVals.push_back(slotArgs[ci][(k + H) % nSlots]);
    bb.setInsertionPointToEnd(nb);
    bb.create<scf::YieldOp>(loc, yieldVals);

    newFor->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());
    newFor->setAttr(kPipelinedAttrName, UnitAttr::get(ctx));
    forOp->replaceAllUsesWith(
        llvm::to_vector(newFor.getResults().take_front(numOrig)));
    forOp.erase();
    for (Candidate &c : cands)
      for (Operation *lp : c.outerPtrOps)
        if (lp->use_empty())
          lp->erase();
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
