//===----------- InsertBarrierPass.cpp - Tx81 Barrier Insertion --------===//
//
// This pass implements the behavior described in `BarrierInsertion.md`:
// - Use a membar-style analysis for SPM(shared memory) hazards to insert
//   `tx::BarrierOp` only when needed.
// - Add a minimal DDR hazard check for WDMA->RDMA (and CPU touching DDR after
//   WDMA) to avoid stale reads.
//
// `tx::BarrierOp` is later lowered in Tx81ToLLVM to `__Barrier` (equivalent to
// TsmWaitfinish) to keep the runtime interface unchanged.
// Some `tx.barrier` in `scf.for` bodies may be hoisted to the preheader; see
// `hoistTxBarriersFromScfForLoops`.
//
//===----------------------------------------------------------------------===//

#include "Analysis/Allocation.h"
#include "Analysis/Membar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "tsingmicro-tx81/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tx81-insert-barrier"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_INSERTBARRIER
#include "tsingmicro-tx81/Transforms/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static bool isTxDialect(Operation *op) {
  auto *d = op->getDialect();
  return d && d->getNamespace() == "tx";
}

static bool isExplicitBarrier(Operation *op) {
  return isa<tx::BarrierOp, tx::AtomicBarrierInOp, tx::AtomicBarrierOutOp>(op);
}

/// Tx ops that touch data (NPU/DMA/compute). Excludes barrier-only tx ops so
/// we can tell "producer tx is inside this loop body" vs "only CPU inside".
static bool isTxDataOp(Operation *op) {
  if (!isTxDialect(op))
    return false;
  if (isExplicitBarrier(op))
    return false;
  return true;
}

/// True if a `tx.barrier` appears in preorder after `lhsOp` and before `rhsOp`
/// in the same function. After NPU→CPU sync once, further CPU ops are serial;
/// a second barrier on the same chain (e.g. before another `scf.for`) is
/// redundant when IR already has `tx.barrier` between the tx write and rhs.
static bool txBarrierBetweenLhsAndRhs(Operation *lhsOp, Operation *rhsOp) {
  auto funcOp = lhsOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp || funcOp != rhsOp->getParentOfType<FunctionOpInterface>())
    return false;

  bool seenLhs = false;
  bool seenBarrierAfterLhs = false;
  bool result = false;

  funcOp.getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rhsOp) {
      result = seenBarrierAfterLhs;
      return WalkResult::interrupt();
    }
    if (op == lhsOp)
      seenLhs = true;
    if (seenLhs &&
        (isa<tx::BarrierOp, tx::AtomicBarrierInOp, tx::AtomicBarrierOutOp>(op)))
      seenBarrierAfterLhs = true;
    return WalkResult::advance();
  });
  return result;
}

static Value getBaseBuffer(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::SubViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::CastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ExtractStridedMetadataOp>(defOp)) {
      if (v == op->getResult(0))
        v = op.getSource();
      else
        break;
    } else
      break;
  }
  return v;
}

static Value traceToOriginMemRef(Value v, unsigned maxDepth = 16) {
  if (maxDepth == 0)
    return {};
  if (isa<MemRefType>(v.getType()))
    return getBaseBuffer(v);
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return {};

  if (auto op = dyn_cast<arith::IndexCastOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::TruncIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtSIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtUIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(defOp))
    return getBaseBuffer(op.getSource());
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::OrIOp>(defOp)) {
    if (Value r = traceToOriginMemRef(defOp->getOperand(0), maxDepth - 1))
      return r;
    return traceToOriginMemRef(defOp->getOperand(1), maxDepth - 1);
  }
  return {};
}

static Value resolveOrigin(Value v) {
  if (!v)
    return {};
  if (isa<MemRefType>(v.getType()))
    return getBaseBuffer(v);
  return traceToOriginMemRef(v);
}

static bool
getAllocationOffsetInterval(Value v,
                            triton::alloc::Interval<size_t> &interval) {
  Value base = resolveOrigin(v);
  if (!base)
    return false;

  auto alloc = base.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return false;

  auto offsetAttr = alloc->getAttrOfType<IntegerAttr>("allocation.offset");
  if (!offsetAttr)
    return false;

  int64_t signedOffset = offsetAttr.getInt();
  if (signedOffset < 0)
    return false;

  MemRefType allocType = alloc.getType();
  if (!allocType.hasStaticShape())
    return false;

  int64_t numElements = allocType.getNumElements();
  unsigned bitWidth = allocType.getElementTypeBitWidth();
  uint64_t elemBytes = (bitWidth + 7) / 8;
  if (numElements < 0 || elemBytes == 0)
    return false;
  if (static_cast<uint64_t>(numElements) >
      std::numeric_limits<uint64_t>::max() / elemBytes)
    return false;

  uint64_t bytes = static_cast<uint64_t>(numElements) * elemBytes;
  uint64_t offset = static_cast<uint64_t>(signedOffset);
  uint64_t maxSize = std::numeric_limits<size_t>::max();
  if (offset > maxSize || bytes > maxSize - offset)
    return false;

  interval = triton::alloc::Interval<size_t>(
      static_cast<size_t>(offset), static_cast<size_t>(offset + bytes));
  return true;
}

static bool mayShareAllocationOffset(Value a, Value b) {
  triton::alloc::Interval<size_t> lhs, rhs;
  if (!getAllocationOffsetInterval(a, lhs) ||
      !getAllocationOffsetInterval(b, rhs))
    return false;
  return lhs.intersects(rhs);
}

static bool mayAliasOrigin(Value a, Value b) {
  if (!a || !b)
    return true;
  if (a == b)
    return true;
  // If both are distinct memref allocs/args, and differ, treat as no-alias.
  auto isDistinct = [](Value v) -> bool {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return isa<MemRefType>(ba.getType());
    if (auto *def = v.getDefiningOp())
      return isa<memref::AllocOp, memref::AllocaOp>(def);
    return false;
  };
  if (isDistinct(a) && isDistinct(b))
    return false;
  return true;
}

/// Collect memref "base" values for SPM-style alias checks (same idea as Membar
/// buffer resolution).
static void collectMemrefBasesForOp(Operation *op,
                                    SmallVectorImpl<Value> &bases) {
  for (Value v : op->getOperands()) {
    if (Value base = resolveOrigin(v))
      bases.push_back(base);
    else if (isa<MemRefType>(v.getType()))
      bases.push_back(getBaseBuffer(v));
  }
}

/// True if `producer` and `consumer` may touch the same SPM allocation.
static bool mayShareSpmMemref(Operation *producer, Operation *consumer) {
  SmallVector<Value, 4> pa, pb;
  collectMemrefBasesForOp(producer, pa);
  collectMemrefBasesForOp(consumer, pb);
  if (pa.empty() || pb.empty())
    return false;
  for (Value a : pa)
    for (Value b : pb)
      if (mayAliasOrigin(a, b) || mayShareAllocationOffset(a, b))
        return true;
  return false;
}

static bool isCpuDataOp(Operation *op) {
  return op && !isTxDialect(op) && !triton::membar::isPureAddressOp(op);
}

static bool touchesSpmAllocation(Operation *op,
                                 triton::alloc::Allocation *allocation) {
  SmallVector<Value, 4> bases;
  collectMemrefBasesForOp(op, bases);
  for (Value base : bases)
    if (!allocation->getBufferIds(base).empty())
      return true;
  return false;
}

/// True if, in one iteration template, some CPU-side non-tx data op appears
/// before a tx data op in preorder and they may share SPM. Then iteration i+1's
/// CPU can run before iteration i's async NPU finishes — need a barrier at the
/// end of the body (before yield) so the next iteration's CPU waits.
static bool cpuPrecedesTxOnSharedSpmInSameBody(scf::ForOp forOp) {
  SmallVector<Operation *> cpuMemOps;
  bool found = false;
  forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isCpuDataOp(op)) {
      cpuMemOps.push_back(op);
      return WalkResult::advance();
    }
    if (isTxDataOp(op)) {
      for (Operation *cpu : cpuMemOps) {
        if (mayShareSpmMemref(cpu, op)) {
          found = true;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return found;
}

/// Insert `tx.barrier` before `scf.yield` when loop-carried SPM sync is needed.
static void insertLoopCarriedSpmBarriers(ModuleOp mod) {
  mod.walk([&](scf::ForOp forOp) {
    if (!cpuPrecedesTxOnSharedSpmInSameBody(forOp))
      return;
    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      return;
    if (Operation *prev = yield->getPrevNode();
        prev && isa<tx::BarrierOp>(prev))
      return;
    OpBuilder b(yield);
    b.create<tx::BarrierOp>(yield->getLoc());
  });
}

/// Hoist barrier out of `forOp` only when the hazard is not "tx producer and
/// CPU consumer both inside this loop, same SPM". Membar inserts the barrier
/// immediately before `consumer`; we find tx ops before the barrier that may
/// produce the memref `consumer` reads — if that tx is in the loop region, keep
/// the barrier inside (per-iteration sync). If the tx producer is outside the
/// loop and only `consumer` is inside, one barrier before the loop is correct.
static bool shouldHoistBarrierFromLoop(scf::ForOp forOp,
                                       tx::BarrierOp barrier) {
  Operation *consumer = barrier->getNextNode();
  if (!consumer)
    return false;

  Operation *pairedTx = nullptr;
  bool anyTxDataBeforeBarrier = false;

  forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == barrier)
      return WalkResult::interrupt();
    if (isTxDataOp(op)) {
      anyTxDataBeforeBarrier = true;
      if (mayShareSpmMemref(op, consumer))
        pairedTx = op;
    }
    return WalkResult::advance();
  });

  if (pairedTx && forOp->isAncestor(pairedTx) && forOp->isAncestor(consumer))
    return false;

  // Could not match producer↔consumer buffers (e.g. i64 address only): if any
  // tx data op precedes the barrier in the loop body, keep barriers inside.
  if (!pairedTx && anyTxDataBeforeBarrier)
    return false;

  return true;
}

/// Hoist `tx.barrier` from a `scf.for` body to the preheader only when
/// `shouldHoistBarrierFromLoop` says so (see dependency-based rule there).
static void hoistTxBarriersFromScfForLoops(ModuleOp mod) {
  mod.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    SmallVector<tx::BarrierOp, 8> barriers;
    forOp.getBody()->walk([&](tx::BarrierOp b) { barriers.push_back(b); });
    if (barriers.empty())
      return;

    SmallVector<tx::BarrierOp, 8> toHoist;
    for (tx::BarrierOp br : barriers) {
      if (!shouldHoistBarrierFromLoop(forOp, br))
        continue;
      // Barrier immediately before scf.yield ends the iteration after NPU work;
      // must stay in the body (per-iter sync). Hoisting would run once outside.
      if (Operation *n = br->getNextNode())
        if (isa<scf::YieldOp>(n))
          continue;
      toHoist.push_back(br);
    }
    if (toHoist.empty())
      return;

    OpBuilder b(forOp);
    b.setInsertionPoint(forOp);
    b.create<tx::BarrierOp>(forOp.getLoc());
    for (tx::BarrierOp br : toHoist)
      br.erase();
  });
}

static bool
topLevelOpHasCpuSpmAccessBeforeBarrier(Operation *topLevelOp,
                                       triton::alloc::Allocation *allocation) {
  bool foundCpuSpmAccess = false;
  topLevelOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != topLevelOp && isExplicitBarrier(op)) {
      return WalkResult::interrupt();
    }
    if (!isCpuDataOp(op))
      return WalkResult::advance();
    if (touchesSpmAllocation(op, allocation)) {
      foundCpuSpmAccess = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return foundCpuSpmAccess;
}

static void insertKernelEntryBarriersBeforeCpuSpmUse(
    ModuleOp mod, triton::alloc::ModuleAllocation &moduleAlloc) {
  mod.walk([&](func::FuncOp fn) {
    if (fn.isExternal() || fn.getBody().empty())
      return;
    auto *allocation = moduleAlloc.getFuncData(fn);
    if (!allocation)
      return;

    for (Operation &op : fn.getBody().front().getOperations()) {
      if (isExplicitBarrier(&op))
        return;
      if (isa<func::ReturnOp>(&op))
        return;
      if (topLevelOpHasCpuSpmAccessBeforeBarrier(&op, allocation)) {
        if (Operation *prev = op.getPrevNode();
            prev && isExplicitBarrier(prev)) {
          return;
        }
        OpBuilder b(&op);
        b.create<tx::BarrierOp>(op.getLoc());
        return;
      }
    }
  });
}

class InsertBarrierPass
    : public triton::impl::InsertBarrierBase<InsertBarrierPass> {
  using InsertBarrierBase<InsertBarrierPass>::InsertBarrierBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tx::Tx81Dialect, func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // 1) SPM hazards: membar + filter. One `tx.barrier` per NPU→CPU sync chain:
    // NPU→barrier→CPU→CPU needs no second barrier between CPUs. Also suppress
    // CPU↔CPU (program order). If a `tx.barrier` already appears on the path
    // from the tx op (lhs) to this CPU op (rhs), suppress (see
    // `txBarrierBetweenLhsAndRhs`). CPU before tx: host finished prior work.
    triton::alloc::ModuleAllocation moduleAlloc(mod);
    auto filter = [](Operation *lhsOp, Operation *rhsOp,
                     triton::membar::MembarHazardKind) -> bool {
      // Return true means "suppress barrier".
      if (isTxDialect(lhsOp) && isTxDialect(rhsOp))
        return true;
      if (!isTxDialect(lhsOp) && isTxDialect(rhsOp))
        return true;
      if (!isTxDialect(lhsOp) && !isTxDialect(rhsOp))
        return true;
      if (isTxDialect(lhsOp) && !isTxDialect(rhsOp) &&
          txBarrierBetweenLhsAndRhs(lhsOp, rhsOp))
        return true;
      return false;
    };
    triton::membar::ModuleMembarAnalysis membar(&moduleAlloc, filter);
    membar.run();

    // 2) DDR hazards: minimal WDMA->RDMA and CPU-touching-DDR-after-WDMA.
    // We track pending WDMA writes by their origin memref and insert a
    // `tx::BarrierOp` before any RDMA/CPU op that may read that same origin.
    mod.walk([&](func::FuncOp fn) {
      SmallVector<Value, 8> pendingDdrWrites;
      auto clearPending = [&]() { pendingDdrWrites.clear(); };

      auto markWdmaWrite = [&](Operation *op) {
        Value tgt;
        if (auto w = dyn_cast<tx::WdmaOp>(op))
          tgt = w.getTarget();
        else if (auto w = dyn_cast<tx::Wdma4dOp>(op))
          tgt = w.getTarget();
        else if (auto w = dyn_cast<tx::Wdma1dOp>(op))
          tgt = w.getTarget();
        Value origin = resolveOrigin(tgt);
        if (origin)
          pendingDdrWrites.push_back(origin);
      };

      auto needsBarrierForDdrRead = [&](Value addrLike) -> bool {
        Value origin = resolveOrigin(addrLike);
        if (!origin)
          return false;
        for (Value w : pendingDdrWrites)
          if (mayAliasOrigin(origin, w))
            return true;
        return false;
      };

      auto insertBarrierBefore = [&](Operation *op) {
        OpBuilder b(op);
        b.create<tx::BarrierOp>(op->getLoc());
        clearPending();
      };

      for (Block &block : fn.getBody()) {
        // Pending WDMA writes are only meaningful within straight-line code.
        // Start each block conservatively with an empty pending set.
        clearPending();
        for (Operation &op :
             llvm::make_early_inc_range(block.getOperations())) {
          if (isa<tx::BarrierOp, tx::AtomicBarrierInOp, tx::AtomicBarrierOutOp>(
                  &op)) {
            clearPending();
            continue;
          }

          if (isa<tx::WdmaOp, tx::Wdma4dOp, tx::Wdma1dOp>(&op)) {
            markWdmaWrite(&op);
            continue;
          }

          if (isa<tx::RdmaOp, tx::Rdma4dOp, tx::Rdma1dOp>(&op)) {
            Value src;
            if (auto r = dyn_cast<tx::RdmaOp>(&op))
              src = r.getSource();
            else if (auto r = dyn_cast<tx::Rdma4dOp>(&op))
              src = r.getSource();
            else if (auto r = dyn_cast<tx::Rdma1dOp>(&op))
              src = r.getSource();
            if (!pendingDdrWrites.empty() && needsBarrierForDdrRead(src)) {
              insertBarrierBefore(&op);
            }
            continue;
          }

          // CPU ops: any non-tx op that uses a pending DDR region triggers a
          // barrier. Skip memref/arith/scf address prep (same as SPM membar).
          if (!isTxDialect(&op) && !pendingDdrWrites.empty() &&
              !triton::membar::isPureAddressOp(&op)) {
            bool conflict = false;
            for (Value operand : op.getOperands()) {
              if (needsBarrierForDdrRead(operand)) {
                conflict = true;
                break;
              }
            }
            if (conflict)
              insertBarrierBefore(&op);
          }
        }
      }
    });

    // 3) Loop-carried SPM: same body has CPU memref access before a tx op on
    // aliasing SPM — next iter's CPU can overlap previous iter's async NPU.
    // Sync at end of body (before scf.yield); do not hoist that barrier.
    insertLoopCarriedSpmBarriers(mod);

    // 4) Kernel-boundary sync: different kernels may reuse the same physical
    // SPM. If a kernel starts with CPU-side SPM access before any explicit
    // barrier, insert one barrier before that first top-level CPU region (for
    // example, before an enclosing `scf.for`).
    insertKernelEntryBarriersBeforeCpuSpmUse(mod, moduleAlloc);

    // 5) Hoist barriers out of `scf.for` only when sync is outer-tx vs
    // inner-CPU (see `hoistTxBarriersFromScfForLoops`); skip yield-adjacent.
    hoistTxBarriersFromScfForLoops(mod);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createInsertBarrierPass() {
  return std::make_unique<InsertBarrierPass>();
}
