#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"

// FlagMega-on-Radeon P1-B: lower ttg.warp_specialize onto a static wave-id
// partition. Correctness-first — no setmaxnreg, no persistent switch loop, no
// LDS capture struct. Lowered at the TTGIR level (like BlockPingpong) so the
// inlined region bodies flow through the normal AMD pipeline.
//
//   wid = workitem.id.x / warpSize     ; wave-uniform
//   gpu.barrier                        ; entry
//   cf.cond_br wid < defaultNumWarps -> default, chk0
//   default: <default region>; warp_yield -> br rejoin
//   chk_i:   cond_br start_i <= wid < start_i+nw_i -> partition_i, chk_{i+1}
//   partition_i: <partition region>; warp_return -> br rejoin
//   chk_N:   br rejoin                 ; padding/unassigned waves
//   rejoin:  gpu.barrier; br after
//
// Captures are dominating SSA values (all waves ran the enclosing code), so
// partition block-args are replaced by the explicit captures and erased — no
// shared memory needed. Intra-partition synchronization is expected to use the
// P1-A LDS mbarriers (count-scoped), never a CTA-wide gpu.barrier.

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTWARPSPECIALIZE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// A rank-1 LDS slot (memdesc<1xT>) used to broadcast a scalar warp_yield result
// from the default waves to all waves.
static ttg::MemDescType scalarSlotType(MLIRContext *ctx, Type elemTy) {
  Attribute enc = ttg::SwizzledSharedEncodingAttr::get(
      ctx, 1, 1, 1, SmallVector<unsigned>{0},
      ttg::CTAEncodingAttr::getDefault(ctx, 1));
  return ttg::MemDescType::get({1}, elemTy, enc,
                               ttg::SharedMemorySpaceAttr::get(ctx),
                               /*mutableMemory=*/true);
}

static RankedTensorType scalarTensorType(Operation *op, Type elemTy) {
  MLIRContext *ctx = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  int nw = ttg::lookupNumWarps(op);
  int tpw = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int nctas = ttg::TritonGPUDialect::getNumCTAs(module);
  Attribute enc = ttg::getDefaultBlockedEncoding(ctx, {1}, nw, tpw, nctas);
  return RankedTensorType::get({1}, elemTy, enc);
}

static LogicalResult lowerWarpSpecialize(ttg::WarpSpecializeOp ws) {
  // warp_yield results only the default waves produce; broadcast scalar results
  // to all waves through LDS (store at warp_yield, load after the rejoin
  // barrier). Non-scalar (tensor/memdesc) results need layout-aware
  // redistribution and are not supported yet.
  for (Type rt : ws->getResultTypes())
    if (!rt.isIntOrIndexOrFloat())
      return ws.emitOpError(
          "warp_specialize returning a non-scalar value is unsupported on the "
          "RDNA backend yet (only uniform scalar warp_yield results are "
          "broadcast via LDS)");

  MLIRContext *ctx = ws.getContext();
  Location loc = ws.getLoc();
  auto i32ty = IntegerType::get(ctx, 32);
  Block *predBlock = ws->getBlock();
  Region *fnRegion = predBlock->getParent();
  auto module = ws->getParentOfType<ModuleOp>();

  int defaultNumWarps = ttg::lookupNumWarps(ws);
  int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (defaultNumWarps <= 0 || warpSize <= 0)
    return ws.emitOpError("requires positive num_warps and threads_per_warp");

  SmallVector<int64_t> numWarps(ws.getPartitionNumWarps().begin(),
                                ws.getPartitionNumWarps().end());
  SmallVector<int64_t> startIds;
  if (std::optional<ArrayRef<int32_t>> ids = ws.getWarpGroupStartIds()) {
    startIds.assign(ids->begin(), ids->end());
  } else {
    // No allocation pass yet: assign contiguous ranges after the default group.
    int64_t cur = defaultNumWarps;
    for (int64_t nw : numWarps) {
      startIds.push_back(cur);
      cur += nw;
    }
  }
  size_t n = numWarps.size();

  // A CTA-wide barrier inside a partition would wait on waves that are on a
  // different path and deadlock. Partitions must sync via pipe/mbarrier.
  for (Region *pr : ws.getPartitionRegions()) {
    auto wr = pr->walk([&](gpu::BarrierOp) { return WalkResult::interrupt(); });
    if (wr.wasInterrupted())
      return ws.emitOpError(
          "a CTA-wide barrier (gpu.barrier) inside a warp_specialize partition "
          "would deadlock on RDNA; use pipe/mbarrier for intra-partition sync");
  }

  // Wave id + entry barrier, emitted before the op.
  OpBuilder b(ws);
  // LDS slots to broadcast scalar warp_yield results to all waves.
  SmallVector<Value> yieldSlots;
  SmallVector<RankedTensorType> yieldTensorTys;
  for (Type rt : ws->getResultTypes()) {
    yieldSlots.push_back(
        ttg::LocalAllocOp::create(b, loc, scalarSlotType(ctx, rt)));
    yieldTensorTys.push_back(scalarTensorType(ws, rt));
  }
  Value tid = ROCDL::ThreadIdXOp::create(b, loc, i32ty);
  Value wsz = arith::ConstantIntOp::create(b, loc, warpSize, 32);
  Value wid = arith::DivUIOp::create(b, loc, tid, wsz);
  gpu::BarrierOp::create(b, loc);

  // Everything after the op becomes the post-join continuation.
  Block *afterBlk = predBlock->splitBlock(std::next(ws->getIterator()));

  // Rejoin block: CTA-wide barrier, then (after the stores are visible) load the
  // broadcast yield results and fall into the continuation.
  Block *rejoin = new Block();
  fnRegion->getBlocks().insert(Region::iterator(afterBlk), rejoin);
  {
    OpBuilder rb(rejoin, rejoin->end());
    gpu::BarrierOp::create(rb, loc);
    for (auto [i, slot] : llvm::enumerate(yieldSlots)) {
      Value t = ttg::LocalLoadOp::create(rb, loc, yieldTensorTys[i], slot);
      Value v = tt::UnsplatOp::create(rb, loc, ws->getResultTypes()[i], t);
      ws.getResult(i).replaceAllUsesWith(v);
    }
    cf::BranchOp::create(rb, loc, afterBlk);
  }

  // Inline the default region; its warp_yield stores its results to LDS and
  // becomes a branch to rejoin.
  Region &defReg = ws.getDefaultRegion();
  Block *defEntry = &defReg.front();
  defReg.walk([&](ttg::WarpYieldOp y) {
    OpBuilder yb(y);
    for (auto [i, operand] : llvm::enumerate(y.getOperands())) {
      Value splat = tt::SplatOp::create(yb, y.getLoc(), yieldTensorTys[i], operand);
      ttg::LocalStoreOp::create(yb, y.getLoc(), splat, yieldSlots[i]);
    }
    cf::BranchOp::create(yb, y.getLoc(), rejoin);
    y.erase();
  });
  fnRegion->getBlocks().splice(Region::iterator(rejoin), defReg.getBlocks());

  // Inline each partition region; replace its captures block-args with the
  // explicit captures (dominating values) and route warp_return to rejoin.
  auto captures = ws.getExplicitCaptures();
  SmallVector<Block *> partEntry;
  for (Region *pr : ws.getPartitionRegions()) {
    Block *pe = &pr->front();
    for (auto [j, cap] : llvm::enumerate(captures))
      pe->getArgument(j).replaceAllUsesWith(cap);
    pe->eraseArguments(0, pe->getNumArguments());
    pr->walk([&](ttg::WarpReturnOp r) {
      OpBuilder rb(r);
      cf::BranchOp::create(rb, r.getLoc(), rejoin);
      r.erase();
    });
    partEntry.push_back(pe);
    fnRegion->getBlocks().splice(Region::iterator(rejoin), pr->getBlocks());
  }

  // ws regions are now empty; drop the op.
  ws.erase();

  // Dispatch check blocks (one per partition, plus a fall-through).
  SmallVector<Block *> chk;
  for (size_t i = 0; i <= n; ++i) {
    Block *c = new Block();
    fnRegion->getBlocks().insert(Region::iterator(rejoin), c);
    chk.push_back(c);
  }

  // predBlock: branch default waves to the default region, rest to chk0.
  {
    OpBuilder pb(predBlock, predBlock->end());
    Value def = arith::ConstantIntOp::create(pb, loc, defaultNumWarps, 32);
    Value isDef =
        arith::CmpIOp::create(pb, loc, arith::CmpIPredicate::ult, wid, def);
    cf::CondBranchOp::create(pb, loc, isDef, defEntry, chk[0]);
  }
  // chk_i: wave in [start_i, start_i+nw_i) -> partition_i, else chk_{i+1}.
  for (size_t i = 0; i < n; ++i) {
    OpBuilder cb(chk[i], chk[i]->end());
    Value lo = arith::ConstantIntOp::create(cb, loc, startIds[i], 32);
    Value hi =
        arith::ConstantIntOp::create(cb, loc, startIds[i] + numWarps[i], 32);
    Value ge = arith::CmpIOp::create(cb, loc, arith::CmpIPredicate::uge, wid, lo);
    Value lt = arith::CmpIOp::create(cb, loc, arith::CmpIPredicate::ult, wid, hi);
    Value in = arith::AndIOp::create(cb, loc, ge, lt);
    cf::CondBranchOp::create(cb, loc, in, partEntry[i], chk[i + 1]);
  }
  // Fall-through: unassigned / padding waves go straight to rejoin.
  {
    OpBuilder cb(chk[n], chk[n]->end());
    cf::BranchOp::create(cb, loc, rejoin);
  }
  return success();
}

struct TritonAMDGPUConvertWarpSpecializePass
    : impl::TritonAMDGPUConvertWarpSpecializeBase<
          TritonAMDGPUConvertWarpSpecializePass> {
  void runOnOperation() override {
    SmallVector<ttg::WarpSpecializeOp> wsOps;
    getOperation().walk([&](ttg::WarpSpecializeOp op) { wsOps.push_back(op); });
    for (ttg::WarpSpecializeOp ws : wsOps)
      if (failed(lowerWarpSpecialize(ws)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir
