#include "TritonHCUGPUTransforms/ExplicitWarpSpecialize.h"
#include "DeferredWarpWait.h"
// Conservative lowering of an explicit terminal warp-specialize region.
// No model shape, operand role, source chart, queue protocol or imported proof
// selects this path. Source CTA rendezvous are preserved; ordinary local
// barriers synchronize only their actual warp partition.
namespace {
namespace manualws {
static LogicalResult early(ModuleOp mod) {
  SmallVector<WarpSpecializeOp> ops;
  mod.walk([&](WarpSpecializeOp ws) { ops.push_back(ws); });
  if (ops.empty()) return success();
  auto reject = [&](StringRef why) -> LogicalResult { return mod.emitError("manual warp specialization: ") << why; };
  if (ops.size() != 1)
    return reject("requires one explicit terminal specialization");
  auto ws = ops.front();
  auto domain = HCU::terminalWarpDomain(ws, 16);
  if (!domain || ws.getPartitionRegions().size() + 1 > kNumBarriers)
    return reject("requires a void terminal wave64 single-CTA specialization within hardware wave/barrier limits");
  ws.setWarpGroupStartIds(domain->starts);
  mod->setAttr("ttg.total-num-warps", IntegerAttr::get(IntegerType::get(mod.getContext(), 32), domain->total));
  HCU::deferred_wait::WindowTrace(ws).run();
  return success();
}

static bool canRematerializeCapture(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return isPure(op) &&
         (name.starts_with("llvm.") || isa<UnrealizedConversionCastOp>(op)) &&
         !name.contains("call") && !name.contains("load") &&
         !name.contains("intrinsic");
}

static void localCtaBarrier(triton::gpu::LocalBarrierOp op) {
  TritonLLVMIRRewriter b(op.getLoc(), op);
  // LDS completion is part of this explicit local-memory CTA rendezvous.
  // Pending global register prefetches are not implicitly completed here.
  b.create<ROCDL::SWaitcntOp>(0xc07f);
  b.create<ROCDL::SBarrierOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  constexpr llvm::StringLiteral markerName = "__triton_hcu_local_barrier_marker";
  auto marker = mod.lookupSymbol<LLVM::LLVMFuncOp>(markerName);
  if (!marker) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(mod.getBody());
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()), {}, false);
    marker = b.create<LLVM::LLVMFuncOp>(markerName, type);
    marker.setConvergent(true);
    marker.setPassthroughAttr(b.getArrayAttr({b.getStringAttr("nounwind"),
        b.getStringAttr("willreturn"), b.getStringAttr("nocallback"), b.getStringAttr("nofree")}));
  }
  LLVM::createLLVMCallOp(b, op.getLoc(), marker, {});
  op.erase();
}

static LogicalResult late(LLVM::LLVMFuncOp fn) {
  SmallVector<WarpSpecializeOp> ops;
  fn.walk([&](WarpSpecializeOp ws) { ops.push_back(ws); });
  if (ops.empty()) return success();
  if (ops.size() != 1) return fn.emitError("manual warp specialization: expected one terminal region");
  auto ws = ops.front();
  auto starts = ws.getWarpGroupStartIds();
  if (ws.getNumResults() || !starts || starts->size() != ws.getPartitionRegions().size())
    return ws.emitError("manual warp specialization: missing actual partition start IDs");
  HCU::UniformWarpCaptures captureProof(fn.getBody().front(),
                                        canRematerializeCapture);
  if (!llvm::all_of(ws.getExplicitCaptures(), captureProof))
    return ws.emitError(
        "manual warp specialization: capture is not CTA-uniform and rematerializable");
  elideTrivialCaptures(fn, ops);
  if (ws.getNumOperands())
    return ws.emitError("manual warp specialization: unresolved captures");
  auto mod = fn->getParentOfType<ModuleOp>();
  auto base = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  SmallVector<Region *> regions{&ws.getDefaultRegion()};
  llvm::append_range(regions, ws.getPartitionRegions());
  for (auto [index, region] : llvm::enumerate(regions)) {
    if (region->empty() || region->getNumArguments())
      return ws.emitError("manual warp specialization: invalid partition entry");
    bool invalid = false;
    SmallVector<Operation *> localBars;
    region->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      invalid |= isa<LLVM::ReturnOp>(op) || name.contains("abarrier") || name.contains("ebarrier");
      if (isa<ROCDL::SBarrierOp, ROCDL::BarrierOp>(op)) localBars.push_back(op);
      if (auto cta = dyn_cast<triton::gpu::LocalBarrierOp>(op)) {
        if (auto request = op->getAttr("hcu.cta_scope")) {
          if (!isa<UnitAttr>(request)) invalid = true;
        }
      }
    });
    if (invalid) return ws.emitError("manual warp specialization: unsupported partition exit or reserved barrier use");
    unsigned count = index ? ws.getPartitionNumWarps()[index - 1] : base.getInt();
    if (!localBars.empty() && count != 1)
      return ws.emitError("manual warp specialization: gfx936 partition-local barriers require a single wave; multi-wave named barriers are not supported by the target toolchain");
    for (Operation *barrier : localBars) {
      // This exact marker belonged to the local barrier being replaced.
      if (auto tag = dyn_cast_or_null<LLVM::CallOp>(barrier->getNextNode()))
        if (auto symbol = tag.getCalleeAttr(); symbol && symbol.getValue() == "__triton_hcu_local_barrier_marker")
          tag.erase();
      TritonLLVMIRRewriter b(barrier->getLoc(), barrier);
      // A single-wave partition needs wave execution/memory ordering, not a
      // CTA rendezvous with other partitions. Do not promote this to s_barrier.
      b.create<ROCDL::SWaitcntOp>(0xc07f);
      LLVM::createLLVMIntrinsicCallOp(b, barrier->getLoc(),
                                     "llvm.amdgcn.wave.barrier", {}, {});
      barrier->erase();
    }
  }
  auto &blocks = fn.getBody().getBlocks();
  Block *before = ws->getBlock();
  TritonLLVMIRRewriter b(ws.getLoc(), OpBuilder::atBlockEnd(before));
  Block *after = b.splitBlock(before, ws->getIterator());
  SmallVector<Block *> entries;
  for (Region *region : regions) {
    entries.push_back(&region->front());
    region->walk([&](Operation *op) {
      if (isa<WarpYieldOp, WarpReturnOp>(op)) {
        TritonLLVMIRRewriter term(op->getLoc(), op);
        term.replaceOpWithNewOp<LLVM::BrOp>(op, ValueRange{}, after);
      }
    });
  }
  b.setInsertionPointToEnd(before);
  Value tid = b.create<mlir::gpu::ThreadIdOp>(mlir::gpu::Dimension::x);
  tid = b.create<arith::IndexCastOp>(b.getI32Type(), tid);
  // Each interval contains exactly its declared waves, including a non-power-
  // of-two total CTA. No dummy parked waves participate in source CTA barriers.
  for (unsigned index = 0; index + 1 < entries.size(); ++index) {
    unsigned end = index ? (*starts)[index] : base.getInt();
    Block *next = new Block;
    blocks.insert(after->getIterator(), next);
    b.create<LLVM::CondBrOp>(b.icmp_ult(tid, b.i32_val(end * 64)), entries[index], next);
    b.setInsertionPointToEnd(next);
  }
  b.create<LLVM::BrOp>(ValueRange{}, entries.back());
  for (Region *region : regions) blocks.splice(after->getIterator(), region->getBlocks());
  ws.erase();
  b.setInsertionPointToStart(after);
  b.create<ROCDL::SBarrierOp>();
  return success();
}
} // namespace manualws
} // namespace
