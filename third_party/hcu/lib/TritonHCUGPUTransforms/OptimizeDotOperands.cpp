#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "MatrixDescriptorLayout.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <memory>

#define DEBUG_TYPE "tritonhcugpu-optimize-dot-operands"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#include "AutoSharedLayout.h"

namespace mlir::triton::hcugpu {

namespace {

// Keep a prefetched operand in its consumers' register distribution across
// loop iterations. Converting it inside a warp partition otherwise introduces
// implicit cross-wave LDS staging. This changes SSA types and conversions,
// never the position or number of memory operations or rendezvous.
class RetargetLoopCarriedDotOperand
    : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp convert,
                               PatternRewriter &rewriter) const override {
    auto layout = dyn_cast<ttg::DotOperandEncodingAttr>(
        convert.getType().getEncoding());
    auto arg = dyn_cast<BlockArgument>(convert.getSrc());
    auto loop = arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!layout || !isa<ttg::AMDMfmaEncodingAttr>(layout.getParent()) ||
        !loop || arg.getArgNumber() == 0 ||
        !loop->getParentOfType<ttg::WarpSpecializeOp>())
      return failure();
    unsigned index = arg.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SmallVector<ttg::ConvertLayoutOp> users;
    for (OpOperand &use : arg.getUses()) {
      if (use.getOwner() == yield && use.getOperandNumber() == index)
        continue;
      auto other = dyn_cast<ttg::ConvertLayoutOp>(use.getOwner());
      if (!other || other.getType() != convert.getType())
        return failure();
      users.push_back(other);
    }
    auto oldType = arg.getType();
    auto newType = convert.getType();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);
    Value initial = rewriter.create<ttg::ConvertLayoutOp>(
        loop.getLoc(), newType, loop.getInitArgs()[index]);
    Value next = yield.getOperand(index);
    if (next != arg) {
      rewriter.setInsertionPoint(yield);
      next = rewriter.create<ttg::ConvertLayoutOp>(yield.getLoc(), newType, next);
    }
    rewriter.modifyOpInPlace(loop, [&] {
      loop.getInitArgsMutable()[index].assign(initial);
      arg.setType(newType);
      loop.getResult(index).setType(newType);
      yield.setOperand(index, next);
    });
    Value result = loop.getResult(index);
    if (!result.use_empty()) {
      rewriter.setInsertionPointAfter(loop);
      auto restore = rewriter.create<ttg::ConvertLayoutOp>(
          loop.getLoc(), oldType, result);
      result.replaceAllUsesExcept(restore.getResult(), restore);
    }
    for (ttg::ConvertLayoutOp user : users)
      rewriter.replaceOp(user, arg);
    return success();
  }
};

// Re-encode an effectful identity once, at its original position. Unlike
// rematerialization this never duplicates the fence or moves its scheduling
// effects past another operation. All users must request the same layout.
class RetargetAccumulatorFence
    : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp convert,
                               PatternRewriter &rewriter) const override {
    auto fence = convert.getSrc().getDefiningOp<AccumulatorFenceOp>();
    if (!fence)
      return failure();
    SmallVector<ttg::ConvertLayoutOp> users;
    for (Operation *user : fence.getResult().getUsers()) {
      auto other = dyn_cast<ttg::ConvertLayoutOp>(user);
      if (!other || other.getType() != convert.getType())
        return failure();
      users.push_back(other);
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(fence);
    Value input = fence.getAcc();
    if (input.getType() != convert.getType())
      input = rewriter.create<ttg::ConvertLayoutOp>(
          fence.getLoc(), convert.getType(), input);
    OperationState state(fence.getLoc(), fence->getName(), {input},
                         {convert.getType()}, fence->getAttrs());
    Operation *retargeted = rewriter.create(state);
    for (ttg::ConvertLayoutOp user : users)
      rewriter.replaceOp(user, retargeted->getResult(0));
    rewriter.eraseOp(fence);
    return success();
  }
};

// gfx936 cannot rendezvous a multi-wave worker independently of the CTA.
// Prefer global loads in the actual MMAC operand layout over implicit
// cross-wave LDS staging. Keep the load at its original memory position and
// reuse ordinary layout removal for the pure pointer/mask computations.
class LoadDotOperandRegisters : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp convert,
                               PatternRewriter &rewriter) const override {
    Attribute layout = convert.getType().getEncoding();
    auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout);
    Operation *load = convert.getSrc().getDefiningOp();
    auto ordinary = dyn_cast_or_null<tt::LoadOp>(load);
    auto matrix = dyn_cast_or_null<tt::MatrixLoadOp>(load);
    bool descriptor = matrix && triton::HCU::isSyncMatrixDescriptor(matrix);
    if ((!descriptor && (!ordinary || ordinary.getIsVolatile() || !dotLayout ||
                         !isa<ttg::AMDMfmaEncodingAttr>(dotLayout.getParent()))) ||
        (descriptor && (!isa<ttg::DistributedEncodingTrait>(layout) ||
                        matrix.getIsVolatile())) ||
        !load ||
        (!descriptor && !load->getParentOfType<ttg::WarpSpecializeOp>()))
      return failure();
    Value pointer = descriptor ? matrix.getBase() : ordinary.getPtr();
    auto ptr = dyn_cast<tt::PointerType>(getElementTypeOrSelf(pointer));
    if (!ptr || ptr.getAddressSpace() != 1 ||
        (!descriptor && !isa<RankedTensorType>(pointer.getType())))
      return failure();
    SmallVector<ttg::ConvertLayoutOp> users;
    for (Operation *user : load->getResult(0).getUsers()) {
      auto other = dyn_cast<ttg::ConvertLayoutOp>(user);
      if (!other || other.getType() != convert.getType())
        return failure();
      users.push_back(other);
    }
    // Only rematerialize pure address/mask/fill computations. In particular,
    // never duplicate a data-dependent gather's index loads or a fence.
    for (OpOperand &operand : load->getOpOperands()) {
      if (!isa<RankedTensorType>(operand.get().getType()))
        continue;
      SetVector<Value> slice;
      DenseMap<Value, Attribute> encodings;
      if (failed(mlir::getConvertBackwardSlice(operand, slice, layout,
                                             encodings)))
        return failure();
      for (Value value : slice)
        if (Operation *op = value.getDefiningOp();
            op && (!isPure(op) || op->getNumRegions()))
          return failure();
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(load);
    SmallVector<Value> operands;
    for (Value value : load->getOperands()) {
      if (auto type = dyn_cast<RankedTensorType>(value.getType()))
        value = rewriter.create<ttg::ConvertLayoutOp>(
            load->getLoc(), type.cloneWithEncoding(layout), value);
      operands.push_back(value);
    }
    OperationState state(load->getLoc(), load->getName(), operands,
                         {convert.getType()}, load->getAttrs());
    Operation *direct = rewriter.create(state);
    for (ttg::ConvertLayoutOp user : users)
      rewriter.replaceOp(user, direct->getResult(0));
    rewriter.eraseOp(load);
    return success();
  }
};


// Infer a closed pure-view tree from the consumers, then change only the SSA
// encodings at the original read. Conflicting fanout or non-view use retains
// the existing path.
class RegisterLoadViews : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp convert,
                               PatternRewriter &rewriter) const override {
    auto isView = [](Operation *op) {
      return isa_and_nonnull<tt::ReshapeOp, tt::TransOp, tt::SplitOp>(op);
    };
    Value root = convert.getSrc();
    unsigned budget = 64;
    while (isView(root.getDefiningOp())) {
      if (!budget--)
        return failure();
      root = root.getDefiningOp()->getOperand(0);
    }
    Operation *read = root.getDefiningOp();
    auto type = dyn_cast<RankedTensorType>(root.getType());
    if (!read || !type || root == convert.getSrc() ||
        !isa<ttg::DistributedEncodingTrait>(type.getEncoding()))
      return failure();
    if (auto load = dyn_cast<tt::MatrixLoadOp>(read)) {
      if (load.getIsVolatile() || !load->hasAttr("hcu.structured_register"))
        return failure();
    } else if (auto load = dyn_cast<ttg::LocalLoadOp>(read)) {
      auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(
          load.getSrc().getType().getEncoding());
      if (!read->getParentOfType<ttg::WarpSpecializeOp>() || !shared ||
          shared.getOrder() != ArrayRef<unsigned>({1, 0}) || load.getToken() ||
          !(type.getElementType().isF16() || type.getElementType().isBF16()))
        return failure();
    } else {
      return failure();
    }
    SetVector<Operation *> views;
    SetVector<ttg::ConvertLayoutOp> leaves;
    DenseSet<Value> visited;
    SmallVector<Value> pending{root};
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      if (!budget--)
        return failure();
      for (Operation *user : value.getUsers()) {
        if (auto leaf = dyn_cast<ttg::ConvertLayoutOp>(user)) {
          auto layout = leaf.getType().getEncoding();
          if (!isa<ttg::DistributedEncodingTrait>(layout))
            return failure();
          leaves.insert(leaf);
        } else if (isView(user)) {
          views.insert(user);
          llvm::append_range(pending, user->getResults());
        } else {
          return failure();
        }
      }
    }
    DenseMap<Value, Attribute> ownership;
    for (ttg::ConvertLayoutOp leaf : leaves) {
      Value value = leaf.getSrc();
      Attribute encoding = ttg::LinearEncodingAttr::get(
          getContext(), ttg::toLinearLayout(leaf.getType()));
      while (true) {
        auto type = cast<RankedTensorType>(value.getType());
        encoding = ttg::LinearEncodingAttr::get(
            getContext(), ttg::toLinearLayout(type.getShape(), encoding));
        auto [it, inserted] = ownership.try_emplace(value, encoding);
        if (!inserted && it->second != encoding)
          return failure();
        if (value == root)
          break;
        Operation *view = value.getDefiningOp();
        encoding = mlir::inferSrcEncoding(view, encoding);
        if (!encoding)
          return failure();
        value = view->getOperand(0);
      }
    }
    Attribute encoding = ownership.lookup(root);
    if (!encoding || encoding == type.getEncoding())
      return failure();
    SmallVector<Operation *> ordered(views.begin(), views.end());
    auto depth = [&](Operation *op) {
      unsigned n = 0;
      for (Value v = op->getOperand(0); v != root;
           v = v.getDefiningOp()->getOperand(0))
        ++n;
      return n;
    };
    llvm::stable_sort(ordered, [&](Operation *a, Operation *b) {
      return depth(a) < depth(b);
    });
    DenseMap<Value, Attribute> forward;
    forward[root] = encoding;
    for (Operation *view : ordered) {
      Attribute output = mlir::inferDstEncoding(
          view, forward.lookup(view->getOperand(0)));
      if (!output)
        return failure();
      for (Value result : view->getResults()) {
        auto type = cast<RankedTensorType>(result.getType());
        if (Attribute required = ownership.lookup(result);
            required && ttg::toLinearLayout(type.getShape(), output) !=
                            ttg::toLinearLayout(type.getShape(), required))
          return failure();
        forward[result] = output;
      }
    }
    rewriter.modifyOpInPlace(read,
                             [&] { root.setType(type.cloneWithEncoding(encoding)); });
    for (Operation *view : ordered)
      rewriter.modifyOpInPlace(view, [&] {
        for (Value result : view->getResults())
          result.setType(cast<RankedTensorType>(result.getType())
                             .cloneWithEncoding(forward.lookup(result)));
      });
    return success();
  }
};

// This pattern creates LocalAllocOp and LocalLoadOp with unswizzled shared
// layout for the scale operand used in ScaledUpcastFp4Op/ScaledUpcastFp8Op.
// StreamPipeliner will respect the layout created here and pipeline ops
// according to the need.
//
// It matches
// tt.load -> ... -> hcug.scaled_upcast_x
//
// And rewrites it to
// tt.load -> ttg.local_alloc -> ttg.local_load -> ... -> hcug.scaled_upcast_x
template <typename OpTy>
class AllocSharedMemForUpcastedScales : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  AllocSharedMemForUpcastedScales(MLIRContext *context,
                                  triton::HCU::ISAFamily isaFamily)
      : OpRewritePattern<OpTy>(context), isaFamily(isaFamily) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (isaFamily != mlir::triton::HCU::ISAFamily::CDNA4)
      return rewriter.notifyMatchFailure(op, "NYI: Only supported on CDNA4");

    auto forOp = op->template getParentOfType<scf::ForOp>();
    if (!forOp)
      return rewriter.notifyMatchFailure(op,
                                         "Don't alloc lds outside for loop");

    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    SetVector<Operation *> slice;
    (void)getBackwardSlice(op.getOperand(1), &slice, options);
    tt::LoadOp loadOp;
    unsigned cnt = 0;
    bool hasAllocatedLDS = false;
    for (auto &op : slice) {
      if (isa<tt::LoadOp>(op)) {
        loadOp = dyn_cast<tt::LoadOp>(op);
        cnt++;
      } else if (isa<ttg::LocalLoadOp>(op)) {
        hasAllocatedLDS = true;
        break;
      }
    }

    if (hasAllocatedLDS)
      return rewriter.notifyMatchFailure(
          op, "There's already lds allocation in the def chain.");

    if (!loadOp || cnt != 1)
      return rewriter.notifyMatchFailure(
          op, "Require exactly 1 load in the def chain.");

    LDBG("Found load of scale: " << loadOp << " for ScaleUpcast: " << op);
    auto srcTy = dyn_cast<RankedTensorType>(loadOp.getType());
    auto sharedOrder = ttg::getOrderForMemory(srcTy);
    auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());

    auto ctx = loadOp.getContext();
    auto attr = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, sharedOrder,
                                                     ctaLayout);
    Location loc = loadOp.getLoc();
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    rewriter.setInsertionPointAfter(loadOp);
    auto alloc = ttg::LocalAllocOp::create(
        rewriter, loc,
        ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(), attr,
                              sharedMemorySpace),
        loadOp.getResult());
    LDBG("Create alloc: " << alloc);

    auto localLoad = ttg::LocalLoadOp::create(rewriter, loc, srcTy, alloc);
    LDBG("Create localload: " << localLoad);

    rewriter.replaceAllUsesExcept(loadOp.getResult(), localLoad, alloc);
    return success();
  }

private:
  triton::HCU::ISAFamily isaFamily;
};
} // namespace

#define GEN_PASS_DEF_TRITONHCUGPUOPTIMIZEDOTOPERANDS
#include "TritonHCUGPUTransforms/Passes.h.inc"

class TritonHCUGPUOptimizeDotOperands
    : public impl::TritonHCUGPUOptimizeDotOperandsBase<
          TritonHCUGPUOptimizeDotOperands> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // gfx936 has no independently usable multi-wave worker rendezvous in
    // this toolchain. A register epilogue is consequently also a legalization
    // for ordinary TL workers, not just the optional whole-kernel tuning.
    // Reuse the existing transformation rather than introducing a second
    // store/layout implementation here.
    if (archGenerationName.getValue() == "gfx936") {
      bool hasPartitions = false;
      m.walk([&](ttg::WarpSpecializeOp) { hasPartitions = true; });
      if (hasPartitions) {
        OpPassManager epilogue;
        epilogue.addPass(mlir::createTritonHCUGPUOptimizeEpilogue());
        if (failed(runPipeline(epilogue, m)))
          return signalPassFailure();
      }
    }
    mlir::RewritePatternSet patterns(context);
    m.walk([&](ttg::LocalAllocOp alloc) {
      if (alloc->hasAttr("hcu.auto_shared_layout"))
        AutoSharedLayout(alloc).run();
    });
    patterns.add<RetargetAccumulatorFence>(context);
    if (archGenerationName.getValue() == "gfx936")
      patterns.add<LoadDotOperandRegisters, RegisterLoadViews,
                   RetargetLoopCarriedDotOperand>(context);
    auto isaFamily = triton::HCU::deduceISAFamily(archGenerationName);
    patterns
        .add<AllocSharedMemForUpcastedScales<tt::hcugpu::ScaledUpcastFp8Op>,
             AllocSharedMemForUpcastedScales<tt::hcugpu::ScaledUpcastFp4Op>>(
            context, isaFamily);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

void registerTritonHCUGPUOptimizeDotOperands() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTritonHCUGPUOptimizeDotOperands();
  });
}

} // namespace mlir::triton::hcugpu
