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


// Descriptor loads have scalar address operands, so ordinary tensor-pointer
// rematerialization cannot retarget them. Infer a closed pure-view tree from
// all dot consumers, then change only SSA encodings at the original read.
// Conflicting fanout or any non-view use retains the existing path.
class RegisterDescriptorViews : public OpRewritePattern<ttg::ConvertLayoutOp> {
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
    auto load = root.getDefiningOp<tt::MatrixLoadOp>();
    if (!load || root == convert.getSrc() || load.getIsVolatile() ||
        !isa<ttg::DistributedEncodingTrait>(load.getType().getEncoding()) ||
        !load->hasAttr("hcu.structured_register"))
      return failure();
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
    if (!encoding || encoding == load.getType().getEncoding())
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
    rewriter.modifyOpInPlace(load, [&] {
      root.setType(load.getType().cloneWithEncoding(encoding));
    });
    for (Operation *view : ordered)
      rewriter.modifyOpInPlace(view, [&] {
        for (Value result : view->getResults())
          result.setType(cast<RankedTensorType>(result.getType())
                             .cloneWithEncoding(forward.lookup(result)));
      });
    return success();
  }
};

// Late-only mode. Reusing the existing binding keeps this experiment a single
// translation unit; the backend consumes the option into this one-shot marker.
constexpr StringLiteral fragmentDotKMarker = "ttg.hcu.internal_fragment_dot_k";

// Moving later B reads across these operations cannot cross an LDS write or
// barrier. Global register reads are disjoint from the AS3 shared source. Do not
// accept generic read effects (which could include shared memory), calls, loops,
// atomics, volatile reads, or conditional branches.
static bool canDelaySharedReadAcross(Operation *op) {
  if (auto load = dyn_cast<tt::LoadOp>(op)) {
    auto ptr = dyn_cast<tt::PointerType>(getElementTypeOrSelf(load.getPtr()));
    return ptr && ptr.getAddressSpace() == 1 && !load.getIsVolatile();
  }
  if (auto load = dyn_cast<tt::hcugpu::BufferLoadOp>(op)) {
    auto ptr = dyn_cast<tt::PointerType>(load.getPtr().getType());
    return ptr && ptr.getAddressSpace() == 1;
  }
  if (auto branch = dyn_cast<scf::IfOp>(op)) {
    for (Region &region : branch->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!isa<scf::YieldOp>(nested) &&
              !canDelaySharedReadAcross(&nested))
            return false;
    return true;
  }
  return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
         isMemoryEffectFree(op);
}

static unsigned registerWords(RankedTensorType type) {
  return llvm::divideCeil(ttg::getTotalElemsPerThread(type) *
                              type.getElementTypeBitWidth(),
                          32u);
}

class FragmentDotK : public OpRewritePattern<tt::DotOp> {
public:
  FragmentDotK(MLIRContext *context, unsigned budget)
      : OpRewritePattern(context), budget(budget) {}

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto a = dyn_cast<RankedTensorType>(dot.getA().getType());
    auto b = dyn_cast<RankedTensorType>(dot.getB().getType());
    auto c = dyn_cast<RankedTensorType>(dot.getType());
    if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
        c.getRank() != 2 || !c.getElementType().isF32() ||
        !(a.getElementType().isBF16() || a.getElementType().isF16()) ||
        a.getElementType() != b.getElementType())
      return failure();
    auto mma = dyn_cast<ttg::AMDMfmaEncodingAttr>(c.getEncoding());
    auto ae = dyn_cast<ttg::DotOperandEncodingAttr>(a.getEncoding());
    auto be = dyn_cast<ttg::DotOperandEncodingAttr>(b.getEncoding());
    if (!mma || !ae || !be || ae.getParent() != mma || be.getParent() != mma ||
        ae.getOpIdx() != 0 || be.getOpIdx() != 1)
      return failure();
    auto load = dot.getB().getDefiningOp<ttg::LocalLoadOp>();
    auto arg = dyn_cast<BlockArgument>(dot.getA());
    auto loop = arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())
                    : scf::ForOp();
    // Full A is carried across iterations. Existing extracts/subslices are
    // deliberately idempotent: this pattern does not fragment an existing chain.
    if (!load || load.getToken() || !load->hasOneUse() || !arg || !loop || arg.getArgNumber() == 0 ||
        loop.getBody() != dot->getBlock() || load->getBlock() != dot->getBlock() ||
        !load->isBeforeInBlock(dot) ||
        load.getSrc().getDefiningOp<ttg::MemDescSubsliceOp>())
      return failure();
    auto source = load.getSrc().getType();
    if (source.getRank() != 2 || source.getShape() != b.getShape() ||
        !isa<ttg::SharedEncodingTrait>(source.getEncoding()))
      return failure();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    Value futureA = yield.getOperand(arg.getArgNumber() - 1);
    if (futureA == arg || futureA.getType() != a)
      return failure();
    for (Operation *op = load->getNextNode(); op != dot.getOperation();
         op = op->getNextNode())
      if (!op || !canDelaySharedReadAcross(op))
        return failure();

    int64_t k = a.getDimSize(1);
    if (k != b.getDimSize(0) || !llvm::isPowerOf2_64(k))
      return failure();
    unsigned fixed = registerWords(c) + 2 * registerWords(a);
    if (fixed + registerWords(b) < budget)
      return failure();
    int64_t instrK = mma.getInstrShape()[2];
    int64_t fragment = 0;
    // Test each legal split with the existing exact layout verifiers. A failed
    // trial only creates detached operations, never partially mutates the IR.
    for (int64_t f = k / 2; f >= instrK; f /= 2) {
      if (f % instrK)
        continue;
      auto bt = RankedTensorType::get({f, b.getDimSize(1)},
                                      b.getElementType(), b.getEncoding());
      if (fixed + registerWords(bt) >= budget)
        continue;
      auto at = RankedTensorType::get({a.getDimSize(0), f},
                                      a.getElementType(), a.getEncoding());
      auto mt = ttg::MemDescType::get(
          {f, source.getDimSize(1)}, source.getElementType(),
          source.getEncoding(), source.getMemorySpace(),
          source.getMutableMemory(), source.getAllocShape());
      Block probe;
      OpBuilder trial(getContext());
      trial.setInsertionPointToEnd(&probe);
      bool legal = true;
      ScopedDiagnosticHandler silence(getContext(), [](Diagnostic &) {
        return success();
      });
      for (int64_t off = 0; off < k && legal; off += f) {
        auto slice = tt::hcugpu::ExtractSliceOp::create(
            trial, dot.getLoc(), Type(at), Value(arg),
            trial.getDenseI64ArrayAttr({0, off}));
        auto view = ttg::MemDescSubsliceOp::create(
            trial, dot.getLoc(), mt, load.getSrc(),
            ArrayRef<int32_t>({static_cast<int32_t>(off), 0}));
        legal = succeeded(slice.verify()) && succeeded(view.verify());
      }
      if (legal) {
        fragment = f;
        break;
      }
    }
    if (!fragment)
      return failure();

    auto bt = RankedTensorType::get({fragment, b.getDimSize(1)},
                                    b.getElementType(), b.getEncoding());
    auto at = RankedTensorType::get({a.getDimSize(0), fragment},
                                    a.getElementType(), a.getEncoding());
    auto mt = ttg::MemDescType::get(
        {fragment, source.getDimSize(1)}, source.getElementType(),
        source.getEncoding(), source.getMemorySpace(),
        source.getMutableMemory(), source.getAllocShape());
    Value acc = dot.getC();
    for (int64_t off = 0; off < k; off += fragment) {
      // Keep the first read at its original point (before A-next prefetch),
      // and issue later reads after the preceding MMA consumes its fragment.
      if (off == 0)
        rewriter.setInsertionPoint(load);
      auto view = ttg::MemDescSubsliceOp::create(
          rewriter, dot.getLoc(), mt, load.getSrc(),
          ArrayRef<int32_t>({static_cast<int32_t>(off), 0}));
      auto partB = ttg::LocalLoadOp::create(rewriter, load.getLoc(), bt, view);
      partB->setAttrs(load->getAttrs());
      if (off == 0)
        rewriter.setInsertionPoint(dot);
      auto partA = tt::hcugpu::ExtractSliceOp::create(
          rewriter, dot.getLoc(), Type(at), Value(arg),
          rewriter.getDenseI64ArrayAttr({0, off}));
      auto partDot = tt::DotOp::create(
          rewriter, dot.getLoc(), c, partA, partB, acc,
          dot.getInputPrecision(), dot.getMaxNumImpreciseAcc());
      partDot->setAttrs(dot->getAttrs());
      acc = partDot;
    }
    rewriter.replaceOp(dot, acc);
    rewriter.eraseOp(load);
    return success();
  }
private:
  unsigned budget;
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

    if (auto marker = m->getAttrOfType<IntegerAttr>(fragmentDotKMarker)) {
      unsigned budget = marker.getInt();
      m->removeAttr(fragmentDotKMarker);
      if (budget != 256) {
        m.emitError("fragment_dot_k: invalid target register budget");
        return signalPassFailure();
      }
      SmallVector<tt::DotOp> dots;
      m.walk([&](tt::DotOp dot) { dots.push_back(dot); });
      PatternRewriter rewriter(context);
      FragmentDotK pattern(context, budget);
      for (tt::DotOp dot : dots)
        (void)pattern.matchAndRewrite(dot, rewriter);
      return;
    }
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
      patterns.add<LoadDotOperandRegisters, RegisterDescriptorViews,
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
