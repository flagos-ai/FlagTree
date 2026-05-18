#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#ifdef __TLE__
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#endif
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <memory>

namespace mlir::triton::gpu {

namespace {

#ifdef __TLE__
static Value getWarpSpecializeCapture(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return value;

  Block *block = blockArg.getOwner();
  auto partitions =
      dyn_cast_or_null<WarpSpecializePartitionsOp>(block->getParentOp());
  if (!partitions)
    return value;

  auto wsOp = dyn_cast<WarpSpecializeOp>(partitions->getParentOp());
  if (!wsOp)
    return value;

  unsigned argNo = blockArg.getArgNumber();
  OperandRange captures = wsOp.getExplicitCaptures();
  if (argNo >= captures.size())
    return value;
  return captures[argNo];
}

static Value stripMemDescViews(Value value) {
  while (true) {
    value = getWarpSpecializeCapture(value);
    if (auto index = value.getDefiningOp<MemDescIndexOp>()) {
      value = index.getSrc();
      continue;
    }
    if (auto subslice = value.getDefiningOp<MemDescSubsliceOp>()) {
      value = subslice.getSrc();
      continue;
    }
    if (auto trans = value.getDefiningOp<MemDescTransOp>()) {
      value = trans.getSrc();
      continue;
    }
    if (auto reshape = value.getDefiningOp<MemDescReshapeOp>()) {
      value = reshape.getSrc();
      continue;
    }
    if (auto reinterpret = value.getDefiningOp<MemDescReinterpretOp>()) {
      value = reinterpret.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<triton::tle::MemDescWGMMAViewOp>()) {
      value = view.getSrc();
      continue;
    }
    return value;
  }
}

static bool isBackedByLocalAlloc(Value value, llvm::DenseSet<Value> &active);

static bool isForIterArgBackedByLocalAlloc(BlockArgument arg,
                                           llvm::DenseSet<Value> &active) {
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp)
    return false;

  unsigned argNo = arg.getArgNumber();
  if (argNo == 0)
    return false;
  unsigned iterIdx = argNo - 1;
  if (iterIdx >= forOp.getInitArgs().size())
    return false;

  if (!isBackedByLocalAlloc(forOp.getInitArgs()[iterIdx], active))
    return false;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || iterIdx >= yield.getNumOperands())
    return false;

  return isBackedByLocalAlloc(yield.getOperand(iterIdx), active);
}

static bool isForResultBackedByLocalAlloc(scf::ForOp forOp, unsigned resultIdx,
                                          llvm::DenseSet<Value> &active) {
  if (resultIdx >= forOp.getInitArgs().size())
    return false;
  if (!isBackedByLocalAlloc(forOp.getInitArgs()[resultIdx], active))
    return false;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || resultIdx >= yield.getNumOperands())
    return false;

  return isBackedByLocalAlloc(yield.getOperand(resultIdx), active);
}

static bool isIfResultBackedByLocalAlloc(scf::IfOp ifOp, unsigned resultIdx,
                                         llvm::DenseSet<Value> &active) {
  if (resultIdx >= ifOp.thenYield().getNumOperands() ||
      resultIdx >= ifOp.elseYield().getNumOperands())
    return false;

  return isBackedByLocalAlloc(ifOp.thenYield().getOperand(resultIdx), active) &&
         isBackedByLocalAlloc(ifOp.elseYield().getOperand(resultIdx), active);
}

static bool isBackedByLocalAlloc(Value value, llvm::DenseSet<Value> &active) {
  value = stripMemDescViews(value);
  if (!active.insert(value).second)
    return true;

  if (value.getDefiningOp<LocalAllocOp>())
    return true;

  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return isBackedByLocalAlloc(select.getTrueValue(), active) &&
           isBackedByLocalAlloc(select.getFalseValue(), active);

  if (auto arg = dyn_cast<BlockArgument>(value))
    return isForIterArgBackedByLocalAlloc(arg, active);

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;

  if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner()))
    return isForResultBackedByLocalAlloc(forOp, result.getResultNumber(),
                                         active);
  if (auto ifOp = dyn_cast<scf::IfOp>(result.getOwner()))
    return isIfResultBackedByLocalAlloc(ifOp, result.getResultNumber(), active);

  return false;
}

static bool isBackedByLocalAlloc(Value value) {
  llvm::DenseSet<Value> active;
  return isBackedByLocalAlloc(value, active);
}

static void
preserveWarpGroupDotAttrs(triton::nvidia_gpu::WarpGroupDotOp oldDot,
                          triton::nvidia_gpu::WarpGroupDotOp newDot) {
  newDot->setAttrs(oldDot->getAttrDictionary());
}

static SmallVector<int64_t> getPermutedAllocShape(MemDescType srcTy,
                                                  ArrayRef<int32_t> order) {
  SmallVector<int64_t> allocShape =
      applyPermutation(srcTy.getAllocShape().take_back(order.size()), order);
  allocShape.insert(allocShape.begin(), srcTy.getAllocShape().begin(),
                    srcTy.getAllocShape().end() - order.size());
  return allocShape;
}
#endif

// Given
//   dot(convert(trans(src)) #dot_operand) ->
//   dot(convert(local_load(trans(alloc(src)))))
// change the encoding of the inner convert to a special, swizzled shared
// encoding.
class SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    if (!cvtOp->hasOneUse() ||
        !isa<triton::DotOp>(cvtOp->use_begin()->getOwner()))
      return failure();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = cvtOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();

    RankedTensorType srcTy = trans.getSrc().getType();

    if (auto srcCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      srcTy = srcCvt.getSrc().getType();
    }
    RankedTensorType sharedLoadTy = cvtOp.getType();
    auto cvtEncoding =
        dyn_cast<DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
    if (!cvtEncoding)
      return failure();

    // Set needTrans to true here. newInnerCvtEnc is computed based on
    // argEncoding which is before the transpose. Without needTrans we will
    // compute vec and maxPhase based on incorrect m, n and k size of mma. The
    // type inference of MemDescTransOp simply swap the order but doesn't fix
    // the vec and maxPhase for the YType, hence it would causing incorrect
    // swizzling code.
    auto ctx = getContext();
    auto oldCTALayout = triton::gpu::getCTALayout(srcTy.getEncoding());
    auto newLl =
        transposeLinearLayout(oldCTALayout.getLinearLayout(), trans.getOrder());
    auto newCTALayout = CTAEncodingAttr::get(ctx, std::move(newLl));
    auto newInnerCvtEnc =
        SwizzledSharedEncodingAttr::get(ctx, cvtEncoding, srcTy.getShape(),
                                        /*order=*/getOrderForMemory(srcTy),
                                        newCTALayout, srcTy.getElementType(),
                                        /*needTrans=*/true);
    if (newInnerCvtEnc == cvtEncoding)
      return failure();
    rewriter.setInsertionPoint(trans);
#ifdef __TLE__
    // If the source is already loaded from a shared memdesc allocated in this
    // function, directly retag that memdesc encoding and avoid inserting an
    // additional local_alloc staging buffer.
    if (auto srcLocalLoad = trans.getSrc().getDefiningOp<LocalLoadOp>()) {
      Value srcMemDesc = srcLocalLoad.getSrc();
      auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
      if (srcMemDescTy && srcMemDescTy.getShape() == srcTy.getShape() &&
          srcMemDescTy.getElementType() == srcTy.getElementType() &&
          isBackedByLocalAlloc(srcMemDesc)) {
        auto updatedMemDescTy = MemDescType::get(
            srcMemDescTy.getShape(), srcMemDescTy.getElementType(),
            newInnerCvtEnc, srcMemDescTy.getMemorySpace(),
            srcMemDescTy.getMutableMemory(), srcMemDescTy.getAllocShape());
        srcMemDesc.setType(updatedMemDescTy);
        auto newTrans = rewriter.create<MemDescTransOp>(
            trans.getLoc(), srcMemDesc, ArrayRef<int32_t>({1, 0}));
        auto localLoadOp = rewriter.create<LocalLoadOp>(
            trans.getLoc(), sharedLoadTy, newTrans, srcLocalLoad.getToken());
        rewriter.modifyOpInPlace(cvtOp, [&]() {
          cvtOp.getSrcMutable().assign(localLoadOp.getResult());
        });
        return success();
      }
    }
#endif
    auto sharedMemorySpace = SharedMemorySpaceAttr::get(getContext());
    auto alloc = LocalAllocOp::create(
        rewriter, trans.getLoc(),
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                         newInnerCvtEnc, sharedMemorySpace),
        trans.getSrc());
    auto newTrans = MemDescTransOp::create(rewriter, trans.getLoc(), alloc,
                                           ArrayRef<int32_t>({1, 0}));
    auto localLoadOp =
        LocalLoadOp::create(rewriter, trans.getLoc(), sharedLoadTy, newTrans);
    rewriter.modifyOpInPlace(cvtOp, [&]() {
      cvtOp.getSrcMutable().assign(localLoadOp.getResult());
    });
    return success();
  }
};

// Rewrite
//
//   dot(alloc(trans() #shared1) ->
//   dot(trans(alloc() #shared2))
//
// if dot is an MMAv3/v5 (because MMAv3/v5 allows us to fold transposes).
class FuseTransMMAV3Plus : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc() || !allocOp->hasOneUse() ||
        !isa<triton::nvidia_gpu::WarpGroupDotOp,
             triton::nvidia_gpu::MMAv5OpInterface>(
            *allocOp->getUsers().begin()))
      return failure();

    auto dot = *allocOp->getUsers().begin();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding = cast<NVMMASharedEncodingAttr>(allocType.getEncoding());
    RankedTensorType srcTy = trans.getSrc().getType();

#ifdef __TLE__
    auto srcLocalLoad = trans.getSrc().getDefiningOp<LocalLoadOp>();
    if (srcLocalLoad && !srcLocalLoad.getToken()) {
      Value srcMemDesc = srcLocalLoad.getSrc();
      auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
      if (srcMemDescTy && srcMemDescTy.getShape() == srcTy.getShape() &&
          srcMemDescTy.getElementType() == srcTy.getElementType() &&
          isBackedByLocalAlloc(srcMemDesc)) {
        Attribute viewEncoding;
        Dialect &srcDialect = srcMemDescTy.getEncoding().getDialect();
        auto srcInferLayoutInterface =
            cast<DialectInferLayoutInterface>(&srcDialect);
        if (failed(srcInferLayoutInterface->inferTransOpEncoding(
                srcMemDescTy.getEncoding(), srcMemDescTy.getShape(),
                trans.getOrder(), viewEncoding, allocOp.getLoc()))) {
          return failure();
        }

        auto viewShape =
            applyPermutation(srcMemDescTy.getShape(), trans.getOrder());
        SmallVector<int64_t> viewAllocShape =
            getPermutedAllocShape(srcMemDescTy, trans.getOrder());
        auto viewType =
            MemDescType::get(viewShape, srcMemDescTy.getElementType(),
                             viewEncoding, srcMemDescTy.getMemorySpace(),
                             srcMemDescTy.getMutableMemory(), viewAllocShape);
        rewriter.getContext()->getOrLoadDialect<triton::tle::TleDialect>();
        rewriter.replaceOpWithNewOp<triton::tle::MemDescWGMMAViewOp>(
            allocOp, viewType, srcMemDesc, ArrayRef<int32_t>({1, 0}));
        return success();
      }
    }
#endif

    auto ctx = getContext();
    Dialect &dialect = allocEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    Attribute newInnerEnc;
    if (failed(inferLayoutInterface->inferTransOpEncoding(
            allocEncoding, srcTy.getShape(), trans.getOrder(), newInnerEnc,
            allocOp.getLoc()))) {
      return failure();
    }

    MemDescType innerTy =
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(), newInnerEnc,
                         allocType.getMemorySpace());
    auto newAlloc = LocalAllocOp::create(rewriter, allocOp.getLoc(), innerTy,
                                         trans.getSrc());
    rewriter.replaceOpWithNewOp<MemDescTransOp>(allocOp, newAlloc,
                                                ArrayRef<int32_t>({1, 0}));
    return success();
  }
};

#ifdef __TLE__
// Rewrite
//
//   memdesc_trans(local_alloc(local_load(existing_smem)))
//
// to
//
//   tle.memdesc_wgmma_view(existing_smem)
//
// This is the canonicalized form of local_alloc(trans(local_load(...))).
class FuseCanonicalizedTransMMAV3Plus
    : public OpRewritePattern<MemDescTransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemDescTransOp transOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = transOp.getSrc().getDefiningOp<LocalAllocOp>();
    if (!allocOp || !allocOp.getSrc())
      return failure();

    auto localLoad = allocOp.getSrc().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    auto transTy = cast<MemDescType>(transOp.getType());
    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(
            transTy.getEncoding()))
      return failure();

    Value srcMemDesc = localLoad.getSrc();
    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
    if (!srcMemDescTy || !localLoadTy)
      return failure();

    if (srcMemDescTy.getShape() != localLoadTy.getShape() ||
        srcMemDescTy.getElementType() != localLoadTy.getElementType())
      return failure();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    // The descriptor view models `trans(local_load(srcMemDesc))` directly over
    // the original slot.  Do not inherit the canonical staging trans result
    // encoding: AccelerateMatmul may represent the staging path as a
    // transposed local_alloc followed by a memdesc_trans back to the logical
    // dot shape, while the WGMMA descriptor must keep the source-derived
    // transposed shared encoding.
    Attribute viewEncoding;
    Dialect &srcDialect = srcMemDescTy.getEncoding().getDialect();
    auto srcInferLayoutInterface =
        cast<DialectInferLayoutInterface>(&srcDialect);
    if (failed(srcInferLayoutInterface->inferTransOpEncoding(
            srcMemDescTy.getEncoding(), srcMemDescTy.getShape(),
            transOp.getOrder(), viewEncoding, transOp.getLoc())))
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(viewEncoding))
      return failure();

    auto viewShape =
        applyPermutation(srcMemDescTy.getShape(), transOp.getOrder());
    auto viewTy = MemDescType::get(
        viewShape, srcMemDescTy.getElementType(), viewEncoding,
        srcMemDescTy.getMemorySpace(), srcMemDescTy.getMutableMemory(),
        getPermutedAllocShape(srcMemDescTy, transOp.getOrder()));
    rewriter.getContext()->getOrLoadDialect<triton::tle::TleDialect>();
    rewriter.setInsertionPoint(transOp);
    auto view = triton::tle::MemDescWGMMAViewOp::create(
        rewriter, transOp.getLoc(), viewTy, srcMemDesc, transOp.getOrder());
    (void)view;
    mlir::triton::replaceUsesAndPropagateType(rewriter, transOp, view);
    if (transOp->use_empty())
      rewriter.eraseOp(transOp);
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    if (localLoad->use_empty())
      rewriter.eraseOp(localLoad);
    return success();
  }
};

// Rewrite
//
//   warp_group_dot(local_alloc(trans(local_load(existing_smem))), b, acc)
//
// to
//
//   warp_group_dot(tle.memdesc_wgmma_view(existing_smem), b, acc)
//
// and propagate the view through any wait operands. This catches pipelined
// WGMMA A/B operands whose staging alloc has more than one use.
class ReuseTransposedLocalLoadAllocAsWGMMAOperand
    : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();

    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    auto localLoad = trans.getSrc().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    Value srcMemDesc = localLoad.getSrc();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
    auto allocTy = dyn_cast<MemDescType>(allocOp.getType());
    if (!srcMemDescTy || !localLoadTy || !allocTy)
      return failure();

    if (srcMemDescTy.getShape() != localLoadTy.getShape() ||
        srcMemDescTy.getElementType() != localLoadTy.getElementType())
      return failure();

    auto transposedShape =
        applyPermutation(srcMemDescTy.getShape(), trans.getOrder());
    if (allocTy.getShape() != ArrayRef<int64_t>(transposedShape) ||
        allocTy.getElementType() != srcMemDescTy.getElementType() ||
        allocTy.getMemorySpace() != srcMemDescTy.getMemorySpace())
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(
            allocTy.getEncoding()))
      return failure();

    if (!llvm::all_of(allocOp->getUsers(), [](Operation *user) {
          return isa<triton::nvidia_gpu::WarpGroupDotOp,
                     triton::nvidia_gpu::WarpGroupDotWaitOp>(user);
        }))
      return failure();

    Attribute viewEncoding;
    Dialect &srcDialect = srcMemDescTy.getEncoding().getDialect();
    auto srcInferLayoutInterface =
        cast<DialectInferLayoutInterface>(&srcDialect);
    if (failed(srcInferLayoutInterface->inferTransOpEncoding(
            srcMemDescTy.getEncoding(), srcMemDescTy.getShape(),
            trans.getOrder(), viewEncoding, allocOp.getLoc())))
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(viewEncoding))
      return failure();

    rewriter.getContext()->getOrLoadDialect<triton::tle::TleDialect>();
    rewriter.setInsertionPoint(allocOp);
    auto viewTy = MemDescType::get(
        transposedShape, srcMemDescTy.getElementType(), viewEncoding,
        srcMemDescTy.getMemorySpace(), srcMemDescTy.getMutableMemory(),
        getPermutedAllocShape(srcMemDescTy, trans.getOrder()));
    auto view = triton::tle::MemDescWGMMAViewOp::create(
        rewriter, allocOp.getLoc(), viewTy, srcMemDesc, trans.getOrder());
    mlir::triton::replaceUsesAndPropagateType(rewriter, allocOp, view);

    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    if (trans->use_empty())
      rewriter.eraseOp(trans);
    if (localLoad->use_empty())
      rewriter.eraseOp(localLoad);
    return success();
  }
};

// Rewrite
//
//   warp_group_dot(convert_layout(trans(local_load(existing_smem))), b, acc)
//
// to
//
//   warp_group_dot(tle.memdesc_wgmma_view(existing_smem), b, acc)
//
// before the transposed tensor is materialized into a WGMMA staging allocation.
class ReuseTransposedLocalLoadConvertAsWGMMAA
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto cvt = dotOp.getA().getDefiningOp<ConvertLayoutOp>();
    if (!cvt)
      return failure();

    auto trans = cvt.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    auto localLoad = trans.getSrc().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    Value srcMemDesc = localLoad.getSrc();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
    auto cvtTy = dyn_cast<RankedTensorType>(cvt.getType());
    if (!srcMemDescTy || !localLoadTy || !cvtTy)
      return failure();

    if (srcMemDescTy.getShape() != localLoadTy.getShape() ||
        srcMemDescTy.getElementType() != localLoadTy.getElementType())
      return failure();

    auto transposedShape =
        applyPermutation(srcMemDescTy.getShape(), trans.getOrder());
    if (cvtTy.getShape() != ArrayRef<int64_t>(transposedShape) ||
        cvtTy.getElementType() != srcMemDescTy.getElementType())
      return failure();

    Attribute viewEncoding;
    Dialect &srcDialect = srcMemDescTy.getEncoding().getDialect();
    auto srcInferLayoutInterface =
        cast<DialectInferLayoutInterface>(&srcDialect);
    if (failed(srcInferLayoutInterface->inferTransOpEncoding(
            srcMemDescTy.getEncoding(), srcMemDescTy.getShape(),
            trans.getOrder(), viewEncoding, dotOp.getLoc())))
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(viewEncoding))
      return failure();

    rewriter.getContext()->getOrLoadDialect<triton::tle::TleDialect>();
    rewriter.setInsertionPoint(dotOp);
    auto viewTy = MemDescType::get(
        transposedShape, srcMemDescTy.getElementType(), viewEncoding,
        srcMemDescTy.getMemorySpace(), srcMemDescTy.getMutableMemory(),
        getPermutedAllocShape(srcMemDescTy, trans.getOrder()));
    auto view = triton::tle::MemDescWGMMAViewOp::create(
        rewriter, dotOp.getLoc(), viewTy, srcMemDesc, trans.getOrder());
    auto newDot = triton::nvidia_gpu::WarpGroupDotOp::create(
        rewriter, dotOp.getLoc(), dotOp.getD().getType(), view, dotOp.getB(),
        dotOp.getC(), dotOp.getUseC(), dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc(), dotOp.getIsAsync());
    preserveWarpGroupDotAttrs(dotOp, newDot);
    rewriter.replaceOp(dotOp, newDot.getD());

    if (cvt->use_empty())
      rewriter.eraseOp(cvt);
    if (trans->use_empty())
      rewriter.eraseOp(trans);
    if (localLoad->use_empty())
      rewriter.eraseOp(localLoad);
    return success();
  }
};

// Rewrite
//
//   warp_group_dot(
//     local_load(local_alloc(trans(local_load(existing_smem)))),
//     b,
//     acc)
//
// to
//
//   warp_group_dot(tle.memdesc_wgmma_view(existing_smem), b, acc)
//
// AccelerateMatmul may materialize transposed WGMMA A through a shared staging
// alloc plus a register local_load. WGMMA A accepts shared descriptors, so keep
// the original slot and expose only a descriptor view.
class ReuseTransposedLocalLoadAllocAsWGMMAA
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto localLoad = dotOp.getA().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    auto allocOp = localLoad.getSrc().getDefiningOp<LocalAllocOp>();
    if (!allocOp || !allocOp.getSrc())
      return failure();

    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    auto srcLocalLoad = trans.getSrc().getDefiningOp<LocalLoadOp>();
    if (!srcLocalLoad || srcLocalLoad.getToken())
      return failure();

    Value srcMemDesc = srcLocalLoad.getSrc();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto srcLocalLoadTy = dyn_cast<RankedTensorType>(srcLocalLoad.getType());
    auto allocTy = dyn_cast<MemDescType>(allocOp.getType());
    auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
    if (!srcMemDescTy || !srcLocalLoadTy || !allocTy || !localLoadTy)
      return failure();

    if (srcMemDescTy.getShape() != srcLocalLoadTy.getShape() ||
        srcMemDescTy.getElementType() != srcLocalLoadTy.getElementType())
      return failure();

    auto transposedShape =
        applyPermutation(srcMemDescTy.getShape(), trans.getOrder());
    if (allocTy.getShape() != ArrayRef<int64_t>(transposedShape) ||
        localLoadTy.getShape() != ArrayRef<int64_t>(transposedShape) ||
        allocTy.getElementType() != srcMemDescTy.getElementType() ||
        localLoadTy.getElementType() != srcMemDescTy.getElementType() ||
        allocTy.getMemorySpace() != srcMemDescTy.getMemorySpace())
      return failure();

    Attribute viewEncoding;
    Dialect &srcDialect = srcMemDescTy.getEncoding().getDialect();
    auto srcInferLayoutInterface =
        cast<DialectInferLayoutInterface>(&srcDialect);
    if (failed(srcInferLayoutInterface->inferTransOpEncoding(
            srcMemDescTy.getEncoding(), srcMemDescTy.getShape(),
            trans.getOrder(), viewEncoding, allocOp.getLoc())))
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(viewEncoding))
      return failure();

    rewriter.getContext()->getOrLoadDialect<triton::tle::TleDialect>();
    rewriter.setInsertionPoint(dotOp);
    auto viewTy = MemDescType::get(
        transposedShape, srcMemDescTy.getElementType(), viewEncoding,
        srcMemDescTy.getMemorySpace(), srcMemDescTy.getMutableMemory(),
        getPermutedAllocShape(srcMemDescTy, trans.getOrder()));
    auto view = triton::tle::MemDescWGMMAViewOp::create(
        rewriter, allocOp.getLoc(), viewTy, srcMemDesc, trans.getOrder());
    auto newDot = triton::nvidia_gpu::WarpGroupDotOp::create(
        rewriter, dotOp.getLoc(), dotOp.getD().getType(), view, dotOp.getB(),
        dotOp.getC(), dotOp.getUseC(), dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc(), dotOp.getIsAsync());
    preserveWarpGroupDotAttrs(dotOp, newDot);
    rewriter.replaceOp(dotOp, newDot.getD());

    if (localLoad->use_empty())
      rewriter.eraseOp(localLoad);
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    if (trans->use_empty())
      rewriter.eraseOp(trans);
    if (srcLocalLoad->use_empty())
      rewriter.eraseOp(srcLocalLoad);
    return success();
  }
};

// Rewrite
//
//   warp_group_dot(local_load(existing_smem), b, acc)
//
// to
//
//   warp_group_dot(existing_smem, b, acc)
//
// WGMMA supports A in shared memory as well as registers. This is a
// descriptor-only reuse for TLE kernels that already staged the full A tile in
// shared memory.
class ReuseLocalLoadAsWGMMAA
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto localLoad = dotOp.getA().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    Value srcMemDesc = localLoad.getSrc();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
    if (!srcMemDescTy || !localLoadTy)
      return failure();

    if (srcMemDescTy.getShape() != localLoadTy.getShape() ||
        srcMemDescTy.getElementType() != localLoadTy.getElementType())
      return failure();

    if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(
            srcMemDescTy.getEncoding()))
      return failure();

    auto newDot = triton::nvidia_gpu::WarpGroupDotOp::create(
        rewriter, dotOp.getLoc(), dotOp.getD().getType(), srcMemDesc,
        dotOp.getB(), dotOp.getC(), dotOp.getUseC(), dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc(), dotOp.getIsAsync());
    preserveWarpGroupDotAttrs(dotOp, newDot);
    rewriter.replaceOp(dotOp, newDot.getD());
    return success();
  }
};

// Rewrite
//
//   warp_group_dot(a, local_alloc(local_load(existing_smem)), acc)
//
// to
//
//   warp_group_dot(a, existing_smem, acc)
//
// This catches WGMMA B operands that were materialized before pipe lowering
// could expose the underlying local_alloc-backed pipe slot.
class ReuseLocalLoadAllocAsWGMMAB
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = dotOp.getB().getDefiningOp<LocalAllocOp>();
    if (!allocOp || !allocOp.getSrc())
      return failure();

    auto localLoad = allocOp.getSrc().getDefiningOp<LocalLoadOp>();
    if (!localLoad || localLoad.getToken())
      return failure();

    Value srcMemDesc = localLoad.getSrc();
    if (!isBackedByLocalAlloc(srcMemDesc))
      return failure();

    auto srcMemDescTy = dyn_cast<MemDescType>(srcMemDesc.getType());
    auto allocTy = dyn_cast<MemDescType>(allocOp.getType());
    if (!srcMemDescTy || !allocTy)
      return failure();

    if (srcMemDescTy.getShape() != allocTy.getShape() ||
        srcMemDescTy.getElementType() != allocTy.getElementType() ||
        srcMemDescTy.getMemorySpace() != allocTy.getMemorySpace() ||
        srcMemDescTy.getEncoding() != allocTy.getEncoding())
      return failure();

    if (!llvm::all_of(allocOp->getUsers(), [](Operation *user) {
          return isa<triton::nvidia_gpu::WarpGroupDotOp,
                     triton::nvidia_gpu::WarpGroupDotWaitOp>(user);
        }))
      return failure();

    mlir::triton::replaceUsesAndPropagateType(rewriter, allocOp, srcMemDesc);
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    if (localLoad->use_empty())
      rewriter.eraseOp(localLoad);
    return success();
  }
};
#endif

// Rewrite
//
//   alloc(reshape(), #shared1) ->
//   memdesc_reshape(alloc() #shared2))
//
// if dot is an MMAv3/v5 (because MMAv3/v5 allows us to fold transposes).
class ReshapeMemDesc : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();

    auto reshapeOp = allocOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding = allocType.getEncoding();

    RankedTensorType srcTy = reshapeOp.getSrc().getType();
    auto srcShape = srcTy.getShape();
    auto dstShape = allocType.getShape();

    // We use the fact that forward and backward inference are the same for
    // MemDescReshapeOp to infer the source MemDescType that would produce
    // `allocType` after a reshape.
    MemDescType innerTy;
    if (failed(MemDescReshapeOp::inferReturnTypes(
            getContext(), allocOp.getLoc(), allocType, srcShape, innerTy)))
      return failure();

    // For now don't apply the transformation if the new encoding is not an
    // MMAv3/v5 encoding as it may not be compatible with the user.
    // The heuristic can be refined once we have more flexible mma ops.
    if (!isa<NVMMASharedEncodingAttr>(innerTy.getEncoding()))
      return failure();

    auto newAlloc = LocalAllocOp::create(rewriter, allocOp.getLoc(), innerTy,
                                         reshapeOp.getSrc());
    rewriter.replaceOpWithNewOp<MemDescReshapeOp>(allocOp, allocOp.getType(),
                                                  newAlloc);
    return success();
  }
};

// Inject TMEM copy instructions into IR to efficiently load blocked scales for
// scaled dot
class UseShmemForScales
    : public OpRewritePattern<triton::nvidia_gpu::TCGen5MMAScaledOp> {
public:
  using OpRewritePattern<
      triton::nvidia_gpu::TCGen5MMAScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    auto aScale = mmaOp.getAScale();
    auto bScale = mmaOp.getBScale();
    LogicalResult ret = failure();
    if (aScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      aScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getAScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    if (bScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      bScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getBScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    return ret;
  }

private:
  LogicalResult rewriteOperand(OpOperand &opOperand,
                               PatternRewriter &rewriter) const {
    auto src = cast<TypedValue<MemDescType>>(opOperand.get());
    auto tmemAlloc = src.getDefiningOp<triton::nvidia_gpu::TMEMAllocOp>();
    if (!tmemAlloc) {
      return failure();
    }
    auto dstType = tmemAlloc.getResult().getType();

    if (!tmemAlloc.getSrc()) {
      return failure();
    }

    // Look for a sequence
    //    local_load
    // -> reshape(..., (BLOCK_MN / 128, BLOCK_K / scale_vec_size / 4, 32, 4,
    // 4)
    // -> transpose(..., (0, 3, 2, 1, 4))
    // -> reshape(..., (BLOCK_MN, BLOCK_K / scale_vec_size)
    // -> tmem_alloc
    // -> tc_gen_mma_scaled
    // and replace it with local_alloc -> tc_gen_mma_scaled
    auto scale2DShape = dstType.getShape();
    auto blockMN = scale2DShape[0];
    auto numScales = scale2DShape[1];
    const SmallVector<int> transposeOrder{0, 3, 2, 1, 4};
    const SmallVector<int64_t> reshape5DShape{blockMN / 128, numScales / 4, 32,
                                              4, 4};

    auto reshapeOp2D = getNextOp<triton::ReshapeOp>(tmemAlloc.getSrc());
    if (!reshapeOp2D ||
        reshapeOp2D.getResult().getType().getShape() != scale2DShape) {
      return failure();
    }

    auto transOp = getNextOp<triton::TransOp>(reshapeOp2D.getSrc());
    if (!transOp || transOp.getOrder() != ArrayRef<int>(transposeOrder)) {
      return failure();
    }

    auto reshapeOp5D = getNextOp<triton::ReshapeOp>(transOp.getSrc());
    if (!reshapeOp5D || reshapeOp5D.getResult().getType().getShape() !=
                            ArrayRef<int64_t>(reshape5DShape)) {
      return failure();
    }

    auto localLoad = getNextOp<triton::gpu::LocalLoadOp>(reshapeOp5D.getSrc());
    if (!localLoad) {
      return failure();
    }
    auto localAlloc = getNextOp<LocalAllocOp>(localLoad.getSrc());
    bool usesTMAload =
        (localAlloc && localAlloc.getSrc() &&
         (getNextOp<DescriptorLoadOp>(localAlloc.getSrc()) != nullptr));
    if (!isTmemCopyCompatible(localLoad.getSrc().getType(), usesTMAload))
      return failure();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(tmemAlloc);

    Value shared = localLoad.getSrc();

    Value reshaped5D = MemDescReshapeOp::create(rewriter, reshapeOp5D.getLoc(),
                                                shared, reshape5DShape);
    SmallVector<int32_t> transposeOrder32(transposeOrder.begin(),
                                          transposeOrder.end());
    Value transposed = MemDescTransOp::create(rewriter, transOp.getLoc(),
                                              reshaped5D, transposeOrder32);
    SmallVector<int64_t> scale2DShapeVec(scale2DShape.begin(),
                                         scale2DShape.end());
    Value reshaped2D = MemDescReshapeOp::create(rewriter, reshapeOp2D.getLoc(),
                                                transposed, scale2DShapeVec);

    opOperand.assign(reshaped2D);
    rewriter.eraseOp(tmemAlloc);
    return success();
  }

  template <typename Op> Op getNextOp(Value op) const {
    while (auto cvtOp = op.getDefiningOp<ConvertLayoutOp>()) {
      op = cvtOp.getSrc();
    }
    return op.getDefiningOp<Op>();
  }

  bool isTmemCopyCompatible(triton::gpu::MemDescType scaleType,
                            bool usesTMAload) const {
    // TMEM copy expects that blocked scale "chunks" in SMEM are stored in
    // innermost axes contiguously.
    if (!isInnermostContiguous(scaleType, 512))
      return false;

    if (usesTMAload) {
      return true;
    }

    if (scaleType.getRank() != 2) {
      // TODO: Add support for higher rank when 5D coalesced load is fixed
      return false;
    }

    auto elemBits = scaleType.getElementType().getIntOrFloatBitWidth();

    // We assume that 32x128b chunks are flattened into the inner most axis.
    auto innerMostBits =
        scaleType.getDimSize(scaleType.getRank() - 1) * elemBits;
    return innerMostBits % (32 * 128) == 0;
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEDOTOPERANDS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public impl::TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  using impl::TritonGPUOptimizeDotOperandsBase<
      TritonGPUOptimizeDotOperandsPass>::TritonGPUOptimizeDotOperandsBase;

#ifdef __TLE__
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::tle::TleDialect>();
  }
#endif

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SwizzleShmemConvert>(context);
    patterns.add<FuseTransMMAV3Plus, ReshapeMemDesc>(context);
#ifdef __TLE__
    patterns.add<FuseCanonicalizedTransMMAV3Plus,
                 ReuseTransposedLocalLoadAllocAsWGMMAOperand,
                 ReuseTransposedLocalLoadConvertAsWGMMAA,
                 ReuseTransposedLocalLoadAllocAsWGMMAA, ReuseLocalLoadAsWGMMAA,
                 ReuseLocalLoadAllocAsWGMMAB>(context);
#endif
    patterns.add<UseShmemForScales>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
