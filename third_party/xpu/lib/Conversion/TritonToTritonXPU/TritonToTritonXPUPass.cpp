// clang-format off
#include <numeric>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // mlir::triton::op
#include "triton/Dialect/Triton/IR/Utility.h"      // isPermutationOfIota
#include "triton/Dialect/TritonGPU/IR/Dialect.h" // AttrNumThreadsPerWarp, AttrNumWarpsName, AttrNumCTAsName

#include "triton/Conversion/TritonToTritonXPU/Passes.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include "triton/Dialect/TritonXPU/Transforms/TritonXPUConversion.h" // TritonXPUTypeConverter + TritonXPUConversionTarget
#include "llvm/Support/ErrorHandling.h" // TODO[dyq]: Check All Pattern And Remove It

#define GEN_PASS_DECL_CONVERTTRITONTOTRITONXPU
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONXPU
#include "triton/Conversion/TritonToTritonXPU/Passes.h.inc"
using ::impl::ConvertTritonToTritonXPUBase;
// clang-format on

namespace {
using namespace mlir;
using namespace mlir::triton;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (dyn_cast<RankedTensorType>(retShapedType)) {
      assert(value);
      if (value.getElementType().isInteger(1) && value.isSplat())
        // Workaround until https://reviews.llvm.org/D133743 is included.
        value =
            DenseElementsAttr::get(retShapedType, value.getSplatValue<bool>());
      else
        // This is a hack. We just want to add encoding
        value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonXPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonXPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonXPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonXPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::xpu::ClusterLayoutAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    // return encoding
    auto retSizePerCore = argEncoding.getSizePerCore().vec();
    retSizePerCore.insert(retSizePerCore.begin() + op.getAxis(), 1);
    auto retCoresPerGroup = argEncoding.getCoresPerGroup().vec();
    retCoresPerGroup.insert(retCoresPerGroup.begin() + op.getAxis(), 1);
    auto retGroupsPerCluster = argEncoding.getGroupsPerCluster().vec();
    retGroupsPerCluster.insert(retGroupsPerCluster.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    triton::xpu::ClusterLayoutAttr retEncoding =
        triton::xpu::ClusterLayoutAttr::get(getContext(), retSizePerCore,
                                            retCoresPerGroup,
                                            retGroupsPerCluster, retOrder);

    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(),
        cast<triton::gpu::DistributedEncodingTrait>(
            static_cast<Attribute>(retEncoding)));
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton::xpu::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

// Reject a Triton op this conversion has no encoding rule for.
//
// Why: these patterns used to be llvm_unreachable(). That aborts the process
// with no op name and no source location, and in an NDEBUG build it is UB that
// falls through to `return failure()`, which surfaces as the equally unhelpful
// "failed to legalize". The conversion is the first place that knows both the
// op and its loc, so it reports there -- same contract as the libdevice
// placeholder diagnostic below.
//
// How:
//   - build the message from the op name plus the operator that needs it;
//   - point at the R05 design note, which records what a real lowering needs.
//
// \param op the Triton op that has no XPU encoding rule yet.
// \param what the operator or feature that reaches it, for the reader.
static LogicalResult rejectUnimplementedOp(Operation *op, StringRef what) {
  return op->emitError()
         << "TritonToTritonXPU has no lowering for " << op->getName() << " ("
         << what
         << "); see "
            "third_party/xpu/docs/r05_unsupported_op_lowering.md";
}

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rejectUnimplementedOp(op, "dot: no cluster encoding rule for the "
                                     "accumulator");
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rejectUnimplementedOp(op, "cat: no joint encoding for the two "
                                     "inputs");
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rejectUnimplementedOp(op, "join: the result gains a trailing dim "
                                     "of size 2 that has no cluster encoding");
  }
};

struct TritonReshapePattern : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  // Give the reshape result the "nop" cluster layout so it is not an expensive
  // view.
  //
  // Why: GenericOpPattern would hand the result the type converter's DEFAULT
  // layout, which generally differs from the source's element-per-core layout;
  // ReshapeOpConversion then sees an elems-per-thread mismatch and rejects it
  // ("expensive view not supported on reshape op"). isreal's
  // tl.reshape([2*BLOCK] -> [BLOCK, 2]) is a pure renaming -- each core keeps
  // its (real, imag) pair -- so the result must carry the layout that
  // ClusterLayoutAttr::inferReshapeOpEncoding derives, which keeps the pair
  // in-core and makes the following tt.split find both halves.
  //
  // How: read the dst layout from inferReshapeOpEncoding (the same rule the op
  // trait verifier uses) and build the reshape with it; fall back to a clean
  // rejection when the reshape would move data across cores.
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = mlir::dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    auto dstTy = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return rejectUnimplementedOp(op, "reshape: only a ranked tensor on both "
                                       "sides is handled");
    auto srcEnc =
        mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());
    if (!srcEnc)
      return rejectUnimplementedOp(op, "reshape: the source carries no cluster "
                                       "layout");
    auto *infer = mlir::dyn_cast<triton::DialectInferLayoutInterface>(
        &srcEnc.getDialect());
    Attribute dstEnc;
    // NOTE: the query is made without a Location on purpose -- with one,
    // inferReshapeOpEncoding emits its own error and the reader gets two
    // diagnostics for one rejection. rejectUnimplementedOp is the single
    // report: it names the op, the operator and the design note.
    if (!infer || failed(infer->inferReshapeOpEncoding(srcTy.getShape(), srcEnc,
                                                       dstTy.getShape(), dstEnc,
                                                       /*loc=*/std::nullopt)))
      return rejectUnimplementedOp(
          op, "reshape: no in-core nop layout for this shape change");
    auto retTy =
        RankedTensorType::get(dstTy.getShape(), dstTy.getElementType(), dstEnc);
    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(op, retTy, adaptor.getSrc(),
                                                   op->getAttrs());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  // Give both halves the source layout with the split dimension dropped.
  //
  // Why: tt.split removes the trailing dim of size 2 and returns its two
  // elements. The halves are the same per-core data as the source, so their
  // layout is the source layout minus that dimension -- no data movement. A
  // slice encoding would carry the parent layout, which Legalize does not
  // re-derive when it slices the source, so the removed dimension is dropped
  // here instead and the layout stays self-contained.
  //
  // How:
  //   - require a rank-2 cluster layout source whose last dim is 2;
  //   - drop the last entry of each per-dim array and remove it from `order`,
  //     renumbering the axes above it;
  //   - build both results with that layout (the LLVM pattern selects the
  //     even/odd slots by offset).
  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = mlir::dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcTy || srcTy.getRank() != 2 || srcTy.getShape()[1] != 2)
      return rejectUnimplementedOp(op, "split: only a rank-2 tensor with a "
                                       "trailing dim of size 2 is handled");
    auto srcEnc =
        mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());
    if (!srcEnc)
      return rejectUnimplementedOp(op, "split: the source carries no cluster "
                                       "layout");

    // ClusterLayoutAttr::inferSplitOpEncoding owns the rule; the op trait
    // verifier calls it too, so reading it here keeps one answer.
    //
    // NOTE: the query is made without a Location on purpose -- with one,
    // inferSplitOpEncoding emits its own error and the reader gets two
    // diagnostics for one rejection. rejectUnimplementedOp is the single
    // report: it names the op, the operator and the design note.
    Attribute retEnc;
    auto *infer = mlir::dyn_cast<triton::DialectInferLayoutInterface>(
        &srcEnc.getDialect());
    if (!infer ||
        failed(infer->inferSplitOpEncoding(srcEnc, retEnc, srcTy.getShape(),
                                           /*loc=*/std::nullopt)))
      return rejectUnimplementedOp(op, "split: no in-core layout for the two "
                                       "halves of this shape");
    auto retShape = srcTy.getShape().drop_back();
    auto retTy =
        RankedTensorType::get(retShape, srcTy.getElementType(), retEnc);

    auto newOp = rewriter.create<triton::SplitOp>(
        op.getLoc(), TypeRange{retTy, retTy}, adaptor.getSrc());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// Is `order` a permutation of 0..rank-1 that also matches the source rank?
//
// Why: TritonTransPattern indexes srcShape, the layout dim vectors and the
// inverse map by order. The Dialect verifier already rejects a non-permutation
// TransOp (Ops.cpp TransOp::verify), so this guard only matters for unverified
// IR; it keeps the pattern from indexing out of bounds if it ever sees it.
static bool isValidTransOrder(ArrayRef<int32_t> order, size_t rank) {
  return order.size() == rank && isPermutationOfIota(order);
}

struct TritonTransPattern : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  // Pick the output layout so the op moves no data.
  //
  // Why: a transpose only renames the element a core holds. If the output
  // layout carries the same per-dimension split, permuted by the transpose
  // order, every core holds exactly the elements it held before -- the same
  // trick tt.trans documents for TritonGPU. The layout order has to be
  // permuted with it, because it decides how a core enumerates its own
  // elements: with order[fastest first] the transposed tensor must enumerate
  // the source axis p[k] where it used to enumerate axis k.
  //
  // How:
  //   - permute sizePerCore / coresPerGroup / groupsPerCluster by the order;
  //   - map the layout order through the inverse permutation;
  //   - keep tt.trans with that result type; the LLVM pattern becomes a
  //     slot-wise copy (see ViewOpToLLVM.cpp).
  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = mlir::dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcTy)
      return failure();
    auto srcEnc =
        mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());
    if (!srcEnc)
      return failure();
    auto order = op.getOrder();
    auto srcShape = srcTy.getShape();
    if (!isValidTransOrder(order, srcShape.size()))
      return rewriter.notifyMatchFailure(op,
                                         "trans order is not a permutation");

    auto retShape = llvm::to_vector(llvm::map_range(
        order, [&](int32_t axis) -> int64_t { return srcShape[axis]; }));

    auto permute = [&](ArrayRef<unsigned> dims) {
      llvm::SmallVector<unsigned> out;
      out.reserve(dims.size());
      for (auto axis : order)
        out.push_back(dims[axis]);
      return out;
    };

    llvm::SmallVector<unsigned> inverse(order.size());
    for (unsigned i = 0; i < order.size(); ++i)
      inverse[order[i]] = i;
    llvm::SmallVector<unsigned> retOrder;
    for (unsigned axis : srcEnc.getOrder())
      retOrder.push_back(inverse[axis]);

    auto retEnc = triton::xpu::ClusterLayoutAttr::get(
        getContext(), permute(srcEnc.getSizePerCore()),
        permute(srcEnc.getCoresPerGroup()),
        permute(srcEnc.getGroupsPerCluster()), retOrder);
    auto retTy =
        RankedTensorType::get(retShape, srcTy.getElementType(), retEnc);

    // Build with the explicit result type: the inferring builder goes through
    // the layout interface, whose trans-inference is still a TODO in the
    // dialect (IR/Dialect.cpp), and aborts.
    auto newOp = rewriter.create<triton::TransOp>(
        op.getLoc(), retTy, adaptor.getSrc(), op.getOrderAttr());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct TritonGatherOpPattern : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rejectUnimplementedOp(op, "gather (inverse): no per-core register "
                                     "gather lowering");
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = RankedTensorType::get(
        op.getType().getShape(), op.getType().getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::xpu::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};
/// Whether `reshapeOp` flattens its source to rank 1 and permits reordering.
///
/// Why: `tl.sum(x)` with no axis arrives as `tt.reshape allow_reorder` down to
/// rank 1 followed by a rank-1 reduce. There is no in-core cluster layout for
/// that flatten -- every element would have to move between cores -- so the
/// conversion rejects the reshape and bilinear / linalg_eig stop there with
/// "reshape: no in-core nop layout for this shape change". `allow_reorder` is
/// the frontend stating the element order carries no meaning, which is what
/// makes reducing the tile axis by axis an equivalent lowering.
static bool isReorderableFlatten(triton::ReshapeOp reshapeOp) {
  if (!reshapeOp.getAllowReorder())
    return false;
  auto srcTy = mlir::dyn_cast<RankedTensorType>(reshapeOp.getSrc().getType());
  auto dstTy = mlir::dyn_cast<RankedTensorType>(reshapeOp.getType());
  if (!srcTy || !dstTy || dstTy.getShape().size() != 1)
    return false;
  return srcTy.getShape().size() > 1;
}

/// Whether `reduceOp` is the scalar reduce that follows such a flatten.
///
/// How: one source, axis 0, a scalar result, and a flatten it is the only user
/// of. A multi-operand reduce (tl.argmax) pairs its sources and is left alone.
/// TODO(@claude): extend to the multi-operand form once an operator needs it.
static bool isScalarReduceOfFlatten(triton::ReduceOp reduceOp) {
  if (reduceOp.getSrcs().size() != 1 || reduceOp.getAxis() != 0)
    return false;
  if (mlir::isa<RankedTensorType>(reduceOp.getResult()[0].getType()))
    return false;
  auto reshapeOp = reduceOp.getSrcs()[0].getDefiningOp<triton::ReshapeOp>();
  return reshapeOp && reshapeOp->hasOneUse() && isReorderableFlatten(reshapeOp);
}

/// Reduce `src` along `axis` with a copy of `combineOp`'s body.
static Value buildReduceAlongAxis(OpBuilder &builder, Location loc, Value src,
                                  unsigned axis, mlir::Region &combineOp) {
  auto reduceOp = builder.create<triton::ReduceOp>(loc, ValueRange{src},
                                                   static_cast<int>(axis));
  mlir::IRMapping mapping;
  combineOp.cloneInto(&reduceOp.getCombineOp(), mapping);
  return reduceOp.getResult()[0];
}

/// Replace a scalar reduce over a flattened tile with one reduce per axis.
///
/// How: reduce the trailing axis first and work forward, so each step drops
/// one dimension and the last step yields the scalar the flatten reduce used
/// to produce. The combine body is copied into every step.
static void fuseFlattenIntoReduce(triton::ReduceOp reduceOp) {
  auto reshapeOp = reduceOp.getSrcs()[0].getDefiningOp<triton::ReshapeOp>();
  OpBuilder builder(reduceOp);
  Location loc = reduceOp.getLoc();
  Value reduced = reshapeOp.getSrc();
  unsigned rank =
      mlir::cast<RankedTensorType>(reduced.getType()).getShape().size();
  for (unsigned step = 0; step < rank; ++step)
    reduced = buildReduceAlongAxis(builder, loc, reduced, rank - 1 - step,
                                   reduceOp.getCombineOp());
  reduceOp.getResult()[0].replaceAllUsesWith(reduced);
  reduceOp.erase();
  reshapeOp.erase();
}

/// Rewrite every scalar reduce that reads a reorderable flatten.
static void fuseFlattenReducesInModule(ModuleOp mod) {
  llvm::SmallVector<triton::ReduceOp> flattenReduces;
  mod.walk([&](triton::ReduceOp reduceOp) {
    if (isScalarReduceOfFlatten(reduceOp))
      flattenReduces.push_back(reduceOp);
  });
  for (triton::ReduceOp reduceOp : flattenReduces)
    fuseFlattenIntoReduce(reduceOp);
}

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan = rewriter.create<triton::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    return success();
  }
};

// Lower tt.clampf to the two arith ops it stands for.
//
// Why: ConvertTritonXPUToLLVM marks the whole triton dialect illegal and has no
// pattern for tt.clampf, so any kernel that calls tl.clamp failed with
// "failed to legalize operation tt.clampf". The arith forms below already
// have lowerings (arith-to-LLVM is registered in the XPU conversion, and the
// vectorized path goes through the XPU elementwise patterns), so rewriting here
// is enough and keeps the clamp out of the vectorizer as two ordinary ops.
//
// How:
//   - PropagateNan::ALL -> arith.maximumf / arith.minimumf (NaN wins);
//   - PropagateNan::NONE -> arith.maxnumf / arith.minnumf (NaN is ignored).
struct TritonClampFPattern : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value clamped;
    if (op.getPropagateNan() == triton::PropagateNan::ALL) {
      auto lo = rewriter.create<arith::MaximumFOp>(loc, adaptor.getX(),
                                                   adaptor.getMin());
      clamped = rewriter.create<arith::MinimumFOp>(loc, lo, adaptor.getMax());
    } else {
      auto lo = rewriter.create<arith::MaxNumFOp>(loc, adaptor.getX(),
                                                  adaptor.getMin());
      clamped = rewriter.create<arith::MinNumFOp>(loc, lo, adaptor.getMax());
    }
    rewriter.replaceOp(op, clamped);
    return success();
  }
};

// Libdevice calls that the XPU device library does not implement. The table in
// python/triton/language/extra/xpu/libdevice.py names such a call
// "Unsupported.<function>". Reject it here, where the operator and its source
// location are both still available. Left in place, the call would reach the
// linker, which reports only an undefined symbol.
//
// kUnsupportedOnXPU prefixes every diagnostic for a lowering the XPU backend
// does not provide. The Python driver matches it in make_ttxir
// (third_party/xpu/backend/compiler.py), so keep the two in sync.
static constexpr llvm::StringLiteral kUnsupportedOnXPU = "[unsupported-on-xpu]";
struct TritonExternElementwisePattern
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The table spells a missing function "Unsupported.<function>". Older
    // entries, and TargetInfo::getMulhiFuncName, still produce the bare
    // placeholder, so match both or the bare form reaches the linker.
    StringRef symbol = op.getSymbol();
    if (symbol == "Unsupported")
      return op.emitError() << kUnsupportedOnXPU
                            << " libdevice call has no XPU implementation for "
                               "these operand types";
    if (symbol.consume_front("Unsupported."))
      return op.emitError()
             << kUnsupportedOnXPU << " libdevice function `" << symbol
             << "` has no XPU implementation for these operand "
                "types";

    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<triton::ExternElementwiseOp>(
        op, retTypes, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

// tt.atomic_cas lowers to a device call (XPUAtomicCASOpConversion in
// LoadStoreOpToLLVM.cpp) for i32, i64 and f32 operands. No other element type
// has a device function, so those are rejected here with the same prefix.
//
// xpu::atomicCAS only exists in device/xpu3/trigonometric.xpu, so the call
// also has to be rejected on every other arch. Without the arch check the
// lowering succeeds and the build fails in the linker instead, which is the
// failure mode this pass exists to prevent.
struct TritonAtomicCASPattern
    : public OpConversionPattern<triton::AtomicCASOp> {
  TritonAtomicCASPattern(TypeConverter &typeConverter, MLIRContext *context,
                         uint32_t xpuArch)
      : OpConversionPattern<triton::AtomicCASOp>(typeConverter, context),
        xpuArch(xpuArch) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (xpuArch != 3)
      return op.emitError()
             << kUnsupportedOnXPU << " tt.atomic_cas has no XPU implementation "
             << "on xpu" << xpuArch
             << "; xpu::atomicCAS is only provided for xpu3";

    Type elemTy = getElementTypeOrSelf(op.getType());
    if (!elemTy.isF32() && !elemTy.isInteger(32) && !elemTy.isInteger(64))
      return op.emitError()
             << kUnsupportedOnXPU << " tt.atomic_cas on " << elemTy
             << " has no XPU implementation; only i32, i64 and f32 are "
                "supported";

    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<triton::AtomicCASOp>(
        op, retTypes, adaptor.getOperands(), op->getAttrs());
    return success();
  }

private:
  uint32_t xpuArch;
};

// tt.mulhiui takes a different route to the same undefined symbol: the
// upstream MulhiUIOpConversion asks TargetInfo::getMulhiFuncName for a device
// function name and gets the bare string "Unsupported" for anything but i32
// (third_party/xpu/lib/Conversion/TritonXPUToLLVM/TargetInfo.cpp). That name
// never passes through tt.extern_elementwise, so reject the op here, where the
// element type and the source location are both available.
struct TritonMulhiUIPattern : public OpConversionPattern<triton::MulhiUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemTy = getElementTypeOrSelf(op.getType());
    if (!elemTy.isInteger(32))
      return op.emitError()
             << kUnsupportedOnXPU << " tt.mulhiui on " << elemTy
             << " has no XPU implementation; only i32 is supported";

    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<triton::MulhiUIOp>(
        op, retTypes, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

void populateTritonPatterns(TritonXPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns, uint32_t xpuArch) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<TritonAtomicCASPattern>(typeConverter, context, xpuArch);
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>, TritonReshapePattern,
      GenericOpPattern<triton::BitcastOp>, GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>, GenericOpPattern<triton::SplatOp>,
      TritonBroadcastPattern, GenericOpPattern<triton::AddPtrOp>,
      TritonCatPattern, TritonJoinOpPattern, TritonSplitOpPattern,
      TritonClampFPattern, GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>, TritonMulhiUIPattern,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>, TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>, TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>, TritonExpandDimsPattern,
      TritonTransPattern, TritonGatherOpPattern, TritonDotPattern,
      GenericOpPattern<triton::LoadOp>, GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>, TritonExternElementwisePattern,
      GenericOpPattern<triton::PrintOp>, GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::xpu::XPUPrintOp>,
      GenericOpPattern<triton::AtomicRMWOp>, GenericOpPattern<ReturnOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::CallOp>, TritonFuncOpPattern>(typeConverter,
                                                             context);
}

//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonXPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonXPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}
//

class ConvertTritonToTritonXPU
    : public ConvertTritonToTritonXPUBase<ConvertTritonToTritonXPU> {
public:
  ConvertTritonToTritonXPU() = default;
  // constructor with some parameters set explicitly.
  ConvertTritonToTritonXPU(uint32_t xpu_arch, uint32_t buffer_size,
                           uint32_t core_num, bool isTLE) {
    this->xpu_arch = xpu_arch;
    this->buffer_size = buffer_size;
    this->core_num = core_num;
    this->isTLE = isTLE;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter. the reason that we cant use TT2TTGPass directly
    bool tle = this->isTLE;
    TritonXPUTypeConverter typeConverter(context, buffer_size, core_num, tle);
    TritonXPUConversionTarget target(*context, typeConverter, tle);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, this->xpu_arch.getValue());
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);

    if (tle) {
      // GM normcopy ops are marked dynamically illegal while their pointer
      // operand is unencoded (see TritonXPUConversionTarget). Rebuild them with
      // the converted (ClusterLayout-encoded) operand so no encoded->unencoded
      // ConvertLayout bridge is materialized in front of them.
      patterns
          .insert<GenericOpPattern<triton::xpu::TLENormCopyGlobalToLocalOp>,
                  GenericOpPattern<triton::xpu::TLENormCopyLocalToGlobalOp>>(
              typeConverter, context);
    }

    auto inti = llvm::APSInt(32, false);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);

    if (!this->xpu_arch.getValue()) {
      mod.emitError("expected target specification to attach to the module op");
      return signalPassFailure();
    }
    mod->setAttr(
        AttrXPUTargetName,
        StringAttr::get(context,
                        "xpu:" + std::to_string(this->xpu_arch.getValue())));

    // Set TritonGPU module attributes so the upstream
    // TritonGPUVerifyTensorLayoutInterface (LLVM22+) accepts ClusterLayoutAttr.
    // ClusterLayoutAttr::toLinearLayout maps `coresPerGroup` to the `lane` dim
    // and `groupsPerCluster` to the `warp` dim, so the module attrs must
    // match: threads-per-warp = core_num (size of coresPerGroup) and
    // num-warps = product(groupsPerCluster) (XPU has no inter-CTA concept).
    {
      Builder b(context);
      mod->setAttr(::mlir::triton::gpu::AttrNumThreadsPerWarp,
                   b.getI32IntegerAttr(this->core_num.getValue()));
      mod->setAttr(::mlir::triton::gpu::AttrNumWarpsName,
                   b.getI32IntegerAttr(1));
      mod->setAttr(::mlir::triton::gpu::AttrNumCTAsName,
                   b.getI32IntegerAttr(1));
    }

    // `tl.sum(x)` without an axis arrives as a flatten plus a rank-1 reduce,
    // and the flatten has no in-core layout on this target. Fuse it into the
    // reduce before the conversion sees it.
    fuseFlattenReducesInModule(mod);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // update layouts
    //  broadcast src => multicast, dst => broadcasted
    // if (failed(target.refineLayouts(mod, numWarps)))
    //   return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonXPUPass(uint32_t xpu_arch,
                                                 uint32_t buffer_size,
                                                 uint32_t core_num,
                                                 bool isTLE) {
  return std::make_unique<::ConvertTritonToTritonXPU>(xpu_arch, buffer_size,
                                                      core_num, isTLE);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonXPUPass() {
  return std::make_unique<::ConvertTritonToTritonXPU>();
}
