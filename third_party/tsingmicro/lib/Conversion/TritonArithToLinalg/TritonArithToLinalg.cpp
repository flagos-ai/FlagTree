//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "triton-arith-to-linalg"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

namespace {
static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

static SmallVector<Value, 4>
getOrCreateOperandsMatchingResultTypes(OpBuilder &b, Operation *op) {
  assert(isElementwiseMappableOpOnRankedTensors(op));
  Location loc = op->getLoc();
  ValueRange operands = op->getOperands();
  TypeRange rankedTensorTypes = op->getResultTypes();
  SmallVector<Value, 4> res;
  res.reserve(rankedTensorTypes.size());
  for (Type t : rankedTensorTypes) {
    // Extract static / dynamic shape mix from the first operand.
    res.push_back(b.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(b, loc, operands.front()),
        cast<RankedTensorType>(t).getElementType()));
  }
  return res;
}

struct ConvertAnyElementwiseMappableOpOnRankedTensors : public RewritePattern {
  ConvertAnyElementwiseMappableOpOnRankedTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    auto rank = cast<RankedTensorType>(op->getResult(0).getType()).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto outputs = getOrCreateOperandsMatchingResultTypes(rewriter, op);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/op->getOperands(),
        /*outputs=*/outputs,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              }));
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};

// FIXME: When we fix the memory management inside the loop, replace it to mlir
// upstream linalg::populateElementwiseToLinalgConversionPatterns
void populateElementwiseToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(
      patterns.getContext());
}

} // namespace

void mlir::triton::populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MinMaxConverter<arith::CmpFOp>, MinMaxConverter<arith::CmpIOp>>(
      patterns.getContext());
}

void mlir::triton::populateTritonArithToLinalgConversionPatterns(
    bool pidsToFuncArgs, bool addptrToLinalg, bool assertToCf,
    RewritePatternSet &patterns) {

  if (pidsToFuncArgs) {
    // Need use tx interface to get pid.
    patterns.add</* GetProgramIDConverter ,*/ GetNumProgramsConverter>(
        patterns.getContext());
  }
  if (addptrToLinalg) {
    patterns.add<AddPtrConverter>(patterns.getContext());
  }
  if (assertToCf) {
    patterns.add<AssertConverter>(patterns.getContext());
  }
  patterns.add<BarrierConverter>(patterns.getContext());
  patterns.add<BroadcastConverter>(patterns.getContext());
  patterns.add<TransposeConverter>(patterns.getContext());
  patterns.add<MakeRangeConverter>(patterns.getContext());
  patterns.add<ExpandDimsConverter>(patterns.getContext());
  patterns.add<BitcastConverter>(patterns.getContext());
  patterns.add<CallConverter>(patterns.getContext());
  patterns.add<MulHiUIOpConverter>(patterns.getContext());
  patterns.add<PreciseSqrtConverter>(patterns.getContext());
  patterns.add<PreciseDivConverter>(patterns.getContext());
  patterns.add<CatConverter>(patterns.getContext());
  patterns.add<SplitConverter>(patterns.getContext());
  patterns.add<JoinConverter>(patterns.getContext());
  patterns.add<FpToFpConverter>(patterns.getContext());
  patterns.add<ClampConverter>(patterns.getContext());
  patterns.add<MatmulConverter>(patterns.getContext());
  patterns.add<DotScaledConverter>(patterns.getContext());
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());
  patterns.add<ScanOpConverter>(patterns.getContext());
  patterns.add<ReshapeConverter>(patterns.getContext());
  patterns.add<GatherConverter>(patterns.getContext());
  patterns.add<HistogramOpConversion>(patterns.getContext());
  patterns.add<PrintOpConverter>(patterns.getContext());

  populateExternElementwiseOpToMLIROps(patterns);

  // Reduce converters
  // Triton's reduce op is idential to linalg.reduce op, so we can clone
  // `tt.reduce` body to `linalg.reduce`. Unfortunately, we still need to
  // perform pattern matching to know what reduce ops we are dealing with
  // so that we know how to initialize the initial reduce values correctly.
  //
  // We can do this in a generic way without pattern matching by always using
  // the first elements along the reduction axis and perform the reduction on
  // the remaining elements. However, this results in creatings sub-tensors that
  // aren't always multiple of 2s, which are sub-optimal for certain hardwares.
  patterns.add<ArgMinConverter>(patterns.getContext());
  patterns.add<ArgMaxConverter>(patterns.getContext());
  patterns.add<ReduceConverter>(patterns.getContext());

  // Note: the ordering here matters!
  // These patterns are added last to they will be tried last.
  populateElementwiseToLinalgConversionPatterns(patterns);
}
