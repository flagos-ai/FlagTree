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
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}
