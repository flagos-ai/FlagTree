//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"

#define DEBUG_TYPE "triton-to-linalg"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

void mlir::triton::populateTritonToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MinMaxConverter<arith::CmpFOp>, MinMaxConverter<arith::CmpIOp>>(
      patterns.getContext());
}

void mlir::triton::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);
  populateFunctionOpInterfaceTypeConversionPattern<triton::CallOp>(
      patterns, typeConverter);

  patterns.add<MetaOpConverter>(patterns.getContext());
  patterns.add<StoreConverter>(patterns.getContext());
  patterns.add<LegacyAddPtrConverter>(patterns.getContext());
  patterns.add<MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<AdvanceConverter>(patterns.getContext());
  patterns.add<GetProgramIDConverter, GetNumProgramsConverter>(
      patterns.getContext());
  patterns.add<YieldConverter>(patterns.getContext());
  patterns.add<LoadConverter>(patterns.getContext());
  patterns.add<LoopConverter>(patterns.getContext());
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
  patterns.add<AssertConverter>(patterns.getContext());
  patterns.add<MatmulConverter>(patterns.getContext());
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());
  patterns.add<UnrealizedCastConverter>(patterns.getContext());
  patterns.add<CumSumConverter>(patterns.getContext());
  patterns.add<ReshapeConverter>(patterns.getContext());
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
  // MetaOpConverter has PatternBenefit == 10 which should take precedence over
  // these linalg patterns, but to be safe, add these patterns last so that they
  // will be tried last. Incorrect ordering or having MetaOpConverter has lower
  // PatternBenefit will result in element-wise meta ops being converted to
  // linalg.generic ops.
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}
