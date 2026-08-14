//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. ALL rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights
// reserved. SPDX-License-Identifier: MIT
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
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp"
#include "triton-shared/Conversion/TritonArithToLinalg/TypeConverter.hpp"

using namespace mlir;
using namespace triton;

#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

void mlir::triton::populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MinMaxConverter<arith::CmpFOp>, MinMaxConverter<arith::CmpIOp>>(
      patterns.getContext());
}

void mlir::triton::populateTritonTensorPtrConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<StorePtrToLinalgConverter, TensorOpConverter<triton::LoadOp>,
               TensorOpConverter<triton::IntToPtrOp>,
               TensorOpConverter<triton::PtrToIntOp>,
               TensorOpConverter<triton::BitcastOp>>(patterns.getContext());
}

void mlir::triton::populateTritonArithToLinalgConversionPatterns(
    bool pidsToFuncArgs, bool addptrToLinalg, bool assertToCf,
    RewritePatternSet &patterns,
    mlir::triton::TritonLinalgTypeConverter &converter) {

  if (pidsToFuncArgs) {
    patterns.add<GetProgramIDConverter, GetNumProgramsConverter>(
        patterns.getContext());
  }
  if (addptrToLinalg) {
    patterns.add<AddPtrConverter>(patterns.getContext());
  }
  if (assertToCf) {
    patterns.add<AssertConverter>(patterns.getContext());
  }
  patterns.add<PrintOpConverter>(patterns.getContext());
  patterns.add<TritonPtrToIntPattern>(converter, patterns.getContext());
  patterns.add<BroadcastConverter>(patterns.getContext());
  patterns.add<TransposeConverter>(patterns.getContext());
  patterns.add<MakeRangeConverter>(patterns.getContext());
  patterns.add<ExpandDimsConverter>(patterns.getContext());
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
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());
  // patterns.add<CumSumConverter>(patterns.getContext());
  patterns.add<ReshapeConverter>(patterns.getContext());
  patterns.add<ConvertExternElementwise>(patterns.getContext());
  patterns.add<ConvertExternSpecialMath>(patterns.getContext());

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
  patterns.add<ConvertExternGeluTanh>(patterns.getContext());
  patterns.add<ConvertExternGeluNone>(patterns.getContext());
  patterns.add<ConvertExternSilu>(patterns.getContext());
  patterns.add<BitcastConverter>(patterns.getContext());

  // Note: the ordering here matters!
  // These patterns are added last to they will be tried last.
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}

void mlir::triton::normalReduceConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<normalReduceConverter>(patterns.getContext());
  patterns.add<TritonReduceReturnPattern>(patterns.getContext());
}
