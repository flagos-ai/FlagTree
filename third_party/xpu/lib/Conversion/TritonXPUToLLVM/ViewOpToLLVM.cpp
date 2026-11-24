#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only

namespace {
struct XPUExpandDimsOpConversion
    : public ConvertOpToLLVMPattern<triton::ExpandDimsOp> {

  XPUExpandDimsOpConversion(LLVMTypeConverter &converter,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ExpandDimsOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcLayout = dyn_cast<SliceEncodingAttr>(srcTy.getEncoding());
    if (!srcLayout) {
      return emitOptionalError(
          loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }
    auto resultLayout = resultTy.getEncoding();

    Value ret = packLLElements(loc, typeConverter, srcVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct XPUBroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::BroadcastOp> {
  using ConvertOpToLLVMPattern<
      triton::xpu::BroadcastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::xpu::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(result.getType());
    Type resElemTy = getElementTypeOrSelf(resultTy);
    bool isVectorized = isa<VectorType>(resElemTy);

    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    auto typeConverter = getTypeConverter();
    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcTy);
    auto srcOffsets = mlir::LLVM::XPU::emitOffsetForLayoutXPU(srcLayout, srcTy);
    auto resultOffsets =
        mlir::LLVM::XPU::emitOffsetForLayoutXPU(resultLayout, resultTy);
    SmallVector<Value> srcVals = unpackLLElements(loc, src, rewriter);
    std::map<SmallVector<unsigned>, Value> srcValues;
    if (isVectorized) {
      // TODO: Enhance VBroadcastOp
      assert(rank == 2 && "BroadcastOp is Vectorized, But Rank != 2");
      auto rowsPerCore = cast<triton::xpu::ClusterLayoutAttr>(resultLayout)
                             .getSizePerCore()[0];
      Type elemTy = getElementTypeOrSelf(resElemTy);
      unsigned vecSize = 512 / elemTy.getIntOrFloatBitWidth();
      if (srcShape[1] == 1) {
        for (size_t i = 0; i < srcOffsets.size(); i++) {
          Value srcVals_0_vector =
              rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemStart = 0; elemStart < vecSize; ++elemStart) {
            srcVals_0_vector = insert_element(resElemTy, srcVals_0_vector,
                                              srcVals[i], i32_val(elemStart));
          }
          srcValues[srcOffsets[i]] = srcVals_0_vector;
        }
      } else if (srcShape[0] == 1) {
        // [1xNxTy] -> [MxNxTy] column broadcast: replicate the N/vecSize
        // source column-vectors across the M rows.
        //
        // The source element type decides how those column-vectors are
        // obtained. If the source is ALREADY vectorized (element type is a
        // VectorType, e.g. tensor<1 x N/vecSize x vector<vecSize x Ty>>), each
        // srcVals[i] IS the column-vector -- use it directly, which avoids
        // materializing all N scalar source elements at once (that scalar
        // packing forces the whole source row live simultaneously and blows up
        // the per-core stack via register spills). Otherwise the source is
        // scalar (tensor<1 x N x Ty>) and we pack vecSize consecutive scalars
        // into each column-vector via insert_element (legacy path).
        bool srcIsVectorized = isa<VectorType>(getElementTypeOrSelf(srcTy));
        SmallVector<Value> srcVectorVals;
        for (size_t i = 0; i < resultOffsets.size() / rowsPerCore; i++) {
          if (srcIsVectorized) {
            srcVectorVals.push_back(srcVals[i]);
            continue;
          }
          Value srcVals_vector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemStart = 0; elemStart < vecSize; ++elemStart) {
            srcVals_vector = insert_element(resElemTy, srcVals_vector,
                                            srcVals[i * vecSize + elemStart],
                                            i32_val(elemStart));
          }
          srcVectorVals.push_back(srcVals_vector);
        }
        for (size_t i = 0; i < resultOffsets.size() / rowsPerCore; i++) {
          srcValues[resultOffsets[i]] = srcVectorVals[i];
        }
      } else {
        llvm_unreachable("Only Support Vectorized BroadcastOp: [Mx1xTy] -> "
                         "[MxNxTy] or [1xNxTy] -> [MxNxTy]");
      }
    } else {
      for (size_t i = 0; i < srcOffsets.size(); i++) {
        srcValues[srcOffsets[i]] = srcVals[i];
      }
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.at(offset));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

// tt.split picks the two halves of a trailing dim of size 2.
//
// Why: TritonToTritonXPU drops that dimension from the layout, so each result
// is a subset of the source register list: the slots whose trailing offset is
// 0 (outLHS) or 1 (outRHS). Both halves live in the same core, so this is a
// selection, not a data movement.
//
// How:
//   - enumerate the per-core offsets of the source and of each result;
//   - for each result slot take the source slot with the same offsets and the
//     wanted trailing value;
//   - fail loudly when a half is missing (a layout that spreads the trailing
//     dim over cores cannot express this op).
struct XPUTSplitOpConversion : public ConvertOpToLLVMPattern<triton::SplitOp> {
  using ConvertOpToLLVMPattern<triton::SplitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto lhsTy = cast<RankedTensorType>(op.getOutLHS().getType());
    auto rhsTy = cast<RankedTensorType>(op.getOutRHS().getType());
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcOffsets =
        mlir::LLVM::XPU::emitOffsetForLayoutXPU(srcTy.getEncoding(), srcTy);
    if (srcVals.size() != srcOffsets.size())
      return op->emitError(
          "tt.split source element count does not match its layout");

    auto selectHalf =
        [&](unsigned half,
            RankedTensorType dstTy) -> std::optional<SmallVector<Value>> {
      auto dstOffsets =
          mlir::LLVM::XPU::emitOffsetForLayoutXPU(dstTy.getEncoding(), dstTy);
      SmallVector<Value> out;
      out.reserve(dstOffsets.size());
      for (ArrayRef<unsigned> dst : dstOffsets) {
        std::optional<Value> found;
        for (unsigned i = 0; i < srcOffsets.size(); ++i) {
          ArrayRef<unsigned> src = srcOffsets[i];
          if (src.back() != half)
            continue;
          if (!std::equal(dst.begin(), dst.end(), src.begin()))
            continue;
          found = srcVals[i];
          break;
        }
        if (!found)
          return std::nullopt;
        out.push_back(*found);
      }
      return out;
    };

    auto lhsVals = selectHalf(/*half=*/0, lhsTy);
    auto rhsVals = selectHalf(/*half=*/1, rhsTy);
    if (!lhsVals || !rhsVals)
      return op->emitError(
          "tt.split could not find both halves in the source layout");
    Value lhs = packLLElements(loc, typeConverter, *lhsVals, rewriter, lhsTy);
    Value rhs = packLLElements(loc, typeConverter, *rhsVals, rewriter, rhsTy);
    rewriter.replaceOp(op, {lhs, rhs});
    return success();
  }
};

// tt.trans is a register rename, not a data movement.
//
// Why: TritonToTritonXPU gives the result the source layout permuted by the
// transpose order (see TritonTransPattern), so every core holds exactly the
// elements it held before, in the same slot order. The element list therefore
// carries over unchanged, exactly like convert_layout. Materialising a real
// permute here would be both wrong (the layout already encodes it) and
// needlessly expensive.
//
// How: unpack the source struct and pack it into the result type; the type
// converter sizes both sides from the (permuted) encoding.
struct XPUTransOpConversion : public ConvertOpToLLVMPattern<triton::TransOp> {
  using ConvertOpToLLVMPattern<triton::TransOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resTy = cast<RankedTensorType>(op.getType());
    auto srcEnc =
        dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());
    auto resEnc =
        dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(resTy.getEncoding());
    if (!srcEnc || !resEnc)
      return failure();
    if (triton::gpu::getTotalElemsPerThread(srcTy) !=
        triton::gpu::getTotalElemsPerThread(resTy))
      return op->emitError(
          "tt.trans layout permutation changed the per-core element count");
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto res = packLLElements(loc, typeConverter, vals, rewriter, resTy);
    rewriter.replaceOp(op, res);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<XPUExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<XPUBroadcastOpConversion>(typeConverter, benefit);
  patterns.add<XPUTransOpConversion>(typeConverter, benefit);
  patterns.add<XPUTSplitOpConversion>(typeConverter, benefit);
}
