/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Tools/LayoutUtils.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using triton::gpu::SwizzledSharedEncodingAttr;

namespace {
struct ExtractTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ExtractTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::ExtractTensorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_tensor %source [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    if (!srcTy)
      return failure();
    auto srcLayout = srcTy.getEncoding();
    Value result = op.getResult();
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();
    auto ctaIdx = op.getCtaIdx();
    auto elemIdx = op.getElemIdx();
    SmallVector<Value> subelems;
    ArrayRef<Type> types =
        cast<LLVM::LLVMStructType>(adaptor.getSource().getType()).getBody();

    auto subIdx =
        mlir::LLVM::emitSubOffsetForLayout(srcLayout, srcTy, ctaIdx, elemIdx);
    for (unsigned i : subIdx) {
      subelems.push_back(b.extract_val(types[i], adaptor.getSource(), i));
    }
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), subelems, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct InsertTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::InsertTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::InsertTensorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::InsertTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_tensor %inserted %insert [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto insertedTy = dyn_cast<RankedTensorType>(op.getInserted().getType());
    if (!insertedTy)
      return failure();
    auto insertedLayout = insertedTy.getEncoding();
    auto insertTy = dyn_cast<RankedTensorType>(op.getInsert().getType());
    if (!insertTy)
      return failure();
    Value result = op.getResult();
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();
    auto ctaIdx = op.getCtaIdx();
    auto elemIdx = op.getElemIdx();
    auto subelems = unpackLLElements(loc, adaptor.getInsert(), rewriter);
    auto subIdx = mlir::LLVM::emitSubOffsetForLayout(insertedLayout, insertedTy,
                                                     ctaIdx, elemIdx);
    unsigned idx = 0;
    auto resultStruct = adaptor.getInserted();
    auto resultStructTy = dyn_cast<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(resultTy));
    assert(subIdx.size() == subelems.size());
    for (unsigned i : subIdx) {
      resultStruct =
          b.insert_val(resultStructTy, resultStruct, subelems[idx], i);
      ++idx;
    }
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BsmPermOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BsmPermOp> {
  using ConvertOpToLLVMPattern<triton::gpu::BsmPermOp>::ConvertOpToLLVMPattern;
  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  LogicalResult
  matchAndRewrite(triton::gpu::BsmPermOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc1().getType());
    auto srcLayout = srcTy.getEncoding();
    Value result = op.getResult();
    auto resultTy = cast<RankedTensorType>(result.getType());
    ArrayRef<Type> types =
        cast<LLVM::LLVMStructType>(adaptor.getSrc1().getType()).getBody();

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());

    auto dotOpEnc =
        mlir::dyn_cast<DotOperandEncodingAttr>(resultTy.getEncoding());
    auto mmaEnc = dyn_cast<MACAMmaEncodingAttr>(
        dyn_cast<DotOperandEncodingAttr>(resultTy.getEncoding()).getParent());

    if (dotOpEnc.getOpIdx() != 1 || !srcTy.getElementType().isInteger(32) ||
        (!resultTy.getElementType().isF16() &&
         !resultTy.getElementType().isBF16())) {
      // TODO(): Currently only supports perm of B.
      assert(false && "Currently only supports perm of B");
    }

    auto elemMNK = mmaEnc.getElementsMNK();
    auto elemsN = elemMNK[dotOpEnc.getOpIdx()];
    int elemsK = elemMNK[2];

    Type elemType;
    if (resultTy.getElementType().isF16()) {
      elemType = type::f16Ty(ctx);
    } else if (resultTy.getElementType().isBF16()) {
      elemType = type::i16Ty(ctx);
    } else if (resultTy.getElementType().isF32()) { // TF32
      elemType = type::f32Ty(ctx);
    } else {
      assert(false && "Invalid smem load");
    }
    Type fp16x2Ty = vec_ty(elemType, 2);
    Value halfValuex2Front = b.undef(fp16x2Ty);
    Value halfValuex2Back = b.undef(fp16x2Ty);

    auto emit_perm_builtin = [&](Value firstVec, Value secondVec, int i32Idx,
                                 bool is_back) -> Value {
      std::string intrinsicPermName = "llvm.mxc.byte.perm";
      StringRef permName(intrinsicPermName);

      Value offsetFront = b.i32_val(0x01000504);
      Value offsetBack = b.i32_val(0x03020706);

      SmallVector<Value> inputs(3);
      if (is_back) {
        inputs[2] = offsetBack;
      } else {
        inputs[2] = offsetFront;
      }
      inputs[0] = firstVec;
      inputs[1] = secondVec;
      ValueRange permValueRange(inputs);
      Value res = b.undef(i32_ty);
      res = mlir::LLVM::createBuiltinFunc(rewriter, loc, op, permName, i32_ty,
                                          permValueRange);
      return res;
    };

    int elemsSize = elemsN * elemsK;
    SmallVector<Value> outVals(32);
    Type elemV2Ty = vec_ty(llvmElemTy, 2);
    int halfOfElemsSize = elemsSize / 2;
    for (int j = 0; j < elemsSize / 2; j += 2 * elemsN) {
      for (int vec2idx = 0; vec2idx < elemsN / 2; ++vec2idx) {
        int index = (j / elemsN + vec2idx) * elemsK;
        int index_0 = 2 * vec2idx + elemsN * (j / elemsK);
        auto firstKVec = b.extract_val(types[index], adaptor.getSrc1(), index);
        auto secondKVec =
            b.extract_val(types[index + 1], adaptor.getSrc1(), index + 1);

        auto firstKVecSt = b.extract_val(types[index + elemsN],
                                         adaptor.getSrc1(), index + elemsN);
        auto secondKVecSt = b.extract_val(
            types[index + elemsN + 1], adaptor.getSrc1(), index + elemsN + 1);

        auto permValueI32Front =
            emit_perm_builtin(firstKVec, secondKVec, vec2idx, /*is_back*/ 0);
        auto permValueI32Back =
            emit_perm_builtin(firstKVec, secondKVec, vec2idx, /*is_back*/ 1);
        auto halfValuex2Front = b.bitcast(permValueI32Front, elemV2Ty);
        auto halfValuex2Back = b.bitcast(permValueI32Back, elemV2Ty);
        outVals[index_0] = b.extract_element(halfValuex2Front, b.i32_val(0));
        outVals[index_0 + 1] =
            b.extract_element(halfValuex2Front, b.i32_val(1));
        outVals[index_0 + elemsK] =
            b.extract_element(halfValuex2Back, b.i32_val(0));
        outVals[index_0 + 1 + elemsK] =
            b.extract_element(halfValuex2Back, b.i32_val(1));

        auto permValueI32FrontSt = emit_perm_builtin(firstKVecSt, secondKVecSt,
                                                     vec2idx, /*is_back*/ 0);
        auto permValueI32BackSt = emit_perm_builtin(firstKVecSt, secondKVecSt,
                                                    vec2idx, /*is_back*/ 1);
        auto halfValuex2FrontSt = b.bitcast(permValueI32FrontSt, elemV2Ty);
        auto halfValuex2BackSt = b.bitcast(permValueI32BackSt, elemV2Ty);
        outVals[index_0 + 2 * elemsK] =
            b.extract_element(halfValuex2FrontSt, b.i32_val(0));
        outVals[index_0 + 1 + 2 * elemsK] =
            b.extract_element(halfValuex2FrontSt, b.i32_val(1));
        outVals[index_0 + 3 * elemsK] =
            b.extract_element(halfValuex2BackSt, b.i32_val(0));
        outVals[index_0 + 1 + 3 * elemsK] =
            b.extract_element(halfValuex2BackSt, b.i32_val(1));
      }
    }

    Type elemTy = outVals[0].getType();
    auto regVal = op.getResult();
    auto regTy = cast<RankedTensorType>(regVal.getType());
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, regTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

struct SwizzleTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::SwizzleTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::SwizzleTensorOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::gpu::SwizzleTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy)
      return failure();

    auto maxPhase = op.getMaxPhase();
    if (maxPhase == 1) {
      rewriter.replaceOp(op, {adaptor.getSrc()});
      return success();
    }

    Value result = op.getResult();
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();

    auto inVec = op.getInVec();
    auto outVec = op.getOutVec();
    auto perPhase = op.getPerPhase();
    auto minVec = std::min(inVec, outVec);
    auto order = triton::gpu::getOrder(srcTy);

    auto swizzleSharedEnc = SwizzledSharedEncodingAttr::get(
        op->getContext(), outVec, perPhase, maxPhase, order,
        triton::gpu::getCTALayout(srcTy.getEncoding()));

    auto noSwizzleSharedEnc = SwizzledSharedEncodingAttr::get(
        op->getContext(), outVec, 1, 1, order,
        triton::gpu::getCTALayout(srcTy.getEncoding()));

    SmallVector<Value> colOffsets = triton::gpu::emitSwizzleColOffsets(
        rewriter, op, srcTy, srcTy.getShape(), swizzleSharedEnc,
        noSwizzleSharedEnc, minVec);

    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned numElems = srcElems.size();
    SmallVector<Value> llResult = srcElems;
    for (int i = 0; i < srcElems.size(); i++) {
      auto colOffset = colOffsets[i / minVec];
      llResult[i] = b.add(srcElems[i], colOffset);
    }
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), llResult, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};
} // namespace

void mlir::triton::METAX::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ExtractTensorOpConversion>(typeConverter, benefit);
  patterns.add<InsertTensorOpConversion>(typeConverter, benefit);
  patterns.add<BsmPermOpConversion>(typeConverter, benefit);
  patterns.add<SwizzleTensorOpConversion>(typeConverter, benefit);
}
