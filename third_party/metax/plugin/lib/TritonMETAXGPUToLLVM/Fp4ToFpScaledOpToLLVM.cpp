#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using llvm::MapVector;

namespace {

int getNumPK(Type elemType) {
  if (elemType.isBF16()) {
    return 2;
  } else {
    assert(false && "Intrinsic fp4 cvt not supported data type");
    return 0;
  }
}

Type getRetTy(ConversionPatternRewriter &rewriter, Type elemType) {
  Type retTy;
  if (elemType.isBF16()) {
    retTy = vec_ty(f16_ty, 4);
  } else {
    assert(false && "Intrinsic fp4 cvt not supported data type");
  }
  return retTy;
}

Type getIntrinsicSrcTy(ConversionPatternRewriter &rewriter, Type elemType) {
  Type srcTy;
  if (elemType.isBF16()) {
    srcTy = i16_ty;
  } else {
    assert(false && "Intrinsic fp4 cvt not supported data type");
  }
  return srcTy;
}

std::string getIntrinsicName(Type elemType) {
  std::string intrinsic;
  if (elemType.isBF16()) {
    intrinsic = "llvm.mxc.cvt.pk4.f4tobf16.scale";
  } else {
    assert(false && "Intrinsic fp4 cvt not supported data type");
  }
  return intrinsic;
}

SmallVector<Value> genCvtIntrinsc(
    TritonLLVMOpBuilder &b, ConversionPatternRewriter &rewriter, Location &loc,
    Fp4ToFpScaledOp &op, const SmallVector<Value> &srcElems, Value scale,
    const SmallVector<size_t> &elemsIdx, std::string intrinsic,
    SmallVector<Value> &results, int numPK, Type retTy, Type intrinsicSrcTy) {
  auto i8xnty = vec_ty(i8_ty, numPK);
  int pks = elemsIdx.size() / numPK;
  int reminder = elemsIdx.size() % numPK;
  for (size_t j = 0; j < pks; j += 1) {
    SmallVector<Value> inputs(2);
    Value elemsComb = b.undef(i8xnty);
    for (int p = 0; p < numPK; p += 1) {
      elemsComb = b.insert_element(
          i8xnty, elemsComb, srcElems[elemsIdx[j * numPK + p]], b.i32_val(p));
    }
    Value elems = b.bitcast(elemsComb, intrinsicSrcTy);
    inputs[0] = elems;
    inputs[1] = scale;
    ValueRange inputsVal(inputs);
    Value resultVals = mlir::LLVM::createBuiltinFunc<triton::Fp4ToFpScaledOp>(
        rewriter, loc, op, intrinsic, retTy, inputsVal);
    if (retTy == vec_ty(f16_ty, 4)) {
      for (int q = 0; q < 4; q += 1) {
        auto resultIdx = (elemsIdx[j * numPK + q >> 1] << 1) + q % numPK;
        Value retElem = b.extract_element(resultVals, b.i32_val(q));
        results[resultIdx] = b.bitcast(retElem, i16_ty);
      }
    } else {
      assert(false && "fp4 retTy not supported!");
    }
  }
  if (reminder) {
    SmallVector<Value> inputs(2);
    Value elemsComb = b.undef(i8xnty);
    for (int p = 0; p < numPK; p += 1) {
      if (p < reminder) {
        elemsComb =
            b.insert_element(i8xnty, elemsComb,
                             srcElems[elemsIdx[pks * numPK + p]], b.i32_val(p));
      } else {
        elemsComb =
            b.insert_element(i8xnty, elemsComb, b.int_val(8, 0), b.i32_val(p));
      }
    }
    Value elems = b.bitcast(elemsComb, intrinsicSrcTy);
    inputs[0] = elems;
    inputs[1] = scale;
    ValueRange inputsVal(inputs);
    Value resultVals = mlir::LLVM::createBuiltinFunc<triton::Fp4ToFpScaledOp>(
        rewriter, loc, op, intrinsic, retTy, inputsVal);
    if (retTy == vec_ty(f16_ty, 4)) {
      for (int q = 0; q < 4; q += 1) {
        int idx = q >> 1;
        if (idx < reminder) {
          auto resultIdx = (elemsIdx[pks * numPK + idx] << 1) + q % numPK;
          Value retElem = b.extract_element(resultVals, b.i32_val(q));
          results[resultIdx] = b.bitcast(retElem, i16_ty);
        }
      }
    } else {
      assert(false && "fp4 retTy not supported!");
    }
  }
  return results;
}

MapVector<Value, llvm::SmallVector<size_t>>
getInputScaleMap(llvm::ArrayRef<Value> inputs, llvm::ArrayRef<Value> scales) {
  MapVector<Value, SmallVector<size_t>> scaleIdxMap;
  if (inputs.size() != scales.size()) {
    assert(false && "value1 and value2 must have the same size!");
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    Value key = scales[i];
    scaleIdxMap[key].push_back(i);
  }
  return scaleIdxMap;
}

class Fp4ToFpScaledOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpScaledOp> {
public:
  Fp4ToFpScaledOpPattern(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpScaledOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);
    auto numPK = getNumPK(elemType);
    auto retTy = getRetTy(rewriter, elemType);
    auto intrinsicSrcTy = getIntrinsicSrcTy(rewriter, elemType);
    auto intrinsicStr = getIntrinsicName(elemType);
    auto intrinsicStrLo = intrinsicStr + ".lo";
    auto intrinsicStrHi = intrinsicStr + ".hi";

    auto srcElems = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleElems = unpackLLElements(loc, adaptor.getScale(), rewriter);
    SmallVector<Value> results(srcElems.size() * 2);
    auto srcTy = op.getInput().getType().getElementType();
    auto scaleTy = op.getScale().getType().getElementType();
    assert(srcTy == i8_ty && scaleTy == i8_ty);
    auto i8x2ty = vec_ty(i8_ty, 2);

    assert(srcElems.size() % 2 == 0);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto scaleMap = getInputScaleMap(srcElems, scaleElems);
    auto numScales = scaleMap.size();

    if (numScales == 0) {
      llvm::errs() << "grouped has no scaleVals\n";
      return failure();
    }

    llvm::SmallVector<Value> scaleVals;
    scaleVals.reserve(numScales);

    for (auto &it : scaleMap) {
      scaleVals.push_back(it.first);
    }

    for (size_t i = 0; i + 1 < scaleVals.size(); i += 2) {
      Value scaleComb = b.undef(i8x2ty);
      Value scaleA = scaleVals[i];
      Value scaleB = scaleVals[i + 1];
      scaleComb = b.insert_element(i8x2ty, scaleComb, scaleA, b.i32_val(0));
      scaleComb = b.insert_element(i8x2ty, scaleComb, scaleB, b.i32_val(1));
      Value scale = b.bitcast(scaleComb, i16_ty);
      SmallVector<size_t> &elemsAIdx = scaleMap[scaleA];
      results =
          genCvtIntrinsc(b, rewriter, loc, op, srcElems, scale, elemsAIdx,
                         intrinsicStrLo, results, numPK, retTy, intrinsicSrcTy);
      SmallVector<size_t> &elemsBIdx = scaleMap[scaleB];
      results =
          genCvtIntrinsc(b, rewriter, loc, op, srcElems, scale, elemsBIdx,
                         intrinsicStrHi, results, numPK, retTy, intrinsicSrcTy);
    }

    if (scaleVals.size() % 2 == 1) {
      Value scaleComb = b.undef(i8x2ty);
      Value lastScale = scaleVals.back();
      scaleComb = b.insert_element(i8x2ty, scaleComb, lastScale, b.i32_val(0));
      scaleComb =
          b.insert_element(i8x2ty, scaleComb, b.int_val(8, 0), b.i32_val(1));
      Value scale = b.bitcast(scaleComb, i16_ty);
      SmallVector<size_t> &elemsIdx = scaleMap[lastScale];
      results =
          genCvtIntrinsc(b, rewriter, loc, op, srcElems, scale, elemsIdx,
                         intrinsicStrLo, results, numPK, retTy, intrinsicSrcTy);
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::METAX::populateFp4ToFpScaledToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpScaledOpPattern>(typeConverter, benefit);
}
