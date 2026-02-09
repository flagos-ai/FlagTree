#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

struct LocalPointersOpConversion
    : public ConvertOpToLLVMPattern<tle::LocalPointersOp> {
  LocalPointersOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(tle::LocalPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[LocalPointersOpConversion] " << msg << "\n";
      return rewriter.notifyMatchFailure(op, msg);
    };

    auto memDescTy = cast<ttg::MemDescType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto ptrTy = cast<triton::PointerType>(resultTy.getElementType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto llvmPtrTy =
        cast<LLVM::LLVMPointerType>(typeConverter->convertType(ptrTy));
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    auto i32Ty = rewriter.getIntegerType(32);
    auto ensureI32 = [&](Value v) -> Value {
      if (v.getType() == i32Ty)
        return v;
      if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
        if (intTy.getWidth() > 32)
          return rewriter.create<LLVM::TruncOp>(loc, i32Ty, v);
        if (intTy.isUnsigned())
          return rewriter.create<LLVM::ZExtOp>(loc, i32Ty, v);
        return rewriter.create<LLVM::SExtOp>(loc, i32Ty, v);
      }
      return Value();
    };

    auto sharedEnc = cast<ttg::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kOffset = str_attr("offset");
    if (!resultTy.getEncoding())
      return reportFailure("local_pointers result must carry an encoding");

    LinearLayout regLayout = ttg::toLinearLayout(resultTy);
    for (Value operand : op.getIndices()) {
      auto idxTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!idxTy)
        return reportFailure("indices must be ranked tensors");
      if (resultTy.getEncoding() && idxTy.getEncoding() &&
          resultTy.getEncoding() != idxTy.getEncoding())
        return reportFailure(
            "indices tensor encoding must match result encoding");
    }

    SmallVector<Value> outVals(regLayout.getInDimSize(kReg), Value());

    TritonLLVMOpBuilder b(loc, rewriter);
    int elemBits = llvmElemTy.getIntOrFloatBitWidth();
    assert(elemBits % 8 == 0 && "element bitwidth must be byte addressable");
    int elemBytes = elemBits / 8;
    Value elemBytesVal =
        elemBytes > 1 ? b.i32_val(static_cast<int32_t>(elemBytes)) : Value();
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx, llvmPtrTy.getAddressSpace());

    SmallVector<unsigned> bufferShape;
    for (int64_t dim : memDescTy.getShape())
      bufferShape.push_back(static_cast<unsigned>(dim));
    auto bufferRank = bufferShape.size();
    auto smemOffsets = smemObj.getOffsets();
    if (smemOffsets.size() != bufferRank)
      return reportFailure("shared memory offsets rank mismatch");

    auto indexVals = adaptor.getIndices();
    if (indexVals.size() != bufferRank)
      return reportFailure("indices must provide buffer-rank values");

    SmallVector<SmallVector<Value>> indexElems;
    indexElems.reserve(indexVals.size());
    for (Value indexVal : indexVals) {
      auto elems = unpackLLElements(loc, indexVal, rewriter);
      if (elems.size() != outVals.size())
        return reportFailure(
            "indices tensors must match local_pointers result shape");
      indexElems.push_back(std::move(elems));
    }

    for (size_t idx = 0; idx < outVals.size(); ++idx) {
      SmallVector<Value> idxCoords;
      idxCoords.reserve(indexVals.size());
      for (size_t dim = 0; dim < indexElems.size(); ++dim) {
        Value val = ensureI32(indexElems[dim][idx]);
        if (!val)
          return reportFailure("indices must lower to i32 scalars");
        Value offset = smemOffsets[dim];
        Value offVal = ensureI32(offset);
        if (!offVal)
          return reportFailure("shared memory offsets must be i32");
        idxCoords.push_back(b.add(val, offVal));
      }

      Value elemOffset;
      if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEnc)) {
        auto order = ttg::getOrder(sharedEnc, memDescTy.getShape());
        elemOffset =
            LLVM::linearize(rewriter, loc, idxCoords, bufferShape, order);
      } else {
        auto dimNames = standardOutDimNames(ctx, bufferRank);
        SmallVector<std::pair<StringAttr, Value>> logicalOffsets;
        logicalOffsets.reserve(bufferRank);
        for (auto [dim, offset] : llvm::zip_equal(dimNames, idxCoords))
          logicalOffsets.push_back({dim, offset});
        LinearLayout sharedLayout = ttg::toLinearLayout(memDescTy);
        sharedLayout = sharedLayout.sublayout({kOffset}, dimNames);
        elemOffset = applyLinearLayout(loc, rewriter, sharedLayout.invert(),
                                       logicalOffsets)[0]
                         .second;
      }

      Value byteOffset = elemOffset;
      if (elemBytes > 1)
        byteOffset = b.mul(byteOffset, elemBytesVal);
      if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEnc)) {
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, elemBits,
                                      byteOffset, /*offsetInBytes=*/true);
        byteOffset = b.add(byteOffset, padOffset);
      }

      Value ptrI8 = b.bitcast(smemObj.getBase(), i8PtrTy);
      Value advanced = b.gep(i8PtrTy, i8Ty, ptrI8, byteOffset,
                             LLVM::GEPNoWrapFlags::inbounds);
      outVals[idx] = b.bitcast(advanced, llvmPtrTy);
    }

    Value result =
        packLLElements(loc, typeConverter, outVals, rewriter, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void tle::populateLocalPointersOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalPointersOpConversion>(typeConverter, targetInfo, benefit);
}
