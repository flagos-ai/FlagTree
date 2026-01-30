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

static SmallVector<Value>
lowerLocalPointers(Location loc, MLIRContext *ctx, LinearLayout cvt,
                   Type llvmElemTy, LLVM::LLVMPointerType llvmPtrTy,
                   ttg::MemDescType srcTy, SharedMemoryObject smemObj,
                   RewriterBase &rewriter, const TargetInfoBase &targetInfo) {
  assert(cvt.getNumOutDims() == 1);
  assert(*cvt.getOutDimNames().begin() == str_attr("offset"));

  auto calcPaddedOffset = [&](Value smemOffset) {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    if (auto paddedEnc =
            dyn_cast<ttg::PaddedSharedEncodingAttr>(srcTy.getEncoding())) {
      Value padOffset =
          emitPadding(loc, rewriter, paddedEnc, bitwidth, smemOffset,
                      /*offsetInBytes=*/true);
      smemOffset = b.add(smemOffset, padOffset);
    }
    return smemOffset;
  };

  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto outVals = lowerLocalPointers(loc, ctx, prmtCvt, llvmElemTy, llvmPtrTy,
                                      srcTy, smemObj, rewriter, targetInfo);
    outVals = broadcastAs(outVals, cvt);
    return outVals;
  }

  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
  assert(bitwidth % 8 == 0 && "local pointers expect byte-addressable src");

  auto emitPointers = [&](RewriterBase &rewriter, Location loc, ArrayRef<Value>,
                          Value shmemAddr, int,
                          VectorType vecTy) -> SmallVector<Value> {
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> ptrVals;
    ptrVals.reserve(vecTy.getNumElements());
    int stride = bitwidth / 8;
    for (int idx = 0; idx < vecTy.getNumElements(); ++idx) {
      Value offset = b.i32_val(idx * stride);
      Value elemAddr = b.gep(shmemAddr.getType(), i8_ty, shmemAddr, offset,
                             LLVM::GEPNoWrapFlags::inbounds);
      ptrVals.push_back(b.bitcast(elemAddr, llvmPtrTy));
    }
    return ptrVals;
  };

  return lowerLdSt(loc, ctx, cvt, {}, llvmElemTy, smemObj.getBase(),
                   calcPaddedOffset, affineOffset, maskSpanAffineOffset, laneId,
                   warpId, rewriter, targetInfo, std::nullopt, emitPointers);
}

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
        return rewriter.create<LLVM::SExtOp>(loc, i32Ty, v);
      }
      return Value();
    };

    SmallVector<Value> flattenedOffsets;
    bool hasTensorOffsets = false;

    if (Value offsetsTensor = adaptor.getOffsets()) {
      hasTensorOffsets = true;
      flattenedOffsets = unpackLLElements(loc, offsetsTensor, rewriter);
      for (auto &elt : flattenedOffsets) {
        elt = ensureI32(elt);
        if (!elt)
          return reportFailure("index tensor must lower to i32 scalars");
      }
    }

    auto sharedEnc = cast<ttg::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    if (!resultTy.getEncoding())
      return reportFailure("local_pointers result must carry an encoding");

    LinearLayout regLayout = ttg::toLinearLayout(resultTy);
    LinearLayout cvt = LinearLayout::empty();
    if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEnc)) {
      cvt = ttg::getPaddedRegToSharedLayout(regLayout, paddedEnc);
    } else {
      auto sharedLayout = ttg::toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
      auto kBlock = str_attr("block");
      if (!cvt.isTrivialOver({kBlock})) {
        return reportFailure("shared layout must be block-invariant");
      }
    }
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    auto outVals = lowerLocalPointers(loc, ctx, cvt, llvmElemTy, llvmPtrTy,
                                      memDescTy, smemObj, rewriter, targetInfo);

    if (hasTensorOffsets) {
      if (flattenedOffsets.empty())
        return reportFailure("index tensor must contain at least one value");
      if (flattenedOffsets.size() != outVals.size()) {
        llvm::errs() << "  flattenedOffsets.size=" << flattenedOffsets.size()
                     << " outVals.size=" << outVals.size() << "\n";
        return reportFailure(
            "index tensor shape must match pointer result shape");
      }

      TritonLLVMOpBuilder b(loc, rewriter);
      int elemBits = llvmElemTy.getIntOrFloatBitWidth();
      assert(elemBits % 8 == 0 && "element bitwidth must be byte addressable");
      int elemBytes = elemBits / 8;
      Value elemBytesVal =
          elemBytes > 1 ? b.i32_val(static_cast<int32_t>(elemBytes)) : Value();
      auto i8Ty = IntegerType::get(ctx, 8);
      auto i8PtrTy =
          LLVM::LLVMPointerType::get(ctx, llvmPtrTy.getAddressSpace());

      SmallVector<unsigned> bufferShape;
      for (int64_t dim : memDescTy.getShape())
        bufferShape.push_back(static_cast<unsigned>(dim));
      SmallVector<unsigned> bufferOrder;
      bufferOrder.reserve(bufferShape.size());
      for (int i = static_cast<int>(bufferShape.size()) - 1; i >= 0; --i)
        bufferOrder.push_back(static_cast<unsigned>(i));

      auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
      auto kBlock = str_attr("block");

      for (size_t idx = 0; idx < outVals.size(); ++idx) {
        SmallVector<std::pair<StringAttr, Value>> inIndices;
        inIndices.reserve(regLayout.getNumInDims());
        for (auto dimName : regLayout.getInDimNames()) {
          if (dimName == kReg) {
            inIndices.push_back(
                {dimName, b.i32_val(static_cast<int32_t>(idx))});
          } else if (dimName == kLane) {
            inIndices.push_back({dimName, laneId});
          } else if (dimName == kWarp) {
            inIndices.push_back({dimName, warpId});
          } else if (dimName == kBlock) {
            inIndices.push_back({dimName, b.i32_val(0)});
          } else {
            inIndices.push_back({dimName, b.i32_val(0)});
          }
        }

        auto logicalCoords =
            applyLinearLayout(loc, rewriter, regLayout, inIndices);
        SmallVector<Value> coordVals;
        coordVals.reserve(logicalCoords.size());
        for (auto &pair : logicalCoords)
          coordVals.push_back(pair.second);

        Value elemLinear =
            LLVM::linearize(rewriter, loc, coordVals, bufferShape, bufferOrder);
        Value deltaElems = b.sub(flattenedOffsets[idx], elemLinear);
        if (elemBytes > 1)
          deltaElems = b.mul(deltaElems, elemBytesVal);
        Value ptrI8 = b.bitcast(outVals[idx], i8PtrTy);
        Value advanced = b.gep(i8PtrTy, i8Ty, ptrI8, deltaElems,
                               LLVM::GEPNoWrapFlags::inbounds);
        outVals[idx] = b.bitcast(advanced, llvmPtrTy);
      }
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
