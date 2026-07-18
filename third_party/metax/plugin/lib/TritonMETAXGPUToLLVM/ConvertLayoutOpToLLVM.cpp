#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::LLVM::METAX::lowerLdStMatrix;

constexpr int kPtrBitWidth = 64;
struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const METAX::TargetInfo &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(
      LLVMTypeConverter &typeConverter, const METAX::TargetInfo &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    bool hasAttr = op->hasAttr(AttrSharedMemForceNoVec);
    if (!hasAttr) {
      return failure();
    }

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kReg = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (!llvm::is_contained(dims, kBlock) &&
        cvtNeedsSharedMemory(srcTy, dstTy)) {
      auto loc = op.getLoc();
      // Remove the kBlock dimension from the layout as it's the identity in the
      // cvt
      srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                      to_vector(srcLayout.getOutDimNames()));
      dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                      to_vector(dstLayout.getOutDimNames()));

      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

      Value result =
          packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, dstLayout,
                                       newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    // Remove broadcasting in src
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzling(loc, rewriter, prmtSrc, dstLayout,
                                          newInVals, llvmElemTy, smemBase);
    }

    // Remove broadcasting in dst
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth);
    // auto [smem, instr] =
    //     optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    // auto [idxSrc, idxDst] = instr;

    // TODO: use optimalSwizzling to support tile
    auto smem = optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth, true);
    auto idxSrc = 0;
    auto idxDst = 0;
    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    storeCvt = storeCvt.reshapeOuts({{kOffset, storeCvt.getTotalOutDimSize()}});
    loadCvt = loadCvt.reshapeOuts({{kOffset, loadCvt.getTotalOutDimSize()}});

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    auto noPaddingOffset = [](Value v) { return v; };
    bool isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        targetInfo.barrier(loc, rewriter, isWarpSync);

      auto tileInVals =
          to_vector(ArrayRef(permutedInVals).slice(i * tileSize, tileSize));
      // Store
      // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans
      if (idxSrc == 0) {
        lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                        noPaddingOffset, affineOffset, maskSpanAffineOffset,
                        rewriter, targetInfo);
      } else {
        assert(idxSrc == 1 || idxSrc == 2);
        bool transpose = idxSrc == 2;
        auto result = lowerLdStMatrix(
            loc, storeCvt, transpose, tileInVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      targetInfo.barrier(loc, rewriter, isWarpSync);
      // Load
      SmallVector<Value> tileOutVals;
      // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans
      if (idxDst == 0) {
        tileOutVals = lowerLdStShared(
            loc, ctx, loadCvt, {}, llvmElemTy, smemBase, noPaddingOffset,
            affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxDst == 1 || idxDst == 2);
        bool transpose = idxDst == 2;
        auto result = lowerLdStMatrix(
            loc, loadCvt, transpose, tileOutVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  LogicalResult
  transferWithinBlockSwizzling(ConvertLayoutOp op, Value src,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // Remove the kBlock dimension from the layout as it's the identity in the
    // cvt
    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(srcLayout.getOutDimNames()));
    dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, src, rewriter);
    auto outVals = transferWithinBlockSwizzling(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const METAX::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            srcLayout) &&
        isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            dstLayout)) {
      if (shouldUseDistSmem(srcLayout, dstLayout))
        return lowerDistToDistWithDistSmem(op, adaptor, rewriter, targetInfo);
    }

    return failure();
  }

private:
  LogicalResult
  lowerDistToDistWithDistSmem(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const METAX::TargetInfo &targetInfo) const {
    MLIRContext *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcShapePerCTA = getShapePerCTA(srcTy);
    auto srcCTAsPerCGA = triton::gpu::getCTAsPerCGA(srcLayout);
    auto srcCTAOrder = triton::gpu::getCTAOrder(srcLayout);
    unsigned rank = srcShapePerCTA.size();

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    smemBase = b.bitcast(smemBase, elemPtrTy);
    auto smemShape = convertType<unsigned, int64_t>(srcShapePerCTA);

    // Store to local shared memory
    {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto inIndices = emitIndices(loc, rewriter, targetInfo, srcLayout, srcTy,
                                   /*withCTAOffset*/ false);

      assert(inIndices.size() == inVals.size() &&
             "Unexpected number of indices emitted");

      for (unsigned i = 0; i < inIndices.size(); ++i) {
        Value offset = LLVM::linearize(rewriter, loc, inIndices[i], smemShape);
        Value ptr = b.gep(elemPtrTy, llvmElemTy, smemBase, offset);
        b.store(inVals[i], ptr);
      }
    }

    // Load from remote shared memory
    {
      SmallVector<Value> srcShapePerCTACache;
      for (unsigned i = 0; i < rank; ++i)
        srcShapePerCTACache.push_back(b.i32_val(srcShapePerCTA[i]));

      SmallVector<Value> outVals;
      auto outIndices = emitIndices(loc, rewriter, targetInfo, dstLayout, dstTy,
                                    /*withCTAOffset*/ true);

      for (unsigned i = 0; i < outIndices.size(); ++i) {
        auto coord = outIndices[i];
        assert(coord.size() == rank && "Unexpected rank of index emitted");

        SmallVector<Value> multiDimCTAId, localCoord;
        for (unsigned d = 0; d < rank; ++d) {
          multiDimCTAId.push_back(b.udiv(coord[d], srcShapePerCTACache[d]));
          localCoord.push_back(b.urem(coord[d], srcShapePerCTACache[d]));
        }

        Value remoteCTAId = LLVM::linearize(rewriter, loc, multiDimCTAId,
                                            srcCTAsPerCGA, srcCTAOrder);
        Value localOffset =
            LLVM::linearize(rewriter, loc, localCoord, smemShape);

        Value ptr = b.gep(elemPtrTy, llvmElemTy, smemBase, localOffset);
        outVals.push_back(targetInfo.loadDShared(rewriter, loc, ptr,
                                                 remoteCTAId, llvmElemTy,
                                                 /*pred=*/b.true_val()));
      }

      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
    }

    return success();
  }

private:
  const METAX::TargetInfo &targetInfo;
};

struct GVMArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GVMArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::GVMArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::GVMArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.arrive");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    num += (1 << 6);
    auto loc = op.getLoc();
    Value num_value = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, num);
    mlir::LLVM::createBuiltinFunc<triton::gpu::GVMArriveOp>(
        rewriter, loc, op, funcName, getVoidType(), {num_value});
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<triton::gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.barrier.inst");
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value voidVal = b.undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::BarrierOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::BarrierSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.barrier.shared");
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value voidVal = b.undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::BarrierSharedOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct SchedBoundOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::SchedBoundOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::SchedBoundOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::SchedBoundOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.schedbound.begin");
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value voidVal = b.undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::SchedBoundOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct IGLPOpConversion : public ConvertOpToLLVMPattern<triton::gpu::IGLPOp> {
  using ConvertOpToLLVMPattern<triton::gpu::IGLPOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::IGLPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.igroup.config");
    // config 0 : enable igroup optimization, default 0 (automatic by compiler).
    // config 1 : the number of prefetched lds instrs, default -1.
    // config 2 : the number of mma instrs between lds group, default -1.
    // config 3 : the number of mma instrs between sts group, default -1.
    // config 4 : the number of other instrs between mma instrs, default -1.
    // config 5 : the number of lds instrs inside lds group, default -1.
    // config 6 : the number of mma instrs between ldg instrs, default -1.
    // config 7 : the number of mma instrs between last lds and arrive, default
    // -1. [EXPERIMENTAL] when config 0 == 3:
    //   config 2 : the number of other instrs between lds group, default -1.
    //   config 3 : the number of other instrs between sts group, default -1.
    //   config 4 : the number of other instrs between mma instrs, default -1.
    //   config 5 : the number of other instrs between ldg instrs, default -1.
    auto config_0 = op->getAttrOfType<IntegerAttr>("config_0").getInt();
    auto config_1 = op->getAttrOfType<IntegerAttr>("config_1").getInt();
    auto config_2 = op->getAttrOfType<IntegerAttr>("config_2").getInt();
    auto config_3 = op->getAttrOfType<IntegerAttr>("config_3").getInt();
    auto config_4 = op->getAttrOfType<IntegerAttr>("config_4").getInt();
    auto config_5 = op->getAttrOfType<IntegerAttr>("config_5").getInt();
    auto config_6 = op->getAttrOfType<IntegerAttr>("config_6").getInt();
    auto config_7 = op->getAttrOfType<IntegerAttr>("config_7").getInt();
    auto loc = op.getLoc();
    Value val_0 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_0);
    Value val_1 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_1);
    Value val_2 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_2);
    Value val_3 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_3);
    Value val_4 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_4);
    Value val_5 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_5);
    Value val_6 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_6);
    Value val_7 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_7);
    ValueRange vals = {val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7};
    mlir::LLVM::createBuiltinFunc<triton::gpu::IGLPOp>(
        rewriter, loc, op, funcName, getVoidType(), vals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpConversion a higher benefit as it only matches
  // optimized or cross CTA cases
  // patterns.add<ConvertLayoutOpConversion,
  patterns.add<ConvertLayoutOpSwizzlingConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}

void mlir::triton::METAX::populateSyncOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GVMArriveOpConversion>(typeConverter, benefit);
  patterns.add<SchedBoundOpConversion>(typeConverter, benefit);
  patterns.add<IGLPOpConversion>(typeConverter, benefit);
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<BarrierSharedOpConversion>(typeConverter, benefit);
}
