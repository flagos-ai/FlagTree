#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
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
using mlir::LLVM::NVIDIA::lowerLdStMatrix;

constexpr int kPtrBitWidth = 64;
constexpr int kMovmatrixMinComputeCapability = 75;

// Recognize the one warp-local register permutation implemented more directly
// by movmatrix.m8n8.trans.b16 than by the generic shuffle lowering. The PTX
// instruction is available from SM75; the MMA-v2 layout checks below are
// additional constraints of this conversion, not hardware type restrictions.
static bool
canLowerMmaCToDotBWithMovmatrix(RankedTensorType srcTy, RankedTensorType dstTy,
                                const NVIDIA::TargetInfo &targetInfo) {
  auto srcMma = dyn_cast<NvidiaMmaEncodingAttr>(srcTy.getEncoding());
  auto dstDot = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!srcMma || !dstDot || !srcMma.isAmpere() || dstDot.getOpIdx() != 1 ||
      dstDot.getKWidth() != 2 || dstDot.getParent() != srcMma ||
      !(srcTy.getElementType().isBF16() || srcTy.getElementType().isF16()) ||
      srcTy.getRank() != 2 ||
      targetInfo.getComputeCapability() < kMovmatrixMinComputeCapability ||
      cvtNeedsSharedMemory(srcTy, dstTy))
    return false;

  auto shape = srcTy.getShape();
  return srcMma.getInstrShape() == ArrayRef<unsigned>({16, 8}) &&
         shape[0] % 16 == 0 && shape[1] % 16 == 0;
}

struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const NVIDIA::TargetInfo &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(
      LLVMTypeConverter &typeConverter, const NVIDIA::TargetInfo &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

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
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;

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
                            const NVIDIA::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (succeeded(lowerMmaCToDotBWithMovmatrix(op, adaptor, rewriter)))
      return success();

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
  lowerMmaCToDotBWithMovmatrix(triton::gpu::ConvertLayoutOp op,
                               OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    if (!canLowerMmaCToDotBWithMovmatrix(srcTy, dstTy, targetInfo))
      return failure();

    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.size() != getTotalElemsPerThread(dstTy) ||
        inVals.size() % 2 != 0)
      return failure();

    Type elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    Type pairTy = vec_ty(elemTy, 2);
    auto pairLocations = [&](const LinearLayout &layout) {
      StringAttr kRegister = str_attr("register");
      StringAttr kLane = str_attr("lane");
      StringAttr kWarp = str_attr("warp");
      StringAttr kBlock = str_attr("block");
      SmallVector<std::pair<uint64_t, unsigned>> locations;
      for (unsigned i = 0; i < inVals.size(); i += 2) {
        auto indices = layout.apply({{kRegister, static_cast<int32_t>(i)},
                                     {kLane, 0},
                                     {kWarp, 0},
                                     {kBlock, 0}});
        uint64_t tileRow = indices[0].second / 16;
        uint64_t tileCol = indices[1].second / 8;
        uint64_t tile = (tileRow << 32) | tileCol;
        unsigned ordinal = llvm::count_if(locations, [tile](auto location) {
          return location.first == tile;
        });
        locations.push_back({tile, ordinal});
      }
      return locations;
    };
    auto srcLocations = pairLocations(toLinearLayout(srcTy));
    auto dstLocations = pairLocations(toLinearLayout(dstTy));

    SmallVector<unsigned> dstPairForSrc(srcLocations.size());
    for (auto [srcIndex, srcLocation] : llvm::enumerate(srcLocations)) {
      auto dstIt = llvm::find(dstLocations, srcLocation);
      if (dstIt == dstLocations.end())
        return failure();
      dstPairForSrc[srcIndex] = std::distance(dstLocations.begin(), dstIt);
    }

    SmallVector<Value> outVals(inVals.size());
    for (unsigned i = 0; i < inVals.size(); i += 2) {
      Value pair = packLLVector(loc, ArrayRef(inVals).slice(i, 2), rewriter);
      Value packed = b.bitcast(pair, i32_ty);

      PTXBuilder ptxBuilder;
      auto *dst = ptxBuilder.newOperand("=r", /*init=*/false);
      auto *src = ptxBuilder.newOperand(packed, "r");
      auto &movmatrix =
          *ptxBuilder.create("movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
      movmatrix({dst, src}, /*onlyAttachMLIRArgs=*/true);
      Value moved = ptxBuilder.launch(rewriter, loc, i32_ty);

      Value movedPair = b.bitcast(moved, pairTy);
      unsigned dstPair = dstPairForSrc[i / 2];
      outVals[2 * dstPair] = b.extract_element(movedPair, b.i32_val(0));
      outVals[2 * dstPair + 1] = b.extract_element(movedPair, b.i32_val(1));
    }

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  lowerDistToDistWithDistSmem(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const NVIDIA::TargetInfo &targetInfo) const {
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

    // Cluster barrier
    triton::nvidia_gpu::ClusterArriveOp::create(rewriter, loc, false);
    triton::nvidia_gpu::ClusterWaitOp::create(rewriter, loc);

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

    // Cluster barrier
    triton::nvidia_gpu::ClusterArriveOp::create(rewriter, loc, false);
    triton::nvidia_gpu::ClusterWaitOp::create(rewriter, loc);

    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpConversion a higher benefit as it only matches
  // optimized or cross CTA cases
  patterns.add<ConvertLayoutOpConversion, ConvertLayoutOpSwizzlingConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
