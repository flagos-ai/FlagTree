#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSAGPUToLLVM/Allocation.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

using ::mlir::LLVM::linearize;

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static bool isMusaSqmmaLike(Attribute layout) {
  return isa<MUSASqmmaEncodingAttr>(layout);
}

static bool useMusaReplicatedScratch(Attribute srcLayout, Attribute dstLayout) {
  return (isMusaSqmmaLike(srcLayout) || isMusaSqmmaLike(dstLayout)) &&
         isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
             srcLayout) &&
         isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
             dstLayout);
}

static bool isSqmmaAccumulatorToBlockedLike(Attribute srcLayout,
                                            Attribute dstLayout) {
  return isa<MUSASqmmaEncodingAttr>(srcLayout) &&
         isa<BlockedEncodingAttr, SliceEncodingAttr>(dstLayout);
}

static bool useMusaSqmmaBlockSwizzling(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  if (!(isMusaSqmmaLike(srcTy.getEncoding()) ||
        isMusaSqmmaLike(dstTy.getEncoding())))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kBlock = str_attr("block");
  StringAttr kWarp = str_attr("warp");
  StringAttr kLane = str_attr("lane");
  auto dims = conversion.getInDimNames();

  if (llvm::is_contained(dims, kBlock))
    return false;
  if (llvm::is_contained(dims, kWarp))
    return true;
  if (llvm::is_contained(dims, kLane))
    return !cvtNeedsWarpShuffle(srcTy, dstTy);
  return false;
}

static bool useConservativeCarrierScratch(RankedTensorType srcTy,
                                          RankedTensorType dstTy) {
  MLIRContext *ctx = srcTy.getContext();
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();

  auto needsByteCarrier = [](Type ty) {
    return ty.isIntOrFloat() && ty.getIntOrFloatBitWidth() < 8;
  };
  bool isPointerCarrier = isa<triton::PointerType>(srcElemTy) &&
                          isa<triton::PointerType>(dstElemTy);
  bool isSubByteCarrier =
      needsByteCarrier(srcElemTy) && needsByteCarrier(dstElemTy);
  if (!isPointerCarrier && !isSubByteCarrier)
    return false;

  if (!isa<BlockedEncodingAttr, SliceEncodingAttr>(srcTy.getEncoding()) ||
      !isa<BlockedEncodingAttr, SliceEncodingAttr>(dstTy.getEncoding()))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  StringAttr kBlock = str_attr("block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

static bool isPlainBlockedLike(Attribute layout) {
  return isa<BlockedEncodingAttr, SliceEncodingAttr>(layout);
}

static bool useMusaGenericBlockSwizzling(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  if (isMusaSqmmaLike(srcTy.getEncoding()) ||
      isMusaSqmmaLike(dstTy.getEncoding()))
    return false;
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kBlock = str_attr("block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

static bool hasOnlyStorageLikeUsers(Value value) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (isa<triton::gpu::LocalAllocOp, triton::gpu::LocalStoreOp,
              triton::DescriptorStoreOp, triton::StoreOp>(user))
        continue;
      if (isa<triton::BitcastOp, triton::gpu::ConvertLayoutOp>(user)) {
        for (Value result : user->getResults())
          worklist.push_back(result);
        continue;
      }
      return false;
    }
  }
  return true;
}

static triton::gpu::LocalAllocOp findRootLocalAlloc(Value memDesc) {
  Value cur = memDesc;
  while (cur) {
    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      break;
    if (auto localAllocOp = dyn_cast<triton::gpu::LocalAllocOp>(defOp))
      return localAllocOp;
    if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(defOp)) {
      cur = indexOp.getSrc();
      continue;
    }
    if (auto subsliceOp = dyn_cast<triton::gpu::MemDescSubsliceOp>(defOp)) {
      cur = subsliceOp.getSrc();
      continue;
    }
    if (auto reinterpretOp =
            dyn_cast<triton::gpu::MemDescReinterpretOp>(defOp)) {
      cur = reinterpretOp.getSrc();
      continue;
    }
    if (auto transOp = dyn_cast<triton::gpu::MemDescTransOp>(defOp)) {
      cur = transOp.getSrc();
      continue;
    }
    if (auto reshapeOp = dyn_cast<triton::gpu::MemDescReshapeOp>(defOp)) {
      cur = reshapeOp.getSrc();
      continue;
    }
    break;
  }
  return {};
}

static FailureOr<Value> getDistributedSharedMemoryBase(
    Location loc, ConversionPatternRewriter &rewriter,
    const MUSA::TargetInfo &targetInfo, triton::gpu::ConvertLayoutOp op,
    Attribute srcLayout) {
  if (isMusaSqmmaLike(srcLayout))
    return LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                     op.getOperation());

  auto rootLocalAlloc = findRootLocalAlloc(op.getSrc());
  if (rootLocalAlloc && rootLocalAlloc->hasAttr("allocation.offset"))
    return LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                     rootLocalAlloc.getOperation());

  if (op->hasAttr("allocation.offset"))
    return LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                     op.getOperation());

  return rewriter.notifyMatchFailure(
      op, "expected allocation.offset on convert_layout or root local_alloc "
          "for MUSA shared layout conversion");
}

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                            const MUSA::TargetInfo &targetInfo,
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

    if (useMusaReplicatedScratch(srcLayout, dstLayout)) {
      if (isSqmmaAccumulatorToBlockedLike(srcLayout, dstLayout))
        return lowerSqmmaBlockSwizzling(op, adaptor, rewriter);
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }

    if (useConservativeCarrierScratch(srcTy, dstTy))
      return lowerDistributedToDistributed(op, adaptor, rewriter);

    if (useMusaSqmmaBlockSwizzling(srcTy, dstTy))
      return lowerSqmmaBlockSwizzling(op, adaptor, rewriter);

    if (useMusaGenericBlockSwizzling(srcTy, dstTy))
      return lowerGenericBlockSwizzling(op, adaptor, rewriter);

    return failure();
  }

private:
  SmallVector<Value> transferWithinBlockSwizzlingImpl(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase,
      bool separateRepScratch) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, dstLayout, newInVals, llvmElemTyPtr,
          smemBase, separateRepScratch);
      for (auto &v : outVals)
        v = b.inttoptr(llvmElemTy, v);
      return outVals;
    }

    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase,
          separateRepScratch);
      for (auto &v : outVals)
        v = b.trunc(llvmElemTy, v);
      return outVals;
    }

    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzlingImpl(loc, rewriter, prmtSrc, dstLayout,
                                              newInVals, llvmElemTy, smemBase,
                                              separateRepScratch);
    }

    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase,
          separateRepScratch);
      return broadcastAs(outVals, dstLayout);
    }

    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth);
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;
    assert(idxSrc == 0 && idxDst == 0 &&
           "MUSA generic swizzling currently supports shared ld/st tiles only");

    auto kReg = str_attr("register");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    storeCvt = storeCvt.reshapeOuts({{kOffset, storeCvt.getTotalOutDimSize()}});
    loadCvt = loadCvt.reshapeOuts({{kOffset, loadCvt.getTotalOutDimSize()}});

    auto tileSize = storeCvt.getInDimSize(kReg);
    assert(permutedInVals.size() == tileSize * nReps);

    SmallVector<Value> outVals;
    auto maskSpanAffineOffset = 0;

    bool isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0) {
        if (isWarpSync)
          targetInfo.warpSync(loc, rewriter);
        else
          targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      }

      auto tileInVals =
          ArrayRef<Value>(permutedInVals).slice(i * tileSize, tileSize);
      Value affineOffset = separateRepScratch
                               ? b.i32_val(i * storeCvt.getTotalOutDimSize())
                               : b.i32_val(0);
      lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                      /*paddingShifts=*/{}, affineOffset, maskSpanAffineOffset,
                      rewriter, targetInfo);

      if (isWarpSync)
        targetInfo.warpSync(loc, rewriter);
      else
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

      SmallVector<Value> tileOutVals = lowerLdStShared(
          loc, ctx, loadCvt, {}, llvmElemTy, smemBase, /*paddingShifts=*/{},
          affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      llvm::append_range(outVals, tileOutVals);
    }

    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  LogicalResult
  lowerGenericBlockSwizzling(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    SmallVector<StringAttr> inDims{kReg, kLane, kWarp};
    srcLayout =
        srcLayout.sublayout(inDims, to_vector(srcLayout.getOutDimNames()));
    dstLayout =
        dstLayout.sublayout(inDims, to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
    bool separateRepScratch =
        musa_gpu::needsMusaRepDisjointGenericScratch(srcTy, dstTy, targetInfo);
    auto outVals = transferWithinBlockSwizzlingImpl(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase,
        separateRepScratch);
    Value result = ::mlir::packLLElements(loc, getTypeConverter(), outVals,
                                          rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  lowerSqmmaBlockSwizzling(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    SmallVector<StringAttr> inDims{kReg, kLane, kWarp};
    srcLayout =
        srcLayout.sublayout(inDims, to_vector(srcLayout.getOutDimNames()));
    dstLayout =
        dstLayout.sublayout(inDims, to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);

    if (isMusaSqmmaLike(srcTy.getEncoding()))
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

    auto outVals = transferWithinBlockSwizzlingImpl(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase,
        /*separateRepScratch=*/false);
    Value result = ::mlir::packLLElements(loc, getTypeConverter(), outVals,
                                          rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (product<unsigned>(convertType<unsigned>(dstTy.getShape())) == 1) {
      auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
      SmallVector<Value> outVals(getTotalElemsPerThread(dstTy), inVals[0]);
      Value result =
          ::mlir::packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto shapePerCTA = convertType<unsigned>(getShapePerCTA(srcTy));
    unsigned rank = dstTy.getRank();

    SmallVector<unsigned> repShape(shapePerCTA.begin(), shapePerCTA.end());
    SmallVector<unsigned> numReplicates(rank, 1);
    auto order = getOrder(dstTy);
    auto dstElemTy = dstTy.getElementType();
    auto llvmElemTy = typeConverter->convertType(dstElemTy);
    bool isPtr = isa<triton::PointerType>(dstElemTy);
    bool useByteCarrier =
        dstElemTy.isIntOrFloat() && dstElemTy.getIntOrFloatBitWidth() < 8;
    Type llvmElemStorageTy = llvmElemTy;
    if (isPtr)
      llvmElemStorageTy = i64_ty;
    else if (useByteCarrier)
      llvmElemStorageTy = i8_ty;
    auto elemPtrTy = ptr_ty(ctx, 3);

    auto srcIndices = emitIndices(loc, rewriter, targetInfo, srcLayout, srcTy,
                                  /*withCTAOffset=*/false);
    auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (isPtr) {
      for (Value &inVal : inVals)
        inVal = b.ptrtoint(i64_ty, inVal);
    } else if (useByteCarrier) {
      for (Value &inVal : inVals)
        inVal = b.zext(i8_ty, inVal);
    }
    assert(srcIndices.size() == inVals.size() &&
           "unexpected source index/value mismatch");

    auto smemBaseOr = getDistributedSharedMemoryBase(loc, rewriter, targetInfo,
                                                     op, srcLayout);
    if (failed(smemBaseOr))
      return failure();
    Value smemBase = *smemBaseOr;
    Value typedSmemBase = b.bitcast(smemBase, elemPtrTy);

    auto dstIndices = emitIndices(loc, rewriter, targetInfo, dstLayout, dstTy,
                                  /*withCTAOffset=*/false);
    SmallVector<Value> outVals(
        dstIndices.size(),
        LLVM::UndefOp::create(rewriter, loc, llvmElemStorageTy));

    unsigned numTotalReps = product<unsigned>(numReplicates);
    if (numTotalReps != 0 && isMusaSqmmaLike(srcLayout))
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    for (unsigned repId = 0; repId < numTotalReps; ++repId) {
      if (repId != 0)
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

      auto multiDimRepId = delinearize(repId, numReplicates, order);
      SmallVector<Value> repBase(rank);
      SmallVector<Value> repLimit(rank);
      for (unsigned d = 0; d < rank; ++d) {
        repBase[d] = b.i32_val(multiDimRepId[d] * repShape[d]);
        repLimit[d] = b.i32_val(multiDimRepId[d] * repShape[d] + repShape[d]);
      }

      for (unsigned i = 0; i < srcIndices.size(); ++i) {
        Value inRep = b.true_val();
        SmallVector<Value> localCoord(rank);
        for (unsigned d = 0; d < rank; ++d) {
          Value ge = b.icmp_sge(srcIndices[i][d], repBase[d]);
          Value lt = b.icmp_slt(srcIndices[i][d], repLimit[d]);
          inRep = b.and_(inRep, b.and_(ge, lt));
          localCoord[d] = b.sub(srcIndices[i][d], repBase[d]);
        }

        Value offset = linearize(rewriter, loc, localCoord, repShape, order);
        Value ptr = b.gep(elemPtrTy, llvmElemStorageTy, typedSmemBase, offset);
        LLVM::MUSA::llStore(rewriter, loc, ptr, inVals[i], inRep);
      }

      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

      for (unsigned i = 0; i < dstIndices.size(); ++i) {
        Value inRep = b.true_val();
        SmallVector<Value> localCoord(rank);
        for (unsigned d = 0; d < rank; ++d) {
          Value ge = b.icmp_sge(dstIndices[i][d], repBase[d]);
          Value lt = b.icmp_slt(dstIndices[i][d], repLimit[d]);
          inRep = b.and_(inRep, b.and_(ge, lt));
          localCoord[d] = b.sub(dstIndices[i][d], repBase[d]);
        }

        Value offset = linearize(rewriter, loc, localCoord, repShape, order);
        Value ptr = b.gep(elemPtrTy, llvmElemStorageTy, typedSmemBase, offset);
        outVals[i] = LLVM::MUSA::llLoad(rewriter, loc, ptr, llvmElemStorageTy,
                                        inRep, outVals[i]);
      }
    }

    if (isPtr) {
      for (Value &outVal : outVals)
        outVal = b.inttoptr(llvmElemTy, outVal);
    } else if (useByteCarrier) {
      for (Value &outVal : outVals)
        outVal = b.trunc(llvmElemTy, outVal);
    }

    Value result =
        ::mlir::packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const MUSA::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::MUSA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 1));
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
