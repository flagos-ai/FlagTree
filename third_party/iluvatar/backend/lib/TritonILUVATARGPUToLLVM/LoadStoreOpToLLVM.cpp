#include "AtomicRMWOpsEmitter.h"
#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::ILUVATAR::getVectorSize;
using ::mlir::LLVM::ILUVATAR::llLoad;
using ::mlir::LLVM::ILUVATAR::llStore;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

std::optional<const char *> getMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  // The default LLVM sync scope is "system", so no string is
  // provided here
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

std::pair<bool, bool> getOrderingFlags(MemSemantic memOrdering) {
  bool emitReleaseFence = false;
  bool emitAcquireFence = false;
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    // In this case, no memory fences are needed
    break;
  case MemSemantic::RELEASE:
    emitReleaseFence = true;
    break;
  case MemSemantic::ACQUIRE:
    emitAcquireFence = true;
    break;
  case MemSemantic::ACQUIRE_RELEASE:
    emitAcquireFence = true;
    emitReleaseFence = true;
  default:
    // default == acq_rel, so we emit the same barriers
    emitAcquireFence = true;
    emitReleaseFence = true;
  }
  return {emitAcquireFence, emitReleaseFence};
}

LogicalResult emitFence(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc, MemSemantic memOrdering,
                        MemSyncScope memScope, bool preAtomic) {
  auto [emitReleaseFence, emitAcquireFence] = getOrderingFlags(memOrdering);
  if (MemSyncScope::SYSTEM == memScope)
    return rewriter.notifyMatchFailure(
        op, "System memory scope is not supported for Buffer Atomic Ops");
  auto scopeStr = getMemScopeStr(memScope);
  if (!scopeStr)
    return rewriter.notifyMatchFailure(
        op, "Unsupported memory scope for Buffer Atomic Ops");

  StringAttr scope = mlir::StringAttr::get(loc.getContext(), *scopeStr);

  if (emitReleaseFence && preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::acquire, scope);
  }
  return success();
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const ILUVATAR::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

std::pair<Block *, Block *> emitBranch(RewriterBase &rewriter, Location loc,
                                       Value cond) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *after =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *body = rewriter.createBlock(after);
  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, cond, body, after);
  rewriter.setInsertionPointToStart(body);
  LLVM::BrOp::create(rewriter, loc, after);
  rewriter.setInsertionPointToStart(body);
  return {body, after};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const ILUVATAR::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = LLVM::ConstantOp::create(builder, loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(RewriterBase &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Return a tensor of pointers with the same type of `basePtr` and the same
  // shape of `offset`
  Type getPointerTypeWithShape(Value basePtr, Value offset) const {
    Type basePtrType = basePtr.getType();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    return offsetType.cloneWith(std::nullopt, basePtrType);
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const ILUVATAR::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Contains some helper functions for direct to lds loads.
struct DirectToLdsLoadConversionBase : public LoadStoreConversionBase {
  explicit DirectToLdsLoadConversionBase(
      const ILUVATAR::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // direct to lds loads do not support per lane shared offsets. We need to
  // ensure that we write coalesced into shared memory. This means we cannot
  // exceed the supported load width because splitting them would cause strided
  // (non coalesced) writes. Additionally:
  //   1) For *non* swizzled shared encodings we check if they result in
  //      coalesced writes and can then lower them directly to the intrinsics.
  //   2) For swizzled shared encodings we need to transfer the swizzling to the
  //      source pointers. For now this is done by swizzling the pointers
  //      between the lane of a warp via permute. This only works if the swizzle
  //      pattern does not exchange elements between warps which holds for all
  //      our swizzle patterns. There is still a check performed to not silently
  //      produce wrong results if we invalidate the condition in the future
  LogicalResult canWriteCoalesced(RewriterBase &rewriter, Operation *op,
                                  RankedTensorType srcTy, MemDescType dstTy,
                                  unsigned vectorSize,
                                  bool requiresSrcPtrSwizzling) const {
    if (targetInfo.supportsDirectToLDSScattering()) {
      return success();
    }

    int vecBits = vectorSize * dstTy.getElementTypeBitWidth();
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(vecBits)) {
      LDBG(*op << " results in unsupported load bitwidth: " << vecBits);
      return failure();
    }
    // Compute the blocked -> shared linear layout to check preconditions
    LinearLayout srcLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

    unsigned threadsPerWarp = lookupThreadsPerWarp(rewriter);
    if (!requiresSrcPtrSwizzling &&
        !LLVM::ILUVATAR::canCoalesceWriteIntoSharedMemory(
            rewriter, srcToSharedLayout, threadsPerWarp, vectorSize)) {
      LDBG(*op << " does not write coalesced into LDS and is not swizzled");
      return failure();
    }

    if (requiresSrcPtrSwizzling &&
        !LLVM::ILUVATAR::doesSwizzleInsideWarp(rewriter, srcToSharedLayout,
                                               threadsPerWarp)) {
      LDBG(*op << " does swizzle across warp boundaries");
      return failure();
    }
    return success();
  }

  // For each load emit the computation to get the lane id offset which holds
  // the source pointers/offsets we need to store to shared memory
  SmallVector<Value>
  emitSwizzledLaneOffsets(RewriterBase &rewriter, Operation *op,
                          RankedTensorType srcTy, MemDescType swizzledTy,
                          MemDescType flatTy, Value llDst, Type resElemTy,
                          unsigned vec) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Create regToShared layout for the swizzled and flat encoding
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    auto sharedSwizz = triton::gpu::toLinearLayout(swizzledTy);
    auto sharedFlat = triton::gpu::toLinearLayout(flatTy);

    auto regToSharedSwizzled = regLayout.invertAndCompose(sharedSwizz);
    auto regToSharedFlat = regLayout.invertAndCompose(sharedFlat);

    MLIRContext *ctx = rewriter.getContext();
    StringAttr kBlock = str_attr("block");
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value blockId = b.i32_val(0);

    int numberOfLoads = regToSharedSwizzled.getInDimSize(kRegister) / vec;

    // For each load compute the difference between the flat and the swizzled
    // linear offsets into shared memory
    // TODO (alex): this is only correct as long as the lds view is a contiguous
    // block. So this can break if we slice along the 2 minor dimensions
    SmallVector<Value> swizzledOffsets;
    swizzledOffsets.reserve(numberOfLoads);
    auto vecVal = b.i32_val(vec);
    for (int i = 0; i < numberOfLoads; i++) {
      auto regId = b.i32_val(i * vec);

      std::array<std::pair<StringAttr, Value>, 4> indices{{
          {kRegister, regId},
          {kLane, laneId},
          {kWarp, warpId},
          {kBlock, blockId},
      }};

      Value swizzledOffset =
          applyLinearLayout(loc, rewriter, regToSharedSwizzled, indices)[0]
              .second;
      Value flatOffset =
          applyLinearLayout(loc, rewriter, regToSharedFlat, indices)[0].second;

      // Normalize the offset by vecTy to obtain the offset in lanes
      auto laneOffet = b.sdiv(b.sub(swizzledOffset, flatOffset), vecVal);
      swizzledOffsets.push_back(laneOffet);
    }
    return swizzledOffsets;
  }

  // Swizzle the mask (1bit) based on selectLane via ballot
  Value shuffleMask(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                    Location loc, const TargetInfoBase &targetInfo,
                    Value selectLane, Value mask) const {
    auto warpMask =
        targetInfo.ballot(rewriter, loc, rewriter.getI64Type(), mask);
    // Extract the selectLane bit
    auto bitMask = b.lshr(warpMask, b.zext(rewriter.getI64Type(), selectLane));
    return b.trunc(i1_ty, bitMask);
  }

  SmallVector<Value> zipLoadValues(RewriterBase &rewriter, Location loc,
                                   unsigned vec, ArrayRef<Value> srcElems,
                                   Type srcTy, ArrayRef<Value> maskElems,
                                   ArrayRef<Value> otherElems, Type otherTy,
                                   ArrayRef<Value> swizzledLaneOffsets) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadVals;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), ArrayRef<Type>{srcTy, i1_ty, otherTy, i32_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
      // src
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      // mask
      auto maskElem = maskElems.empty() ? b.true_val() : maskElems[i];
      packedArr = b.insert_val(packedArr, maskElem, 1);
      // other
      if (!otherElems.empty())
        packedArr = b.insert_val(packedArr, otherElems[i], 2);
      // swizzleOffset are per vec so we need to duplicate values vec times
      auto swizzleOffset = swizzledLaneOffsets.empty()
                               ? b.i32_val(0)
                               : swizzledLaneOffsets[i / vec];
      packedArr = b.insert_val(packedArr, swizzleOffset, 3);

      loadVals.push_back(packedArr);
    }
    return loadVals;
  }

  auto unzipLoadValues(RewriterBase &rewriter, Location loc, int startIdx,
                       ArrayRef<Value> values, Type srcTy, Type otherTy,
                       bool hasOther, unsigned vec) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto structElem = values[startIdx];
    Value offsetElem = b.extract_val(srcTy, structElem, 0);
    Value maskElem = b.extract_val(i1_ty, structElem, 1);
    // Gather other elements
    SmallVector<Value> otherElems;
    if (hasOther) {
      for (int i = 0; i < vec; i++) {
        otherElems.push_back(b.extract_val(otherTy, values[startIdx + i], 2));
      }
    }

    Value swizzleLaneOffset = b.extract_val(i32_ty, structElem, 3);

    return std::make_tuple(offsetElem, maskElem, std::move(otherElems),
                           swizzleLaneOffset);
  }

  void applySwizzling(RewriterBase &rewriter, Location loc, Value &srcOrOffset,
                      Value &mask, Value laneId,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    // laneId + swizzleOffset will always stay inside the warp [0,
    // threadsPerWarp) because we only swizzle inside a warp
    Value swizzledLaneId = b.add(laneId, swizzleLaneOffset);
    // Shuffle based on swizzleLaneId to apply the swizzling
    srcOrOffset =
        targetInfo.shuffleIdx(rewriter, loc, srcOrOffset, swizzledLaneId);

    if (mask) {
      mask = shuffleMask(rewriter, b, loc, targetInfo, swizzledLaneId, mask);
    }
  }

  LogicalResult lowerDirectToLDSLoad(
      RewriterBase &rewriter, Location loc, RankedTensorType srcTy,
      MemDescType dstTy, SmallVector<Value> loadVals, Value llDst,
      Type resElemTy, unsigned vec,
      std::function<SmallVector<Value>(RewriterBase &, Location,
                                       ArrayRef<Value>, Value, int, VectorType,
                                       Value)>
          lowerInst) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();

    // Build src to shared layout and remove broadcasted registers
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    loadVals = removeBroadcastSrc.apply(loadVals);

    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto cvt = srcLayout.invertAndCompose(sharedLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(
          loc,
          "direct to lds loads do not support non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    Value ctaMulticastMask;

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);

    Value laneId, warpId;
    std::tie(laneId, warpId) = getLaneAndWarpId(rewriter, loc);

    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto bitwidth = dstTy.getElementTypeBitWidth();
      if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              dstTy.getEncoding())) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitwidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };

    auto lowerInstForwardMulticastMask =
        [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
            Value shmemAddr, int idx, VectorType vecTy) {
          return lowerInst(rewriter, loc, vals, shmemAddr, idx, vecTy,
                           ctaMulticastMask);
        };

    // If we do not support scattering the address should be the start
    // address (scalar) of the warp
    laneId = targetInfo.supportsDirectToLDSScattering() ? laneId : b.i32_val(0);
    lowerLdSt(loc, ctx, cvt, loadVals, resElemTy, smemObj.getBase(),
              calcPaddedOffset, affineOffset, maskSpanAffineOffset, laneId,
              warpId, rewriter, targetInfo, vec, lowerInstForwardMulticastMask);
    return success();
  }

  void emitOtherStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, VectorType vecTy,
                      Value mask, ArrayRef<Value> otherElems, Value shmemAddr,
                      Value laneId, bool requiresSrcPtrSwizzling,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value storeVal = packElementRangeIntoVector(rewriter, typeConverter, loc,
                                                vecTy, otherElems, 0);
    Type ptrTy = shmemAddr.getType();
    Value ldsAddr = shmemAddr;
    // When scattering is unsupported, shmemAddr is the warp base address.
    // Use shmemAddr + lane_id [+ swizzleOffset] to compute each lane's address.
    if (!targetInfo.supportsDirectToLDSScattering()) {
      ldsAddr = b.gep(ptrTy, vecTy, shmemAddr, laneId);
      if (requiresSrcPtrSwizzling)
        ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzleLaneOffset);
    }
    llStore(rewriter, loc, ldsAddr, storeVal, b.icmp_ne(mask, b.true_val()),
            CacheModifier::NONE, targetInfo.requiresAliasInfoForAsyncOps());
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const ILUVATAR::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // no mask use sme, only pass-through begin ptr
    auto loadresTy = dyn_cast<RankedTensorType>(valueTy);

    if (loadresTy) {
      auto loadEncoding =
          mlir::dyn_cast<BlockedEncodingAttr>(loadresTy.getEncoding());
      if (loadEncoding && loadEncoding.getIsSme()) {
        SmallVector<Value> loadedVals;
        unsigned numElementsPerThread = getTotalElemsPerThread(valueTy);
        for (int i = 0; i < numElementsPerThread; i++) {
          loadedVals.push_back(ptrElems[0]);
        }
        Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);

        Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                            rewriter, llvmResultStructTy);
        rewriter.replaceOp(op, {resultStruct});
        return success();
      }
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    Value multicastMask;
    auto mod = op->getParentOfType<ModuleOp>();
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs > 1) {
      Value clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
      auto regLayout =
          triton::gpu::toLinearLayout(cast<RankedTensorType>(ptr.getType()));
      multicastMask = LLVM::ILUVATAR::emitCtaMulticastMask(
          rewriter, loc, clusterCTAId, regLayout);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    bool isVolatile = op.getIsVolatile();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);

      Value loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal,
                             multicastMask, cacheMod, false, isVolatile);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const ILUVATAR::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const ILUVATAR::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op, "Unknown memory ordering");
    auto scope = op.getScope();
    auto scopeStr = getMemScopeStr(scope);
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unknown memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);

    bool isFp32 = valueElemTy.isF32();
    bool isFp16 = valueElemTy.isF16() || valueElemTy.isBF16();
    Type atomicValTy = valueElemTy;
    if (isFp32) {
      atomicValTy = i32_ty;
    }
    if (isFp16) {
      atomicValTy = i16_ty;
    }

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      if (isFp32 || isFp16) {
        casCmp = b.bitcast(casCmp, atomicValTy);
        casVal = b.bitcast(casVal, atomicValTy);
      }

      // use op
      if (tensorTy) { // for tensor
        auto retType = valueElemTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = *atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, StringRef(scopeStr.value()));

        // Extract the new_loaded value from the pair.
        Value ret = b.extract_val(atomicValTy, cmpxchg, 0);
        if (isFp32 || isFp16) {
          ret = b.bitcast(ret, valueElemTy);
        }
        resultVals[i] = ret;
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, StringRef("agent"));

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded = b.extract_val(atomicValTy, cmpxchg, 0);
          if (isFp32 || isFp16) {
            newLoaded = b.bitcast(newLoaded, valueElemTy);
          }
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        LLVM::InlineAsmOp::create(
            rewriter, loc, void_ty(ctx), /*operands=*/ValueRange{},
            /*asm_string=*/"sl_wait lmcnt(0)", /*constraints=*/"",
            /*has_side_effects=*/true, /*is_align_stack=*/false,
            LLVM::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            /*operand_attrs=*/ArrayAttr::get(ctx, {}));
        b.barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    // FIXME: threadPred = b.true_val() is buggy
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, b.true_val(), targetInfo,
                                getTypeConverter());
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const ILUVATAR::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    auto emitter = LLVM::ILUVATAR::AtomicRMWEmitter(targetInfo, *binOp,
                                                    *memOrder, *scopeStr);

    Value val = op.getVal();
    Value ptr = op.getPtr();
    Value opResult = op.getResult();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    int numElems = 1;
    auto vec = getVectorSize(ptr, axisAnalysisPass);

    if (tensorTy) {
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      numElems = tensorTy.getNumElements();
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto tid = getThreadId(rewriter, loc);

    bool needLdsStaging = !tensorTy && !opResult.use_empty();
    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset") && needLdsStaging
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      {
        Value valElement;
        if (vec == 1) {
          valElement = valElements[i];
        } else {
          Value vecVal = b.undef(vecTy);
          for (size_t ii = 0; ii < vec; ++ii)
            vecVal = b.insert_element(vecTy, vecVal, valElements[i + ii],
                                      b.i32_val(ii));
          valElement = vecVal;
        }

        // If we have a single tl.atomic_rmw that is lowered into multiple
        // llvm.atomic_rmw, and we set the ordering for each to aql_rel (the
        // default if no sem value is explicitly set in the DSL level
        // tl.atomic_add. The llvm backend will insert extra buffer invalidates
        // and L2 write backs causing a perforance degration. To avoid this we
        // set the ordering to release for the first, acquire for the last, and
        // relaxed for anything in between so that only a single set of
        // buffer_inv and buffer_wbl2 instructions are inserted by the backend
        // for any "cluster" of atomic ops.
        if ((vec > 1 || elemsPerThread > 1) &&
            op.getSem() == MemSemantic::ACQUIRE_RELEASE) {
          if (i == 0) {
            // First
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::release);
          } else if (i == elemsPerThread - vec) {
            // Last
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::acquire);
          } else {
            // Middle
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::monotonic);
          }
        }

        Value retVal = emitter.emitAtomicRMW(
            rewriter, ptrElements[i], valElement, rmwMask, atomicSharedMemBase);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        } else {
          if (!atomicSharedMemBase.has_value()) {
            rewriter.eraseOp(op);
            return success();
          }
          Value atomPtr = *atomicSharedMemBase;
          b.barrier();
          Value ret = b.load(valueElemTy, atomPtr);

          rewriter.replaceOp(op, {ret});
          return success();
        }
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

} // namespace

namespace mlir::triton::ILUVATAR {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                  benefit);
}
} // namespace mlir::triton::ILUVATAR
