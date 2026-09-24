#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#ifdef __ILUVATAR__
#include "llvm/IR/Intrinsics.h"
#endif

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#ifdef __ILUVATAR__
static LogicalResult
lowerSmeStore(Location loc, MLIRContext *ctx, Value regVal,
              MemDescType memDescTy, SharedMemoryObject smemObj,
              ArrayRef<Value> inVals, const LLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter,
              const TargetInfoBase &targetInfo, triton::LoadOp loadOp);
#endif

LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());

#ifdef __ILUVATAR__
  if (auto blockedEnc =
          mlir::dyn_cast<BlockedEncodingAttr>(regTy.getEncoding())) {
    if (blockedEnc.getIsSme()) {
      auto preOp = regVal.getDefiningOp();
      auto loadOp = dyn_cast<triton::LoadOp>(preOp);
      assert(loadOp &&
             "SME requires LoadOp to be the defining op of the store source");
      return lowerSmeStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                           typeConverter, rewriter, targetInfo, loadOp);
    }
  }
#endif

  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto regLayout = toLinearLayout(regTy);
  auto paddedEnc =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  if (paddedEnc) {
    const auto &sharedLL = paddedEnc.getLinearComponent();
    cvt = regLayout.invertAndCompose(sharedLL);
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
  }
  auto kBlock = str_attr("block");
  // NYI. We would need to emit a map.shared::cluster instruction.
  if (!cvt.isTrivialOver({kBlock})) {
    return failure();
  }
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);

  return success();
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  const TargetInfoBase *targetInfo;

  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr = LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo, funcOp,
                                          b.i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      if (failed(lowerLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                 inVals, typeConverter, rewriter,
                                 targetInfo))) {
        return failure();
      }
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto regLayout = toLinearLayout(regTy);
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
#ifdef __ILUVATAR__
      auto sharedLayout = isIluvatarRowXfb8LocalLoad(op)
                              ? iluvatarRowXfb8ToLinearLayout(
                                    memDescTy.getShape(),
                                    mlir::cast<SwizzledSharedEncodingAttr>(
                                        memDescTy.getEncoding()))
                              : toLinearLayout(memDescTy);
#else
      auto sharedLayout = toLinearLayout(memDescTy);
#endif
      cvt = regLayout.invertAndCompose(sharedLayout);
    }
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

#ifdef __ILUVATAR__
// SME global→shared store.  Each SME intrinsic call transfers 16 rows × 64B
// (= tileRows * tileColBytes).  One intrinsic call per warp per CTA-level
// tile repetition.
//
// shapePerCTA = {smeWpt[0] * tileRows, smeWpt[1] * tileCols}  (total CTA tile)
// Iteration grid:  shape[0] / shapePerCTA[0]  ×  shape[1] / shapePerCTA[1]
//
// Per-call:
//   smem offset = cta_tile_offset + warp_offset_within_tile   (element units)
//   gmem offset = cta_tile_offset + warp_offset_within_tile   (element units)
//   ABase       = {ptr_lo, ptr_hi, -1, stride_bytes}
//
// Row-major (order[0] != 0): contiguous dim is the inner (K) dimension.
//   tileRows=16, tileCols=elems_per_64B
// Col-major (order[0] == 0): contiguous dim is the outer (M) dimension.
//   tileRows=elems_per_64B, tileCols=16  ← swapped in hardware
static LogicalResult
lowerSmeStore(Location loc, MLIRContext *ctx, Value regVal,
              MemDescType memDescTy, SharedMemoryObject smemObj,
              ArrayRef<Value> inVals, const LLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter,
              const TargetInfoBase &targetInfo, triton::LoadOp loadOp) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto elemTy = typeConverter->convertType(memDescTy.getElementType());
  Value mask = loadOp.getMask();

  std::optional<bool> constantMask = getIluvatarSmeConstantMask(mask);
  std::optional<unsigned> validContiguousElems =
      getIluvatarSmeConstantMaskExtent(mask);
  auto smeEnc = cast<BlockedEncodingAttr>(regTy.getEncoding());
  unsigned contiguousDim = smeEnc.getOrder()[0];
  if (validContiguousElems) {
    if (*validContiguousElems == 0)
      constantMask = false;
    else if (*validContiguousElems >= regTy.getShape()[contiguousDim])
      constantMask = true;
  }

  // A compile-time false tile does not need a global transaction at all:
  // directly initialize its shared representation with `other`.
  bool emittedSmeLoad = !constantMask || *constantMask;
  if (emittedSmeLoad) {
    if (failed(emitIluvatarSmeTileLoads(
            loc, ctx, regTy, elemTy, smemObj.getBase(), inVals[0],
            loadOp.getInputStride(), rewriter, validContiguousElems)))
      return failure();
  }

  // A CTA barrier only synchronizes threads; it does not drain the hardware
  // G2S queue. Synchronous local_alloc/local_load users must wait before the
  // shared-memory value can be consumed or reused by a later K iteration.
  if (emittedSmeLoad) {
    auto i64Ty = rewriter.getIntegerType(64);
    Value waitCnt = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             IntegerAttr::get(i64Ty, 8));
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.bi.sl.waitcnt", {},
                                    {waitCnt});
  }
  if (!mask || (constantMask && *constantMask))
    return success();

  Value llvmMask = rewriter.getRemappedValue(mask);
  if (!llvmMask)
    return failure();
  Value llvmOther;
  if (loadOp.getOther())
    llvmOther = rewriter.getRemappedValue(loadOp.getOther());
  return lowerIluvatarSmeMaskFixup(loc, regTy, elemTy, memDescTy, smemObj,
                                   llvmMask, llvmOther, rewriter, targetInfo);
}
#endif

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (failed(lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class LocalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
public:
  LocalBarrierOpConversion(const LLVMTypeConverter &converter,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp>(converter,
                                                            benefit) {}
  using OpAdaptor = typename triton::gpu::LocalBarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<mlir::gpu::BarrierOp>(op);

    return success();
  }
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalBarrierOpConversion>(typeConverter, benefit);
}
