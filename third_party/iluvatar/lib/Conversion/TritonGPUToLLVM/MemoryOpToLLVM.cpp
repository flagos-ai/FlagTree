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
              const TargetInfoBase &targetInfo, Value inputStride);
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
                           typeConverter, rewriter, targetInfo,
                           loadOp.getInputStride());
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
      auto sharedLayout = toLinearLayout(memDescTy);
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
              const TargetInfoBase &targetInfo, Value inputStride) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto smeEnc = mlir::cast<BlockedEncodingAttr>(regTy.getEncoding());
  auto order = smeEnc.getOrder();
  auto smeWpt = smeEnc.getSmeWarpsPerCTA();
  auto shape = regTy.getShape();

  auto elemTy = typeConverter->convertType(memDescTy.getElementType());
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  bool isRowMajor = order[0] != 0;

  // SME hardware tile: 16 rows × 64B.
  unsigned offset0, offset1; // rows per tile, cols per tile
  if (isRowMajor) {
    offset0 = 16;
    offset1 = 64 / elemBytes;
  } else {
    offset0 = 64 / elemBytes;
    offset1 = 16;
  }
  // CTA-level tile (all warps): smeWpt[0]*offset0 rows × smeWpt[1]*offset1 cols
  SmallVector<unsigned> shapePerCTA({smeWpt[0] * offset0, smeWpt[1] * offset1});

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto smemBase = smemObj.getBase();
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  // Per-warp position within the CTA tile.
  Value warp0 = b.urem(warpId, b.i32_val(smeWpt[0]));
  Value warp1 =
      b.urem(b.udiv(warpId, b.i32_val(smeWpt[0])), b.i32_val(smeWpt[1]));

  // Stride in elements (= inputStride, already element count per row).
  Value stride = inputStride;
  Value strideBytes = b.mul(stride, b.i32_val(static_cast<int32_t>(elemBytes)));

  // ABase = {ptr_lo, ptr_hi, -1, stride_bytes}
  Value gPtr = inVals[0];
  Value gPtrAsInt = b.ptrtoint(i64_ty, gPtr);
  auto i32x4Ty = vec_ty(i32_ty, 4);
  Value abaseVal = b.undef(i32x4Ty);
  abaseVal = b.insert_element(i32x4Ty, abaseVal, b.trunc(i32_ty, gPtrAsInt),
                              b.i32_val(0));
  abaseVal = b.insert_element(i32x4Ty, abaseVal,
                              b.trunc(i32_ty, b.lshr(gPtrAsInt, b.i64_val(32))),
                              b.i32_val(1));
  abaseVal = b.insert_element(i32x4Ty, abaseVal, b.i32_val(-1), b.i32_val(2));
  abaseVal = b.insert_element(i32x4Ty, abaseVal, strideBytes, b.i32_val(3));

  Type elemPtrTy = ptr_ty(ctx, 1);

  for (unsigned m = 0; m < shape[0] / shapePerCTA[0]; m++) {
    for (unsigned k = 0; k < shape[1] / shapePerCTA[1]; k++) {
      Value tileSmemOff, tileGmemOff;
      if (isRowMajor) {
        // row-major: dim1 contiguous
        tileSmemOff =
            b.add(b.mul(b.i32_val(m), b.i32_val(shape[1] * shapePerCTA[0])),
                  b.mul(b.i32_val(k), b.i32_val(offset0 * shapePerCTA[1])));
        tileGmemOff =
            b.add(b.mul(b.mul(b.i32_val(m), b.i32_val(shapePerCTA[0])), stride),
                  b.mul(b.i32_val(k), b.i32_val(shapePerCTA[1])));
        // Per-warp offset within CTA tile
        Value warpSmemOff =
            b.add(b.mul(warp0, b.mul(b.i32_val(offset0), b.i32_val(shape[1]))),
                  b.mul(warp1, b.mul(b.i32_val(offset1), b.i32_val(offset0))));
        Value warpGmemOff =
            b.add(b.mul(b.mul(warp0, b.i32_val(offset0)), stride),
                  b.mul(warp1, b.i32_val(offset1)));
        tileSmemOff = b.add(tileSmemOff, warpSmemOff);
        tileGmemOff = b.add(tileGmemOff, warpGmemOff);
      } else {
        // col-major: dim0 contiguous
        tileSmemOff =
            b.add(b.mul(b.i32_val(m), b.i32_val(shapePerCTA[0] * offset1)),
                  b.mul(b.mul(b.i32_val(k), b.i32_val(shapePerCTA[1])),
                        b.i32_val(shape[0])));
        tileGmemOff = b.add(
            b.mul(b.i32_val(m), b.i32_val(shapePerCTA[0])),
            b.mul(b.mul(b.i32_val(k), b.i32_val(shapePerCTA[1])), stride));
        // Per-warp offset within CTA tile
        Value warpSmemOff =
            b.add(b.mul(warp0, b.mul(b.i32_val(offset0), b.i32_val(offset1))),
                  b.mul(warp1, b.mul(b.i32_val(shape[0]), b.i32_val(offset1))));
        Value warpGmemOff =
            b.add(b.mul(warp0, b.i32_val(offset0)),
                  b.mul(b.mul(warp1, b.i32_val(offset1)), stride));
        tileSmemOff = b.add(tileSmemOff, warpSmemOff);
        tileGmemOff = b.add(tileGmemOff, warpGmemOff);
      }

      Value smemPtr = b.gep(elemPtrTy, elemTy, smemBase, tileSmemOff);
      Value smemAsInt = b.ptrtoint(i32_ty, smemPtr);

      // Goffset: byte offset within the tile from ABase
      Value gmemByteOff =
          b.mul(tileGmemOff, b.i32_val(static_cast<int32_t>(elemBytes)));

      SmallVector<Value> args = {smemAsInt, abaseVal, gmemByteOff,
                                 b.i32_val(0)};
      // SME global->shared intrinsic is selected by element bitwidth. The
      // hardware moves 16 rows x 64 contiguous bytes regardless of dtype; the
      // bN suffix tells it the element width so it can de-interleave correctly.
      // NOTE: row-major fp32 uses the unsuffixed name (the canonical/default
      // variant), which is asymmetric with col-major fp32 (colxfb32). Keep this
      // exactly as provided by the ixcc intrinsic reference.
      unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
      StringRef intrName;
      if (isRowMajor) {
        if (bitwidth == 8)
          intrName = "llvm.bi.sme.load.16x1b64.rowxfb8";
        else if (bitwidth == 16)
          intrName = "llvm.bi.sme.load.16x1b64.rowxfb16";
        else if (bitwidth == 32)
          intrName = "llvm.bi.sme.load.16x1b64";
        else
          llvm_unreachable("SME row intrinsic only supports i8/fp16/bf16/fp32");
      } else {
        if (bitwidth == 8)
          intrName = "llvm.bi.sme.load.16x1b64.colxfb8";
        else if (bitwidth == 16)
          intrName = "llvm.bi.sme.load.16x1b64.colxfb16";
        else if (bitwidth == 32)
          intrName = "llvm.bi.sme.load.16x1b64.colxfb32";
        else
          llvm_unreachable("SME col intrinsic only supports i8/fp16/bf16/fp32");
      }
      TypeRange resultTypes{};
      auto intrOp =
          LLVM::CallIntrinsicOp::create(rewriter, loc, resultTypes, args);
      intrOp.getProperties().setIntrin(StringAttr::get(ctx, intrName));
      intrOp.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      intrOp.getProperties().setOperandSegmentSizes(
          {static_cast<int32_t>(args.size()), 0});
    }
  }
  return success();
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
