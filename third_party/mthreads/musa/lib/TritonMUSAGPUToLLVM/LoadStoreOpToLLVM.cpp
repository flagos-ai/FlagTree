#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include <algorithm>
#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

inline constexpr llvm::StringLiteral kInplaceLoadAttr =
    "musa.inplace_load_candidate";

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

static bool requiresAbsoluteSwizzledAsyncCopy(MemDescType memDescTy) {
  if (memDescTy.getShape().size() != 2)
    return false;
  auto swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  return succeeded(swizzle) && swizzle->swizzleGranularity !=
                                   triton::musa::TMESwizzleGranularity::SG_NONE;
}

struct PH1TMESwizzleValueConfig {
  Value granularityBytes;
  Value strideBytes;
  Value lineBytes;
  bool hasSwizzle = false;
};

static FailureOr<PH1TMESwizzleValueConfig>
resolvePH1TMESwizzleValueConfig(Location loc, MemDescType memDescTy,
                                ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  PH1TMESwizzleValueConfig config;

  auto emitConfig = [&](triton::musa::TMESwizzleGranularity granularity,
                        triton::musa::TMESwizzleStride stride,
                        triton::musa::TMESwizzleLine line) {
    config.hasSwizzle =
        granularity != triton::musa::TMESwizzleGranularity::SG_NONE;
    config.granularityBytes = b.i32_val(static_cast<int32_t>(
        triton::musa::getSwizzleGranularityBytes(granularity)));
    config.strideBytes = b.i32_val(
        static_cast<int32_t>(triton::musa::getSwizzleStrideBytes(stride)));
    config.lineBytes = b.i32_val(
        static_cast<int32_t>(triton::musa::getSwizzleLineBytes(line)));
  };

  auto swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  if (failed(swizzle))
    return failure();

  emitConfig(swizzle->swizzleGranularity, swizzle->swizzleStride,
             swizzle->swizzleLine);
  return config;
}

static Value
applyPH1TMESwizzleToByteAddressValue(TritonLLVMOpBuilder &b, Value addrBytes,
                                     const PH1TMESwizzleValueConfig &config) {
  if (!config.hasSwizzle)
    return addrBytes;

  Value lineOffset = b.urem(addrBytes, config.lineBytes);
  Value lineId = b.udiv(addrBytes, config.lineBytes);
  Value swizzleGroup = b.udiv(config.strideBytes, config.granularityBytes);
  Value swizzleLineId = b.urem(lineId, swizzleGroup);
  Value sectorInLine = b.udiv(lineOffset, config.granularityBytes);
  Value offsetInSector = b.urem(lineOffset, config.granularityBytes);
  Value targetSectorInLine = b.xor_(sectorInLine, swizzleLineId);
  return b.add(b.add(b.mul(lineId, config.lineBytes),
                     b.mul(targetSectorInLine, config.granularityBytes)),
               offsetInSector);
}

static LogicalResult lowerLocalAllocSrcToShared(
    Location loc, MLIRContext *ctx, Value regVal, MemDescType memDescTy,
    SharedMemoryObject smemObj, ArrayRef<Value> inVals,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
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
  if (!cvt.isTrivialOver({kBlock}))
    return failure();
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);
  return success();
}

static std::optional<int>
getResidualDotOperandLocalLoadMaxVecElems(MemDescType memDescTy) {
  unsigned elemBitWidth = memDescTy.getElementTypeBitWidth();
  if (elemBitWidth == 8)
    return 1;
  return std::nullopt;
}

static SmallVector<Value>
getBlockedThreadIds(Value threadId, ArrayRef<unsigned> shapePerCTATile,
                    ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order,
                    ConversionPatternRewriter &rewriter, Location loc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> threadIds(order.size());
  for (unsigned i = 0; i < order.size() - 1; ++i) {
    unsigned dim = order[i];
    Value dimSize = b.i32_val(
        static_cast<int32_t>(shapePerCTATile[dim] / sizePerThread[dim]));
    Value rem = b.urem(threadId, dimSize);
    threadId = b.udiv(threadId, dimSize);
    threadIds[dim] = rem;
  }
  unsigned dim = order.back();
  threadIds[dim] =
      b.urem(threadId, b.i32_val(static_cast<int32_t>(shapePerCTATile[dim] /
                                                      sizePerThread[dim])));
  return threadIds;
}

static SmallVector<unsigned> getShapePerCTATile(BlockedEncodingAttr layout) {
  SmallVector<unsigned> shapePerCTATile;
  for (auto [reg, thread, warp] :
       llvm::zip(layout.getSizePerThread(), layout.getThreadsPerWarp(),
                 layout.getWarpsPerCTA())) {
    shapePerCTATile.push_back(reg * thread * warp);
  }
  return shapePerCTATile;
}

static unsigned getShapePerCTATileForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(layout);
  unsigned mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  unsigned nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  return isM ? mShapePerCTATile : nShapePerCTATile;
}

static unsigned getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto sizePerThread = layout.getSizePerThread();
  unsigned mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  unsigned nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  return isM ? mSizePerThread : nSizePerThread;
}

static FailureOr<SmallVector<unsigned>>
getStaticStrides(ArrayRef<int64_t> shape, ArrayRef<unsigned> order) {
  if (shape.size() != order.size())
    return failure();

  SmallVector<unsigned> strides(shape.size());
  uint64_t stride = 1;
  for (unsigned dim : order) {
    if (dim >= shape.size() || shape[dim] <= 0 ||
        stride > std::numeric_limits<unsigned>::max())
      return failure();
    strides[dim] = static_cast<unsigned>(stride);
    stride *= static_cast<uint64_t>(shape[dim]);
  }
  return strides;
}

static unsigned product(ArrayRef<unsigned> values) {
  return std::accumulate(values.begin(), values.end(), 1u,
                         std::multiplies<unsigned>());
}

static FailureOr<Value> lowerResidualFMAOperandLoad(
    triton::gpu::LocalLoadOp op, Value llSrc, MemDescType memDescTy,
    RankedTensorType regTy, DotOperandEncodingAttr dotEnc,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  if (memDescTy.getShape().size() != 2 || regTy.getShape().size() != 2)
    return failure();

  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto dLayout = dyn_cast<BlockedEncodingAttr>(dotEnc.getParent());
  if (!dLayout)
    return failure();

  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, llSrc, llvmElemTy, rewriter);
  auto sharedOrder = triton::gpu::getOrder(memDescTy);
  if (sharedOrder.size() != 2)
    return failure();
  auto strides = getStaticStrides(
      memDescTy.getAllocShape().take_back(memDescTy.getRank()), sharedOrder);
  if (failed(strides))
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = llvm::to_vector(dLayout.getSizePerThread());
  auto order = llvm::to_vector(dLayout.getOrder());
  auto threadIds = getBlockedThreadIds(threadId, shapePerCTATile, sizePerThread,
                                       order, rewriter, loc);

  auto shape = memDescTy.getShape();
  unsigned opIdx = dotEnc.getOpIdx();
  SmallVector<Value> vals;
  Value smemBase = smemObj.getShmemAffineBase(loc, rewriter, memDescTy);
  Type ptrTy = smemBase.getType();
  auto inRepOrder = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto repOrder = expandMatrixOrderWithBatch(dLayout.getRepOrder());

  if (opIdx == 0) {
    Value strideAM = b.i32_val(static_cast<int32_t>((*strides)[0]));
    Value strideAK = b.i32_val(static_cast<int32_t>((*strides)[1]));
    unsigned kExtent = static_cast<unsigned>(shape[1]);
    unsigned mExtent = static_cast<unsigned>(shape[0]);
    unsigned mShapePerCTATile = getShapePerCTATileForMN(dLayout, true);
    unsigned mSizePerThread = getSizePerThreadForMN(dLayout, true);
    unsigned mContig = mSizePerThread;
    Value threadIdM = b.urem(
        threadIds[0],
        b.i32_val(static_cast<int32_t>(llvm::divideCeil(mExtent, mContig))));
    Value threadBaseM = b.mul(threadIdM, b.i32_val(mContig));

    SmallVector<unsigned> perRepShape = {1, mSizePerThread, kExtent};
    SmallVector<unsigned> repetitions = {
        1, llvm::divideCeil(mExtent, mShapePerCTATile), 1};
    unsigned elemsPerRep = product(perRepShape);
    unsigned totalElems = elemsPerRep * product(repetitions);
    vals.reserve(totalElems);
    for (unsigned idx = 0; idx < totalElems; ++idx) {
      auto inRepIdx =
          mlir::LLVM::delinearize(idx % elemsPerRep, perRepShape, inRepOrder);
      auto repIdx =
          mlir::LLVM::delinearize(idx / elemsPerRep, repetitions, repOrder);
      Value mCoord = b.add(
          b.add(threadBaseM, b.i32_val(static_cast<int32_t>(inRepIdx[1]))),
          b.i32_val(static_cast<int32_t>(repIdx[1] * mShapePerCTATile)));
      Value kCoord = b.i32_val(static_cast<int32_t>(inRepIdx[2]));
      Value offset = b.add(b.mul(mCoord, strideAM), b.mul(kCoord, strideAK));
      Value ptr = b.gep(ptrTy, llvmElemTy, smemBase, offset);
      vals.push_back(targetInfo.loadDShared(rewriter, loc, ptr, std::nullopt,
                                            llvmElemTy, b.true_val(),
                                            op.getOperation()));
    }
  } else if (opIdx == 1) {
    Value strideBK = b.i32_val(static_cast<int32_t>((*strides)[0]));
    Value strideBN = b.i32_val(static_cast<int32_t>((*strides)[1]));
    unsigned kExtent = static_cast<unsigned>(shape[0]);
    unsigned nExtent = static_cast<unsigned>(shape[1]);
    unsigned nShapePerCTATile = getShapePerCTATileForMN(dLayout, false);
    unsigned nSizePerThread = getSizePerThreadForMN(dLayout, false);
    unsigned nContig = nSizePerThread;
    Value threadIdN = b.urem(
        threadIds[1],
        b.i32_val(static_cast<int32_t>(llvm::divideCeil(nExtent, nContig))));
    Value threadBaseN = b.mul(threadIdN, b.i32_val(nContig));

    SmallVector<unsigned> perRepShape = {1, kExtent, nSizePerThread};
    SmallVector<unsigned> repetitions = {
        1, 1, llvm::divideCeil(nExtent, nShapePerCTATile)};
    unsigned elemsPerRep = product(perRepShape);
    unsigned totalElems = elemsPerRep * product(repetitions);
    vals.reserve(totalElems);
    for (unsigned idx = 0; idx < totalElems; ++idx) {
      auto inRepIdx =
          mlir::LLVM::delinearize(idx % elemsPerRep, perRepShape, inRepOrder);
      auto repIdx =
          mlir::LLVM::delinearize(idx / elemsPerRep, repetitions, repOrder);
      Value kCoord = b.i32_val(static_cast<int32_t>(inRepIdx[1]));
      Value nCoord = b.add(
          b.add(threadBaseN, b.i32_val(static_cast<int32_t>(inRepIdx[2]))),
          b.i32_val(static_cast<int32_t>(repIdx[2] * nShapePerCTATile)));
      Value offset = b.add(b.mul(kCoord, strideBK), b.mul(nCoord, strideBN));
      Value ptr = b.gep(ptrTy, llvmElemTy, smemBase, offset);
      vals.push_back(targetInfo.loadDShared(rewriter, loc, ptr, std::nullopt,
                                            llvmElemTy, b.true_val(),
                                            op.getOperation()));
    }
  } else {
    return failure();
  }

  return packLLElements(loc, typeConverter, vals, rewriter, regTy);
}

Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc, Value a,
               Value b) {
  if (a && b) {
    return TritonLLVMOpBuilder(loc, rewriter).and_(a, b);
  }
  return a ? a : b;
}

template <typename Fn>
void emitIfPredicated(RewriterBase &rewriter, Location loc, Value pred,
                      Fn &&emitFn) {
  if (!pred) {
    emitFn();
    return;
  }
  if (matchPattern(pred, m_One())) {
    emitFn();
    return;
  }

  Block *curBlock = rewriter.getInsertionBlock();
  Block *afterBlock =
      rewriter.splitBlock(curBlock, rewriter.getInsertionPoint());
  Block *thenBlock = rewriter.createBlock(afterBlock);

  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, thenBlock, afterBlock);

  rewriter.setInsertionPointToEnd(thenBlock);
  emitFn();
  LLVM::BrOp::create(rewriter, loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
}

Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred;
  int32_t laneMask = freeVarMasks.lookup(kLane);
  if (laneMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(laneId, b.i32_val(laneMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }
  int32_t warpMask = freeVarMasks.lookup(kWarp);
  if (warpMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(warpId, b.i32_val(warpMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }
  int32_t blockMask = freeVarMasks.lookup(kBlock);
  if (blockMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(blockId, b.i32_val(blockMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }

  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  unsigned contiguity = axisInfoAnalysis.getContiguity(ptr);
  unsigned pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  // Keep vectorized memory operations within 128-bit lanes.
  unsigned maxVec = std::max(1u, 128u / pointeeBitWidth);
  return std::min(contiguity, maxVec);
}

StringRef getMusaMemcpyG2SIntrinsic(Type elemTy) {
  switch (getIntOrFloatOrPtrBitWidth(elemTy)) {
  case 8:
  case 16:
  case 32:
  case 64:
    return "llvm.musa.memcpy.g2s";
  default:
    return {};
  }
}

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const mlir::triton::MUSA::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisInfoAnalysis,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));

    Value ptr = op.getPtr();
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    auto ptrElems = ::mlir::unpackLLElements(loc, llPtr, rewriter);
    unsigned numElems = ptrElems.size();

    unsigned vec = getVectorSize(ptr, axisInfoAnalysis);
    vec = std::min(vec, numElems);

    bool forceScalarSharedByteLoad = false;
    if (!ptrElems.empty()) {
      if (auto llPtrTy =
              dyn_cast<LLVM::LLVMPointerType>(ptrElems.front().getType())) {
        constexpr unsigned kSharedAddrSpace = 3;
        forceScalarSharedByteLoad =
            llPtrTy.getAddressSpace() == kSharedAddrSpace &&
            valueElemTy.getIntOrFloatBitWidth() == 8;
      }
    }

    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min(vec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    SmallVector<Value> otherElems;
    if (llOther) {
      otherElems = ::mlir::unpackLLElements(loc, llOther, rewriter);
      assert(otherElems.size() == numElems);
    }

    while (vec > 1 && (numElems % vec != 0)) {
      vec /= 2;
    }
    vec = std::max(1u, vec);
    if (forceScalarSharedByteLoad)
      vec = 1;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    bool useInplaceLoad = op->hasAttr(kInplaceLoadAttr);

    SmallVector<Value> loaded;
    loaded.reserve(numElems);

    if (vec == 1) {
      for (unsigned i = 0; i < numElems; ++i) {
        if (auto canonical = getCanonicalIndex(i, regMask); i != canonical) {
          loaded.push_back(loaded[canonical]);
          continue;
        }

        Value pred = b.true_val();
        if (!maskElems.empty())
          pred = maybeAnd(rewriter, loc, pred, maskElems[i]);
        Value falseVal =
            otherElems.empty()
                ? LLVM::ConstantOp::create(rewriter, loc, valueElemTy,
                                           rewriter.getZeroAttr(valueElemTy))
                : otherElems[i];
        Value val = useInplaceLoad
                        ? LLVM::MUSA::llInplaceLoad(rewriter, loc, ptrElems[i],
                                                    valueElemTy, pred, falseVal)
                        : LLVM::MUSA::llLoad(rewriter, loc, ptrElems[i],
                                             valueElemTy, pred, falseVal);
        loaded.push_back(val);
      }
    } else {
      auto vecTy = LLVM::getVectorType(valueElemTy, vec);
      for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
        unsigned canonicalVecStart = getCanonicalIndex(vecStart, regMask);
        if (vecStart != canonicalVecStart) {
          for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx)
            loaded.push_back(loaded[canonicalVecStart + elemIdx]);
          continue;
        }

        Value pred = b.true_val();
        if (!maskElems.empty())
          pred = maybeAnd(rewriter, loc, pred, maskElems[vecStart]);

        Value falseVal;
        if (otherElems.empty()) {
          auto zeroAttr = rewriter.getZeroAttr(valueElemTy);
          auto dense =
              DenseElementsAttr::get(cast<ShapedType>(vecTy), zeroAttr);
          falseVal = LLVM::ConstantOp::create(rewriter, loc, vecTy, dense);
        } else {
          falseVal = LLVM::UndefOp::create(rewriter, loc, vecTy);
          for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
            falseVal = b.insert_element(vecTy, falseVal,
                                        otherElems[vecStart + elemIdx],
                                        b.i32_val(elemIdx));
          }
        }

        Value vecVal =
            useInplaceLoad
                ? LLVM::MUSA::llInplaceLoad(rewriter, loc, ptrElems[vecStart],
                                            vecTy, pred, falseVal)
                : LLVM::MUSA::llLoad(rewriter, loc, ptrElems[vecStart], vecTy,
                                     pred, falseVal);
        for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
          loaded.push_back(
              b.extract_element(valueElemTy, vecVal, b.i32_val(elemIdx)));
        }
      }
    }

    Value packed = ::mlir::packLLElements(loc, typeConverter, loaded, rewriter,
                                          op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const mlir::triton::MUSA::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getValue();
    Value llMask = adaptor.getMask();

    auto ptrElems = ::mlir::unpackLLElements(loc, llPtr, rewriter);
    auto valElems = ::mlir::unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = ptrElems.size();
    assert(numElems == valElems.size());

    unsigned vec = getVectorSize(ptr, axisInfoAnalysis);
    vec = std::min(vec, numElems);

    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
      vec = std::min(vec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
    }

    while (vec > 1 && (numElems % vec != 0)) {
      vec /= 2;
    }
    vec = std::max(1u, vec);

    auto *ctx = rewriter.getContext();
    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    if (vec == 1) {
      for (unsigned i = 0; i < numElems; ++i) {
        if (!isCanonicalIndex(i, regMask))
          continue;
        Value pred = threadPred ? threadPred : b.true_val();
        if (!maskElems.empty()) {
          pred = maybeAnd(rewriter, loc, pred, maskElems[i]);
        }
        LLVM::MUSA::llStore(rewriter, loc, ptrElems[i], valElems[i], pred);
      }
      rewriter.eraseOp(op);
      return success();
    }

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(op.getValue()));
    auto vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask))
        continue;

      Value pred = threadPred ? threadPred : b.true_val();
      if (!maskElems.empty()) {
        pred = maybeAnd(rewriter, loc, pred, maskElems[vecStart]);
      }

      Value storeVal = LLVM::UndefOp::create(rewriter, loc, vecTy);
      for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
        storeVal = b.insert_element(
            vecTy, storeVal, valElems[vecStart + elemIdx], b.i32_val(elemIdx));
      }
      LLVM::MUSA::llStore(rewriter, loc, ptrElems[vecStart], storeVal, pred);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const mlir::triton::MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  Value convertToAtomicType(Location loc, Value val, Type elemTy,
                            ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    auto intTy = rewriter.getIntegerType(elemTy.getIntOrFloatBitWidth());
    return LLVM::BitcastOp::create(rewriter, loc, intTy, val);
  }

  Value convertFromAtomicType(Location loc, Value val, Type elemTy,
                              ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    return LLVM::BitcastOp::create(rewriter, loc, elemTy, val);
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto ptrElements =
        ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto cmpElements =
        ::mlir::unpackLLElements(loc, adaptor.getCmp(), rewriter);
    auto valElements =
        ::mlir::unpackLLElements(loc, adaptor.getVal(), rewriter);

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    auto atomicOrdering =
        getMemoryOrdering(op.getSem()).value_or(LLVM::AtomicOrdering::acq_rel);
    auto successOrdering = atomicOrdering;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value storePred = threadPred ? threadPred : b.true_val();
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    auto emitPredicated = [&](Value pred, Type retTy,
                              auto emitAtomic) -> Value {
      if (!pred) {
        return emitAtomic();
      }
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument(retTy, loc);

      rewriter.setInsertionPointToEnd(curBlock);
      Value undefVal = LLVM::UndefOp::create(rewriter, loc, retTy);
      LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock,
                             undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      Value atom = emitAtomic();
      LLVM::BrOp::create(rewriter, loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      return endBlock->getArgument(0);
    };

    if (!tensorTy) {
      Value casPtr = ptrElements.front();
      Value casCmp = cmpElements.front();
      Value casVal = valElements.front();
      auto atomicCmp = convertToAtomicType(loc, casCmp, valueElemTy, rewriter);
      auto atomicVal = convertToAtomicType(loc, casVal, valueElemTy, rewriter);
      Type atomicTy = atomicCmp.getType();

      Value retVal = emitPredicated(threadPred, valueElemTy, [&]() {
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, atomicCmp, atomicVal, successOrdering,
            failureOrdering);
        Value old = b.extract_val(atomicTy, cmpxchg, 0);
        return convertFromAtomicType(loc, old, valueElemTy, rewriter);
      });

      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      if (!op->hasAttr("allocation.offset")) {
        return rewriter.notifyMatchFailure(
            op, "missing allocation.offset for scalar atomic result");
      }

      Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
      targetInfo.storeDShared(rewriter, loc, atomPtr, std::nullopt, retVal,
                              storePred);
      b.barrier(triton::gpu::AddrSpace::Local);
      Value ret = b.load(valueElemTy, atomPtr);
      rewriter.replaceOp(op, {ret});
      return success();
    }

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonical = getCanonicalIndex(i, regMask); canonical != i) {
        resultVals[i] = resultVals[canonical];
        continue;
      }
      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      Value casVal = valElements[i];
      auto atomicCmp = convertToAtomicType(loc, casCmp, valueElemTy, rewriter);
      auto atomicVal = convertToAtomicType(loc, casVal, valueElemTy, rewriter);
      Type atomicTy = atomicCmp.getType();

      Value oldVal = emitPredicated(threadPred, valueElemTy, [&]() {
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, atomicCmp, atomicVal, successOrdering,
            failureOrdering);
        Value old = b.extract_val(atomicTy, cmpxchg, 0);
        return convertFromAtomicType(loc, old, valueElemTy, rewriter);
      });
      resultVals[i] = oldVal;
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, storePred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const mlir::triton::MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();
    auto maybeKind = matchAtomicOp(atomicRmwAttr);
    if (!maybeKind) {
      return rewriter.notifyMatchFailure(op, "unsupported atomic op");
    }
    auto atomicOrdering =
        getMemoryOrdering(op.getSem()).value_or(LLVM::AtomicOrdering::acq_rel);

    auto ptrElements =
        ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valElements =
        ::mlir::unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> maskElements;
    if (adaptor.getMask()) {
      maskElements = ::mlir::unpackLLElements(loc, adaptor.getMask(), rewriter);
    }

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value storePred = threadPred ? threadPred : b.true_val();
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    auto emitPredicated = [&](Value pred, Type retTy,
                              auto emitAtomic) -> Value {
      if (!pred) {
        return emitAtomic();
      }
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument(retTy, loc);

      rewriter.setInsertionPointToEnd(curBlock);
      Value undefVal = LLVM::UndefOp::create(rewriter, loc, retTy);
      LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock,
                             undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      Value atom = emitAtomic();
      LLVM::BrOp::create(rewriter, loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      return endBlock->getArgument(0);
    };

    if (!tensorTy) {
      Value rmwPtr = ptrElements.front();
      Value rmwVal = valElements.front();
      Value rmwMask =
          maskElements.empty() ? b.true_val() : maskElements.front();
      rmwMask = maybeAnd(rewriter, loc, rmwMask, threadPred);

      auto emitAtomic = [&]() -> Value {
        if (*maybeKind == LLVM::AtomicBinOp::fadd &&
            (valueElemTy.isF16() || valueElemTy.isF32() ||
             valueElemTy.isF64())) {
          StringRef funcName;
          Type fpType = valueElemTy;
          if (valueElemTy.isF16()) {
            funcName = "__mt_atomicAdd_f16";
          } else if (valueElemTy.isF32()) {
            funcName = "__mt_atomicAdd_f32";
          } else {
            funcName = "__mt_atomicAdd_f64";
          }
          auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
          auto funcType = LLVM::LLVMFunctionType::get(fpType, {ptrTy, fpType});
          LLVM::LLVMFuncOp funcOp =
              appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
          Value addressCast =
              LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, rmwPtr);
          return LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange{addressCast, rmwVal})
              .getResult();
        }
        return LLVM::AtomicRMWOp::create(rewriter, loc, *maybeKind, rmwPtr,
                                         rmwVal, atomicOrdering)
            .getResult();
      };

      Value retVal = emitPredicated(rmwMask, valueElemTy, emitAtomic);

      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      if (!op->hasAttr("allocation.offset")) {
        return rewriter.notifyMatchFailure(
            op, "missing allocation.offset for scalar atomic result");
      }

      auto *ctx = rewriter.getContext();
      Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
      targetInfo.storeDShared(rewriter, loc, atomPtr, std::nullopt, retVal,
                              rmwMask);
      b.barrier(triton::gpu::AddrSpace::Local);
      Value ret = b.load(valueElemTy, atomPtr);
      rewriter.replaceOp(op, {ret});
      return success();
    }

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonical = getCanonicalIndex(i, regMask); canonical != i) {
        resultVals[i] = resultVals[canonical];
        continue;
      }
      Value rmwPtr = ptrElements[i];
      Value rmwVal = valElements[i];
      Value rmwMask = maskElements.empty() ? b.true_val() : maskElements[i];
      rmwMask = maybeAnd(rewriter, loc, rmwMask, threadPred);

      auto emitAtomic = [&]() -> Value {
        if (*maybeKind == LLVM::AtomicBinOp::fadd &&
            (valueElemTy.isF16() || valueElemTy.isF32() ||
             valueElemTy.isF64())) {
          StringRef funcName;
          Type fpType = valueElemTy;
          if (valueElemTy.isF16()) {
            funcName = "__mt_atomicAdd_f16";
          } else if (valueElemTy.isF32()) {
            funcName = "__mt_atomicAdd_f32";
          } else {
            funcName = "__mt_atomicAdd_f64";
          }
          auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
          auto funcType = LLVM::LLVMFunctionType::get(fpType, {ptrTy, fpType});
          LLVM::LLVMFuncOp funcOp =
              appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
          Value addressCast =
              LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, rmwPtr);
          return LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange{addressCast, rmwVal})
              .getResult();
        }
        return LLVM::AtomicRMWOp::create(rewriter, loc, *maybeKind, rmwPtr,
                                         rmwVal, atomicOrdering)
            .getResult();
      };

      Value atom = emitPredicated(rmwMask, valueElemTy, emitAtomic);
      resultVals[i] = atom;
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, storePred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct SqmmaLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  SqmmaLocalAllocOpConversion(LLVMTypeConverter &converter,
                              const mlir::triton::MUSA::TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc() || !op.getSrc())
      return failure();

    auto srcTensorTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    bool isSqmmaAccumulatorSpill =
        srcTensorTy &&
        isa<triton::gpu::MUSASqmmaEncodingAttr>(srcTensorTy.getEncoding());

    bool isSqmma = triton::musa::hasSqmmaOpIdxAttr(op.getOperation());
    if (!isSqmma && !isSqmmaAccumulatorSpill)
      return failure();
    Location loc = op.getLoc();
    auto memDescTy = cast<MemDescType>(op.getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);

    if (isSqmmaAccumulatorSpill) {
      auto *ctx = op.getContext();
      auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      if (failed(lowerLocalAllocSrcToShared(loc, ctx, op.getSrc(), memDescTy,
                                            smemObj, inVals, getTypeConverter(),
                                            rewriter, targetInfo))) {
        return failure();
      }
      auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());

    auto sharedEnc = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
        memDescTy.getEncoding());
    if (!sharedEnc)
      return failure();
    auto order = triton::gpu::getOrder(memDescTy);
    if (order.size() != 2)
      return failure();

    auto opIdx = triton::musa::getSqmmaOpIdx(op.getOperation());
    if (!opIdx)
      return failure();
    unsigned sqmmaOpIdx = static_cast<unsigned>(*opIdx);

    if (sqmmaOpIdx == 0)
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

    unsigned elemBytes =
        std::max<unsigned>(1, memDescTy.getElementTypeBitWidth() / 8);
    auto swizzleConfig =
        resolvePH1TMESwizzleValueConfig(loc, memDescTy, rewriter);
    if (failed(swizzleConfig))
      return failure();

    auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcIndices =
        emitIndices(loc, rewriter, targetInfo, regTy.getEncoding(), regTy,
                    /*withCTAOffset=*/false);
    if (srcIndices.size() != inVals.size())
      return failure();
    auto freeVarMasks = getFreeVariableMasks(regTy);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto shape = regTy.getShape();
    Value elemBytesVal = b.i32_val(static_cast<int32_t>(elemBytes));
    Value smemOffsetBytes = b.i32_val(0);
    if (auto offAttr = op->getAttrOfType<IntegerAttr>("allocation.offset")) {
      smemOffsetBytes = b.i32_val(static_cast<int32_t>(offAttr.getInt()));
    }

    Value smemElemBase = b.bitcast(smemBase, ptr_ty(ctx, 3));
    for (auto [idx, coord] : llvm::enumerate(srcIndices)) {
      if (!isCanonicalIndex(static_cast<unsigned>(idx), regMask))
        continue;
      auto lmsOffsetInElem = triton::musa::linearizePH1TMELinearCoords(
          b, coord, shape, order, elemBytes);
      if (failed(lmsOffsetInElem))
        return failure();
      Value lmsAddrInByte =
          b.add(b.mul(*lmsOffsetInElem, elemBytesVal), smemOffsetBytes);
      Value swizzledAddrInByte = applyPH1TMESwizzleToByteAddressValue(
          b, lmsAddrInByte, *swizzleConfig);
      Value swizzledAddrRelInByte = b.sub(swizzledAddrInByte, smemOffsetBytes);
      Value swizzledElemOffset = b.udiv(swizzledAddrRelInByte, elemBytesVal);

      Value ptr = b.gep(smemElemBase.getType(), llvmElemTy, smemElemBase,
                        swizzledElemOffset);
      targetInfo.storeDShared(rewriter, loc, ptr, std::nullopt, inVals[idx],
                              threadPred ? threadPred : b.true_val());
    }

    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct DotOperandLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
  DotOperandLocalLoadOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Keep SQMMA local_load paths on the generic lowering.
    if (triton::musa::hasSqmmaOpIdxAttr(op.getOperation()))
      return failure();

    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = dyn_cast<MemDescType>(op.getSrc().getType());
    if (!memDescTy)
      return failure();

    auto regTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!regTy)
      return failure();
    auto dotEnc = dyn_cast<DotOperandEncodingAttr>(regTy.getEncoding());
    if (!dotEnc || !isa<BlockedEncodingAttr>(dotEnc.getParent()))
      return failure();

    auto typeConverter = getTypeConverter();
    auto fmaLoad = lowerResidualFMAOperandLoad(op, adaptor.getSrc(), memDescTy,
                                               regTy, dotEnc, typeConverter,
                                               rewriter, targetInfo);
    if (succeeded(fmaLoad)) {
      rewriter.replaceOp(op, *fmaLoad);
      return success();
    }

    std::optional<int> maxVecElems =
        getResidualDotOperandLocalLoadMaxVecElems(memDescTy);
    if (!maxVecElems)
      return failure();

    auto sharedEnc =
        dyn_cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    if (!sharedEnc)
      return failure();

    Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto kBlock = str_attr("block");
    auto regLayout = toLinearLayout(regTy);

    LinearLayout cvt = LinearLayout::empty();
    if (auto paddedEnc =
            dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc)) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
    }
    if (!cvt.isTrivialOver({kBlock}))
      return failure();
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    Value affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
    uint64_t maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescTy);
    SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            memDescTy.getEncoding())) {
      auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
      paddingShifts = getPaddedSharedShifts(paddedEnc, bitwidth,
                                            /*offsetInBytes=*/true);
    }

    SmallVector<Value> outVals = lowerLdStShared(
        loc, ctx, cvt, /*valsArray=*/{}, llvmElemTy, smemObj.getBase(),
        paddingShifts, affineOffset, maskSpanAffineOffset, rewriter, targetInfo,
        maxVecElems, op.getOperation());
    Value result =
        ::mlir::packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp> {
  AsyncCopyGlobalToLocalOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo),
        axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<MemDescType>(op.getResult().getType());
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    StringRef memcpyIntrinsic = getMusaMemcpyG2SIntrinsic(llvmElemTy);
    if (memcpyIntrinsic.empty())
      return op.emitError("async_copy unsupported element bitwidth for "
                          "llvm.musa.memcpy.g2s.* intrinsic");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();

    auto srcElems = ::mlir::unpackLLElements(loc, llSrc, rewriter);
    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);

    auto ptrTy = srcElems.front().getType();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});

    SmallVector<Value> vals;
    vals.reserve(srcElems.size());
    for (size_t i = 0; i < srcElems.size(); ++i) {
      Value packed = LLVM::UndefOp::create(rewriter, loc, structTy);
      packed = b.insert_val(packed, srcElems[i], 0);
      Value maskElem = llMask ? maskElems[i] : b.true_val();
      packed = b.insert_val(packed, maskElem, 1);
      vals.push_back(packed);
    }

    auto srcLayout = toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    vals = removeBroadcastSrc.apply(vals);

    unsigned maxVec = getVectorSize(op.getSrc(), axisInfoAnalysis);
    if (op.getMask())
      maxVec =
          std::min(maxVec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
    maxVec = std::max(maxVec, op.getContiguity());
    maxVec = std::max(1u, maxVec);
    int vecBytes = maxVec * llvmElemTy.getIntOrFloatBitWidth() / 8;
    if (vecBytes < 4) {
      return op.emitError(
                 "async_copy does not support transfers smaller than 4 bytes")
             << "; calculated " << vecBytes << " bytes";
    }

    Value threadPred = b.true_val();

    bool usePred = false;
    if (op.getMask()) {
      Operation *maskOp = op.getMask().getDefiningOp();
      usePred = !isa_and_nonnull<triton::SplatOp>(maskOp);
    }

    if (requiresAbsoluteSwizzledAsyncCopy(dstTy)) {
      auto sharedEnc = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
          dstTy.getEncoding());
      auto order = triton::gpu::getOrder(dstTy);
      if (!sharedEnc || order.size() != 2)
        return op.emitError(
            "PH1 swizzled async_copy expects 2D swizzled shared destination");

      unsigned elemBytes =
          std::max<unsigned>(1, dstTy.getElementTypeBitWidth() / 8);
      auto swizzleConfig =
          resolvePH1TMESwizzleValueConfig(loc, dstTy, rewriter);
      if (failed(swizzleConfig))
        return failure();

      auto srcIndices =
          emitIndices(loc, rewriter, targetInfo, srcTy.getEncoding(), srcTy,
                      /*withCTAOffset=*/false);
      if (srcIndices.size() != srcElems.size())
        return failure();

      int64_t strideRow64 = srcTy.getShape()[order[0]];
      if (strideRow64 <= 0 || strideRow64 > std::numeric_limits<int32_t>::max())
        return failure();

      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, llDst, llvmElemTy, rewriter);
      Value smemObjBase = smemObj.getBase();
      Value smemElemBase = b.bitcast(smemObjBase, ptr_ty(ctx, 3));
      auto rootLocalAlloc = findRootLocalAlloc(op.getResult());
      if (!rootLocalAlloc || !rootLocalAlloc->hasAttr("allocation.offset")) {
        return op.emitError("PH1 swizzled async_copy requires root local_alloc "
                            "with allocation.offset");
      }
      Value smemRawBase = LLVM::getSharedMemoryBase(
          loc, rewriter, targetInfo, rootLocalAlloc.getOperation());
      auto rootOffsetAttr =
          rootLocalAlloc->getAttrOfType<IntegerAttr>("allocation.offset");
      Value rootOffsetBytes =
          b.i32_val(static_cast<int32_t>(rootOffsetAttr.getInt()));
      Value smemOffsetFromRoot = b.sub(b.ptrtoint(i32_ty, smemObjBase),
                                       b.ptrtoint(i32_ty, smemRawBase));
      Value smemOffsetBytes = b.add(smemOffsetFromRoot, rootOffsetBytes);

      Value elemBytesVal = b.i32_val(static_cast<int32_t>(elemBytes));

      unsigned inVec = std::max<unsigned>(1, op.getContiguity());
      unsigned outVec = sharedEnc.getVec();
      unsigned minVec = inVec;
      if (outVec > 1)
        minVec = std::min(outVec, inVec);
      unsigned numElems = getTotalElemsPerThread(srcTy);

      auto shape = srcTy.getShape();

      for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
        auto idx = srcIndices[elemIdx];
        auto lmsOffsetInElem = triton::musa::linearizePH1TMELinearCoords(
            b, idx, shape, order, elemBytes);
        if (failed(lmsOffsetInElem))
          return failure();

        Value lmsAddrInByte =
            b.add(b.mul(*lmsOffsetInElem, elemBytesVal), smemOffsetBytes);
        Value swizzledAddrInByte = applyPH1TMESwizzleToByteAddressValue(
            b, lmsAddrInByte, *swizzleConfig);
        Value swizzledAddrRelInByte =
            b.sub(swizzledAddrInByte, smemOffsetBytes);
        Value swizzledElemOffset = b.udiv(swizzledAddrRelInByte, elemBytesVal);
        Value basePtr = b.gep(smemElemBase.getType(), llvmElemTy, smemElemBase,
                              swizzledElemOffset);

        auto maxBitWidth =
            std::max<unsigned>(128, llvmElemTy.getIntOrFloatBitWidth());
        auto vecBitWidth = llvmElemTy.getIntOrFloatBitWidth() * minVec;
        auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
        auto numWords = vecBitWidth / bitWidth;
        auto numWordElems = bitWidth / llvmElemTy.getIntOrFloatBitWidth();
        auto byteWidth = bitWidth / 8;
        auto resByteWidth = llvmElemTy.getIntOrFloatBitWidth() / 8;

        for (unsigned wordIdx = 0; wordIdx < numWords; ++wordIdx) {
          unsigned wordElemIdx = wordIdx * numWordElems;
          unsigned offset = wordElemIdx * resByteWidth;
          Value packedVal = vals[elemIdx + wordElemIdx];
          Value srcPtr = b.extract_val(ptrTy, packedVal, 0);
          Value maskElem = b.extract_val(i1_ty, packedVal, 1);
          Value dst = b.gep(basePtr.getType(), llvmElemTy, basePtr,
                            b.i32_val(static_cast<int32_t>(offset)));
          Value copyPred = b.true_val();
          if (usePred) {
            Value notMask = b.xor_(maskElem, b.true_val());
            Value zeroPred = notMask;
            Value zeroElem = LLVM::ConstantOp::create(
                rewriter, loc, llvmElemTy, rewriter.getZeroAttr(llvmElemTy));
            auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
            Value dstBase = b.bitcast(dst, elemPtrTy);
            for (unsigned elem = 0; elem < numWordElems; ++elem) {
              Value dstPtr =
                  b.gep(elemPtrTy, llvmElemTy, dstBase, b.i32_val(elem));
              LLVM::MUSA::llStore(rewriter, loc, dstPtr, zeroElem, zeroPred);
            }
            copyPred = b.and_(copyPred, maskElem);
          }

          Value cpSize = b.i32_val(static_cast<int32_t>(byteWidth));
          Value prefetchSize = b.i32_val(0);
          emitIfPredicated(rewriter, loc, copyPred, [&]() {
            auto funcType = LLVM::LLVMFunctionType::get(
                void_ty(ctx),
                {dst.getType(), srcPtr.getType(), cpSize.getType(),
                 prefetchSize.getType()},
                /*isVarArg=*/false);
            auto funcOp = appendOrGetExternFuncOp(rewriter, op, memcpyIntrinsic,
                                                  funcType);
            LLVM::CallOp::create(rewriter, loc, funcOp,
                                 ValueRange{dst, srcPtr, cpSize, prefetchSize});
          });
        }
      }

      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    auto emitAsyncCopy = [&](RewriterBase &rewriter, Location emitLoc,
                             ArrayRef<Value> values, Value shmemAddr,
                             int startIdx,
                             VectorType vecTy) -> SmallVector<Value> {
      auto tb = TritonLLVMOpBuilder(emitLoc, rewriter);
      unsigned elemsPerVec = vecTy.getNumElements();
      unsigned byteWidth = elemsPerVec * llvmElemTy.getIntOrFloatBitWidth() / 8;

      Value packedVal = values[startIdx];
      Value srcPtr = tb.extract_val(ptrTy, packedVal, 0);
      Value maskElem = tb.extract_val(i1_ty, packedVal, 1);
      Value copyPred = threadPred ? threadPred : tb.true_val();

      if (usePred) {
        Value notMask = tb.xor_(maskElem, tb.true_val());
        Value zeroPred = threadPred ? tb.and_(threadPred, notMask) : notMask;
        Value zeroElem = LLVM::ConstantOp::create(
            rewriter, emitLoc, llvmElemTy, rewriter.getZeroAttr(llvmElemTy));
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value dstBase = tb.bitcast(shmemAddr, elemPtrTy);
        for (unsigned elem = 0; elem < elemsPerVec; ++elem) {
          Value dstPtr =
              tb.gep(elemPtrTy, llvmElemTy, dstBase, tb.i32_val(elem));
          LLVM::MUSA::llStore(rewriter, emitLoc, dstPtr, zeroElem, zeroPred);
        }
        copyPred = tb.and_(copyPred, maskElem);
      }

      Value cpSize = tb.i32_val(byteWidth);
      Value prefetchSize = tb.i32_val(0);
      emitIfPredicated(rewriter, emitLoc, copyPred, [&]() {
        auto funcType = LLVM::LLVMFunctionType::get(
            void_ty(ctx),
            {shmemAddr.getType(), srcPtr.getType(), cpSize.getType(),
             prefetchSize.getType()},
            /*isVarArg=*/false);
        auto funcOp =
            appendOrGetExternFuncOp(rewriter, op, memcpyIntrinsic, funcType);
        LLVM::CallOp::create(
            rewriter, emitLoc, funcOp,
            ValueRange{shmemAddr, srcPtr, cpSize, prefetchSize});
      });
      return {};
    };

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, llvmElemTy, rewriter);
    auto smemLayout = toLinearLayout(dstTy);
    auto cvt = srcLayout.invertAndCompose(smemLayout);
    if (!cvt.isTrivialOver({str_attr("block")}))
      return op.emitError(
          "async_copy does not support non-trivial block dimension");
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    Value affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    uint64_t maskSpanAffineOffset = smemObj.getMaskSpanOffsets(dstTy);
    std::optional<int> maybeMaxVecElems;
    SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      maybeMaxVecElems = paddedEnc.getMinInterval();
      auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
      paddingShifts =
          getPaddedSharedShifts(paddedEnc, bitwidth, /*offsetInBytes=*/true);
    }

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(loc, ctx, cvt, vals, llvmElemTy, smemObj.getBase(), paddingShifts,
              affineOffset, maskSpanAffineOffset, laneId, warpId, rewriter,
              targetInfo, maxVec, emitAsyncCopy);

    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto waitFnTy = LLVM::LLVMFunctionType::get(void_ty(ctx), {}, false);
    auto waitFn = appendOrGetExternFuncOp(
        rewriter, op, "llvm.musa.memcpy.g2s.wait", waitFnTy);
    LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{});
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    int /*computeCapability*/, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<SqmmaLocalAllocOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 2));
  patterns.add<DotOperandLocalLoadOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 3));
  patterns.add<LoadOpConversion, StoreOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
}
