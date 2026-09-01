#include "Dialect/MUSA/IR/Dialect.h"
#include "DotOpToLLVM/DotOpToLLVM.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <optional>

using namespace mlir;

namespace ttg = mlir::triton::gpu;

namespace {

StringRef getTMEStoreIntrinsicName(unsigned rank) {
  switch (rank) {
  case 1:
    return "llvm.musa.tme.st.1d";
  case 2:
    return "llvm.musa.tme.st.2d";
  case 3:
    return "llvm.musa.tme.st.3d";
  case 4:
    return "llvm.musa.tme.st.4d";
  case 5:
    return "llvm.musa.tme.st.5d";
  default:
    return {};
  }
}

Value normalizeTMEDescriptorAddr(Value value, Type srcType, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  if (value.getType().isInteger(64))
    return value;
  if (srcType.isInteger(64))
    return value;
  if (isa<triton::TensorDescType>(srcType)) {
    return LLVM::PtrToIntOp::create(rewriter, loc, rewriter.getI64Type(),
                                    value);
  }
  if (isa<LLVM::LLVMPointerType>(value.getType())) {
    return LLVM::PtrToIntOp::create(rewriter, loc, rewriter.getI64Type(),
                                    value);
  }
  return value;
}

Value normalizeTMESharedPtr(Value value, Type srcType, Type elemType,
                            Location loc, ConversionPatternRewriter &rewriter,
                            const LLVMTypeConverter *typeConverter) {
  if (auto memDescTy = dyn_cast<triton::gpu::MemDescType>(srcType)) {
    Type llvmElemTy = typeConverter->convertType(elemType);
    auto memObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, value, llvmElemTy, rewriter);
    return memObj.getShmemAffineBase(loc, rewriter, memDescTy);
  }
  return value;
}

template <typename AttrT>
Value materializeTMEEnumAttr(Location loc, AttrT attr,
                             ConversionPatternRewriter &rewriter) {
  return arith::ConstantIntOp::create(
      rewriter, loc, static_cast<int32_t>(attr.getValue()), 32);
}

Value reverseTMEVector(Value value, unsigned rank, Location loc,
                       ConversionPatternRewriter &rewriter) {
  if (rank <= 1)
    return value;

  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy || vecTy.getNumElements() != rank ||
      !vecTy.getElementType().isInteger(32))
    return value;

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> elems;
  elems.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    elems.push_back(b.extract_element(value, b.i32_val(i)));

  Value reversed = b.undef(vecTy);
  for (unsigned i = 0; i < rank; ++i)
    reversed = b.insert_element(reversed, elems[rank - i - 1], b.i32_val(i));
  return reversed;
}

Value materializeTMECoord(Location loc, ValueRange coord,
                          ConversionPatternRewriter &rewriter) {
  if (coord.empty())
    return {};
  if (coord.size() == 1)
    return triton::musa::materializeI32Value(coord.front(), loc, rewriter);

  SmallVector<Value> elems;
  elems.reserve(coord.size());
  for (Value value : coord) {
    Value i32Value = triton::musa::materializeI32Value(value, loc, rewriter);
    if (!i32Value)
      return {};
    elems.push_back(i32Value);
  }

  auto vecTy = VectorType::get({static_cast<int64_t>(coord.size())},
                               rewriter.getI32Type());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value coordVector = b.undef(vecTy);
  for (unsigned i = 0; i < elems.size(); ++i)
    coordVector = b.insert_element(vecTy, coordVector, elems[i], b.i32_val(i));
  return reverseTMEVector(coordVector, coord.size(), loc, rewriter);
}

template <typename IntT>
Value materializeTMEBlockShape(Location loc, ArrayRef<IntT> blockShape,
                               ConversionPatternRewriter &rewriter) {
  if (blockShape.empty())
    return {};
  if (blockShape.size() == 1)
    return arith::ConstantIntOp::create(
        rewriter, loc, static_cast<int32_t>(blockShape.front()), 32);

  SmallVector<Value> elems;
  elems.reserve(blockShape.size());
  for (IntT dim : blockShape)
    elems.push_back(arith::ConstantIntOp::create(
        rewriter, loc, static_cast<int32_t>(dim), 32));

  auto vecTy = VectorType::get({static_cast<int64_t>(blockShape.size())},
                               rewriter.getI32Type());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value blockShapeVector = b.undef(vecTy);
  for (unsigned i = 0; i < elems.size(); ++i)
    blockShapeVector =
        b.insert_element(vecTy, blockShapeVector, elems[i], b.i32_val(i));
  return reverseTMEVector(blockShapeVector, blockShape.size(), loc, rewriter);
}

std::optional<int64_t> getI32Constant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
    auto intAttr = dyn_cast<IntegerAttr>(splatAttr.getSplatValue<Attribute>());
    if (intAttr)
      return intAttr.getInt();
  }

  return std::nullopt;
}

std::optional<int64_t> getPositiveIntAttrFromParents(Operation *op,
                                                     StringRef name) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (auto attr = cur->getAttrOfType<IntegerAttr>(name)) {
      if (attr.getInt() > 0)
        return attr.getInt();
    }
  }
  return std::nullopt;
}

std::optional<bool> inferRowMajorFromMemDesc(Type type) {
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(type);
  if (!memDescTy)
    return std::nullopt;
  auto order = triton::gpu::getOrder(memDescTy);
  if (order.empty())
    return std::nullopt;
  return static_cast<int64_t>(order.front() + 1) ==
         static_cast<int64_t>(memDescTy.getShape().size());
}

std::optional<int64_t> inferElemBytesFromMemDesc(Type type) {
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(type);
  if (!memDescTy)
    return std::nullopt;
  int bitWidth = memDescTy.getElementTypeBitWidth();
  if (bitWidth <= 0)
    return std::nullopt;
  return static_cast<int64_t>((bitWidth + 7) / 8);
}

#ifdef __TLE__
Value buildTMEIssuePredicate(Value userPred, Location loc,
                             ConversionPatternRewriter &rewriter,
                             Operation *issueOp = nullptr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  int32_t issueThread = 0;
  bool hasExplicitIssueThread = false;
  if (issueOp) {
    if (auto attr = issueOp->getAttrOfType<IntegerAttr>(
            triton::musa::kTMEIssueThreadAttr)) {
      issueThread = static_cast<int32_t>(attr.getInt());
      hasExplicitIssueThread = true;
    }
  }
  Value threadId;
  if (hasExplicitIssueThread) {
    threadId = ::mlir::gpu::ThreadIdOp::create(rewriter, loc,
                                               ::mlir::gpu::Dimension::x);
    threadId = arith::IndexCastOp::create(rewriter, loc, i32_ty, threadId);
  } else {
    threadId = getThreadId(rewriter, loc);
  }
  Value issuerPred = b.icmp_eq(threadId, b.i32_val(issueThread));
  return b.and_(userPred, issuerPred);
}
#else
Value buildTMEIssuePredicate(Value userPred, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadId = getThreadId(rewriter, loc);
  Value issuerPred = b.icmp_eq(threadId, b.i32_val(0));
  return b.and_(userPred, issuerPred);
}
#endif // __TLE__

Value buildTMEIssueOnlyPredicate(Location loc,
                                 ConversionPatternRewriter &rewriter) {
  Value truePred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  return buildTMEIssuePredicate(truePred, loc, rewriter);
}

void emitPredicatedVoidIntrinsic(ConversionPatternRewriter &rewriter,
                                 Location loc, Value pred, StringRef intrinsic,
                                 ArrayRef<Value> operands) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *afterCall =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *trueBlock = rewriter.createBlock(afterCall);
  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, trueBlock, afterCall);
  rewriter.setInsertionPointToStart(trueBlock);
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, TypeRange{},
                                  operands);
  LLVM::BrOp::create(rewriter, loc, afterCall);
  rewriter.setInsertionPointToStart(afterCall);
}

struct TMELoadSegment {
  Value dstAddr;
  Value blockDim;
  Value blockPos;
};

SmallVector<TMELoadSegment>
buildTMELoadSegments(triton::musa::AsyncTMECopyGlobalToLocalOp op,
                     std::optional<triton::musa::RecoveredSqmmaConsumerContract>
                         recoveredContract,
                     Value dstAddr, Value blockDim, Value blockPos,
                     Location loc, ConversionPatternRewriter &rewriter,
                     const LLVMTypeConverter *typeConverter) {
  SmallVector<TMELoadSegment> segments;
  auto appendSegment = [&](Value segDstAddr, Value segBlockDim,
                           Value segBlockPos) {
    segments.push_back(TMELoadSegment{segDstAddr, segBlockDim, segBlockPos});
  };

  if (!recoveredContract) {
    appendSegment(dstAddr, blockDim, blockPos);
    return segments;
  }

  auto contract = *recoveredContract;
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(op.getResult().getType());
  if (!memDescTy)
    return segments;
  if (memDescTy.getShape().size() != 2)
    return segments;

  auto shape = memDescTy.getShape();
  auto order = triton::musa::getSharedOrder(memDescTy.getEncoding(),
                                            memDescTy.getShape());
  if (order.empty())
    return segments;

  auto maybeElemBytes = triton::musa::inferElemBytesFromMemDesc(memDescTy);
  if (!maybeElemBytes || *maybeElemBytes <= 0 ||
      *maybeElemBytes != contract.elemBytes)
    return segments;

  int64_t majorDimIdx = static_cast<int64_t>(order.front());
  int64_t minorDimIdx = majorDimIdx == 0 ? 1 : 0;
  int64_t leading = shape[majorDimIdx];
  int64_t leadingWidthBytes = leading * *maybeElemBytes;
  if (leadingWidthBytes <= 256) {
    appendSegment(dstAddr, blockDim, blockPos);
    return segments;
  }

  auto vecTy = dyn_cast<VectorType>(blockDim.getType());
  if (!vecTy || vecTy.getNumElements() < 2 ||
      !vecTy.getElementType().isInteger(32))
    return segments;

  int64_t vectorRank = vecTy.getNumElements();
  int64_t majorVectorIdx = vectorRank - majorDimIdx - 1;
  if (majorVectorIdx < 0 || majorVectorIdx >= vectorRank)
    return segments;
  int64_t maxLeadingElems = 256 / *maybeElemBytes;
  if (maxLeadingElems <= 0) {
    appendSegment(dstAddr, blockDim, blockPos);
    return segments;
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value majorVectorIdxVal = b.i32_val(static_cast<int32_t>(majorVectorIdx));
  Value majorBlockPos = b.extract_element(blockPos, majorVectorIdxVal);
  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int64_t leadingOffset = 0;
  while (leadingOffset < leading) {
    int64_t groupLeading =
        std::min<int64_t>(leading - leadingOffset, maxLeadingElems);
    Value groupBlockDim =
        b.insert_element(blockDim, b.i32_val(groupLeading), majorVectorIdxVal);
    Value groupMajorBlockPos = b.add(majorBlockPos, b.i32_val(leadingOffset));
    Value groupBlockPos =
        b.insert_element(blockPos, groupMajorBlockPos, majorVectorIdxVal);

    Value groupDstAddr = dstAddr;
    if (leadingOffset != 0) {
      int64_t tileElemOffset = shape[minorDimIdx] * leadingOffset;
      groupDstAddr =
          b.gep(elemPtrTy, llvmElemTy, dstAddr, b.i32_val(tileElemOffset));
    }

    appendSegment(groupDstAddr, groupBlockDim, groupBlockPos);
    leadingOffset += groupLeading;
  }

  return segments;
}

struct SquadDotOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::SquadDotOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::SquadDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::SquadDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value threadId = getThreadId(rewriter, loc);
    if (failed(mlir::triton::MUSA::convertSQMMADot(
            op, adaptor, this->getTypeConverter(), rewriter, threadId)))
      return op.emitError("MUSA SQMMA: ttmg direct lowering failed");
    return success();
  }
};

struct SquadDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::SquadDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::SquadDotWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::SquadDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                    "llvm.musa.sqmma.wait", TypeRange{}, {});
    rewriter.replaceOp(op, adaptor.getInputs());
    return success();
  }
};

struct WmmaDotOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::WmmaDotOp> {
  using ConvertOpToLLVMPattern<triton::musa::WmmaDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::WmmaDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(mlir::triton::MUSA::convertWMMADot(
            op, adaptor, this->getTypeConverter(), rewriter)))
      return op.emitError("MUSA WMMA: ttmg direct lowering failed");
    return success();
  }
};

struct WmmaDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::WmmaDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::WmmaDotWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::WmmaDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarRecordOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::BarRecordOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::BarRecordOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::BarRecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = {adaptor.getBarId()};
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                    "llvm.musa.async.bar.record", TypeRange{},
                                    operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InitArrivalOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::InitArrivalOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::InitArrivalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::InitArrivalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value arriveCount = adaptor.getArriveCount();
    if (auto count = getI32Constant(arriveCount); count && *count <= 0) {
      auto numWarps = getPositiveIntAttrFromParents(
          op.getOperation(), triton::gpu::AttrNumWarpsName);
      if (numWarps)
        arriveCount = TritonLLVMOpBuilder(loc, rewriter)
                          .i32_val(static_cast<int32_t>(*numWarps));
    }

    Value launchPred = buildTMEIssueOnlyPredicate(loc, rewriter);
    SmallVector<Value> operands = {adaptor.getBarId(), arriveCount,
                                   adaptor.getPhaseId()};
    emitPredicatedVoidIntrinsic(rewriter, loc, launchPred,
                                "llvm.musa.async.init.arrival", operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierAddTransOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::BarrierAddTransOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::BarrierAddTransOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::BarrierAddTransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
#ifdef __TLE__
    Value launchPred =
        buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter, op);
#else
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
#endif // __TLE__
    SmallVector<Value> operands = {adaptor.getBarId(), adaptor.getTransBytes()};
    emitPredicatedVoidIntrinsic(rewriter, loc, launchPred,
                                "llvm.musa.async.add.trans", operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::ArriveBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = {adaptor.getBarId()};
    auto call = LLVM::createLLVMIntrinsicCallOp(
        rewriter, op.getLoc(), "llvm.musa.async.arrive",
        TypeRange{op.getResult().getType()}, operands);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct ArriveBarrierNoRetOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::ArriveBarrierNoRetOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::ArriveBarrierNoRetOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::ArriveBarrierNoRetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
#ifdef __TLE__
    Value launchPred =
        buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter, op);
#else
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
#endif // __TLE__
    SmallVector<Value> operands = {adaptor.getBarId()};
    emitPredicatedVoidIntrinsic(rewriter, loc, launchPred,
                                "llvm.musa.async.arrive.none.phaseid",
                                operands);
    rewriter.eraseOp(op);
    return success();
  }
};

#ifdef __TLE__
struct WarpArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::WarpArriveBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::WarpArriveBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::WarpArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = {adaptor.getBarId()};
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                    "llvm.musa.async.arrive.none.phaseid",
                                    TypeRange{}, operands);
    rewriter.eraseOp(op);
    return success();
  }
};
#endif // __TLE__

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::WaitBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::WaitBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = {adaptor.getBarId(), adaptor.getPhaseId()};
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, op.getLoc(), "llvm.musa.async.wait", TypeRange{}, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTMECopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::AsyncTMECopyGlobalToLocalOp> {
  AsyncTMECopyGlobalToLocalOpConversion(
      LLVMTypeConverter &converter, PatternBenefit benefit,
      const triton::MUSA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::musa::AsyncTMECopyGlobalToLocalOp>(
            converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::musa::AsyncTMECopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto coordRank = adaptor.getCoord().size();
    auto blockShape = op.getBlockShape();
    if (coordRank == 0 || coordRank > 5 || blockShape.size() != coordRank) {
      return op.emitError(
          "MUSA async_tme_copy_global_to_local expects coord/blockShape rank "
          "in [1, 5] to match");
    }
    StringRef intrinsic =
        targetInfo.getTMELoadIntrinsicName(static_cast<unsigned>(coordRank));
    if (intrinsic.empty())
      return op.emitError(
          "MUSA async_tme_copy_global_to_local unsupported rank");

    Value blockDim = materializeTMEBlockShape(loc, blockShape, rewriter);
    Value blockPos = materializeTMECoord(loc, adaptor.getCoord(), rewriter);
    if (!blockDim || !blockPos)
      return op.emitError("unable to materialize TME coord/blockShape");

    Value dstAddr = adaptor.getResult();
    if (auto memDescTy =
            dyn_cast<triton::gpu::MemDescType>(op.getResult().getType())) {
      dstAddr = normalizeTMESharedPtr(adaptor.getResult(), memDescTy,
                                      memDescTy.getElementType(), loc, rewriter,
                                      this->getTypeConverter());
    }

    auto sgAttr = op->getAttrOfType<triton::musa::TMESwizzleGranularityAttr>(
        "swizzleGranularity");
    auto ssAttr =
        op->getAttrOfType<triton::musa::TMESwizzleStrideAttr>("swizzleStride");
    auto slAttr =
        op->getAttrOfType<triton::musa::TMESwizzleLineAttr>("swizzleLine");
    auto prefetchAttr =
        op->getAttrOfType<triton::musa::TMEPrefetchSizeAttr>("prefetchSize");
    auto cacheAttr =
        op->getAttrOfType<triton::musa::TMEL2CachePolicyAttr>("cachePolicy");
    auto innerAttr =
        op->getAttrOfType<triton::musa::TMEPersistenceAttr>("innerPersistence");
    auto outerAttr =
        op->getAttrOfType<triton::musa::TMEPersistenceAttr>("outerPersistence");
    if (!sgAttr || !ssAttr || !slAttr || !prefetchAttr || !cacheAttr ||
        !innerAttr || !outerAttr)
      return op.emitError("missing typed TME policy attrs");

    Value swizzleGranularity = materializeTMEEnumAttr(loc, sgAttr, rewriter);
    Value swizzleStride = materializeTMEEnumAttr(loc, ssAttr, rewriter);
    Value swizzleLine = materializeTMEEnumAttr(loc, slAttr, rewriter);
    Value prefetchSize = materializeTMEEnumAttr(loc, prefetchAttr, rewriter);
    Value innerPersistence = materializeTMEEnumAttr(loc, innerAttr, rewriter);
    Value outerPersistence = materializeTMEEnumAttr(loc, outerAttr, rewriter);
    Value cachePolicy = materializeTMEEnumAttr(loc, cacheAttr, rewriter);

#ifdef __TLE__
    if (!op->hasAttr(triton::musa::kTMEExplicitCompletionAttr))
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.barrier0",
                                      TypeRange{}, {});
#else
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.barrier0",
                                    TypeRange{}, {});
#endif // __TLE__

#ifdef __TLE__
    Value launchPred =
        buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter, op);
#else
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
#endif // __TLE__
    auto recoveredContract =
        triton::musa::recoverAndVerifyGroupedTMELoadConsumerContract(op);
    if (failed(recoveredContract))
      return failure();
    auto loadSegments =
        buildTMELoadSegments(op, *recoveredContract, dstAddr, blockDim,
                             blockPos, loc, rewriter, this->getTypeConverter());
    if (loadSegments.empty()) {
      return op.emitError("unable to materialize grouped TME load segments");
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCall =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCall);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, launchPred, trueBlock, afterCall);
    rewriter.setInsertionPointToStart(trueBlock);
    Value descAddr = normalizeTMEDescriptorAddr(
        adaptor.getDesc(), op.getDesc().getType(), loc, rewriter);
    for (const auto &segment : loadSegments) {
      SmallVector<Value> operands = {
          adaptor.getBarId(), segment.dstAddr,  descAddr,
          segment.blockDim,   segment.blockPos, swizzleGranularity,
          swizzleStride,      swizzleLine,      prefetchSize};
      targetInfo.appendTMELoadPolicyOperands(rewriter, loc, cachePolicy,
                                             innerPersistence, outerPersistence,
                                             operands);
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, TypeRange{},
                                      operands);
    }
    LLVM::BrOp::create(rewriter, loc, afterCall);
    rewriter.setInsertionPointToStart(afterCall);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const triton::MUSA::TargetInfo &targetInfo;
};

struct AsyncTMECopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::AsyncTMECopyLocalToGlobalOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::AsyncTMECopyLocalToGlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::AsyncTMECopyLocalToGlobalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto coordRank = adaptor.getCoord().size();
    auto blockShape = op.getBlockShape();
    if (coordRank == 0 || coordRank > 5 || blockShape.size() != coordRank) {
      return op.emitError(
          "MUSA async_tme_copy_local_to_global expects coord/blockShape rank "
          "in [1, 5] to match");
    }
    StringRef intrinsic =
        getTMEStoreIntrinsicName(static_cast<unsigned>(coordRank));
    if (intrinsic.empty())
      return op.emitError(
          "MUSA async_tme_copy_local_to_global unsupported rank");

    Value blockDim = materializeTMEBlockShape(loc, blockShape, rewriter);
    Value blockPos = materializeTMECoord(loc, adaptor.getCoord(), rewriter);
    if (!blockDim || !blockPos)
      return op.emitError("unable to materialize TME coord/blockShape");

    Value srcAddr = adaptor.getSrc();
    if (auto memDescTy =
            dyn_cast<triton::gpu::MemDescType>(op.getSrc().getType())) {
      srcAddr = normalizeTMESharedPtr(adaptor.getSrc(), memDescTy,
                                      memDescTy.getElementType(), loc, rewriter,
                                      this->getTypeConverter());
    }

    auto sgAttr = op->getAttrOfType<triton::musa::TMESwizzleGranularityAttr>(
        "swizzleGranularity");
    auto ssAttr =
        op->getAttrOfType<triton::musa::TMESwizzleStrideAttr>("swizzleStride");
    auto slAttr =
        op->getAttrOfType<triton::musa::TMESwizzleLineAttr>("swizzleLine");
    auto cacheAttr =
        op->getAttrOfType<triton::musa::TMEL2CachePolicyAttr>("cachePolicy");
    auto innerAttr =
        op->getAttrOfType<triton::musa::TMEPersistenceAttr>("innerPersistence");
    auto outerAttr =
        op->getAttrOfType<triton::musa::TMEPersistenceAttr>("outerPersistence");
    if (!sgAttr || !ssAttr || !slAttr || !cacheAttr || !innerAttr || !outerAttr)
      return op.emitError("missing typed TME policy attrs");

    Value swizzleGranularity = materializeTMEEnumAttr(loc, sgAttr, rewriter);
    Value swizzleStride = materializeTMEEnumAttr(loc, ssAttr, rewriter);
    Value swizzleLine = materializeTMEEnumAttr(loc, slAttr, rewriter);
    Value innerPersistence = materializeTMEEnumAttr(loc, innerAttr, rewriter);
    Value outerPersistence = materializeTMEEnumAttr(loc, outerAttr, rewriter);
    Value cachePolicy = materializeTMEEnumAttr(loc, cacheAttr, rewriter);
#ifdef __TLE__
    Value launchPred =
        buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter, op);
#else
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
#endif // __TLE__

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCall =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCall);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, launchPred, trueBlock, afterCall);
    rewriter.setInsertionPointToStart(trueBlock);
    Value descAddr = normalizeTMEDescriptorAddr(
        adaptor.getDesc(), op.getDesc().getType(), loc, rewriter);
    SmallVector<Value> operands = {
        srcAddr,     descAddr,           blockDim,
        blockPos,    swizzleGranularity, swizzleStride,
        swizzleLine, innerPersistence,   outerPersistence,
        cachePolicy};
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, TypeRange{},
                                    operands);
    LLVM::BrOp::create(rewriter, loc, afterCall);
    rewriter.setInsertionPointToStart(afterCall);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TMEStoreCommitOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::TMEStoreCommitOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::TMEStoreCommitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::TMEStoreCommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, op.getLoc(), "llvm.musa.tme.store.commit", TypeRange{}, {});
    rewriter.eraseOp(op);
    return success();
  }
};

struct TMEStoreReadWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::TMEStoreReadWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::TMEStoreReadWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::TMEStoreReadWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                    "llvm.musa.tme.store.read.wait",
                                    TypeRange{}, {});
    rewriter.eraseOp(op);
    return success();
  }
};

struct TMEEncodeDescriptorOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::TMEEncodeDescriptorOp> {
  using ConvertOpToLLVMPattern<
      triton::musa::TMEEncodeDescriptorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::TMEEncodeDescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Value descBuf = adaptor.getDescBuf();
    Value basePtr = adaptor.getBase();
    auto shapeOps = adaptor.getShape();
    auto strideOps = adaptor.getStrides();
    unsigned elemSize = static_cast<unsigned>(op.getElemSize());
    unsigned rank = shapeOps.size();
    int32_t elemTypeEnum = static_cast<int32_t>(op.getElemType());
    auto constantFill = triton::musa::getMUSATMEConstantFill(
        elemTypeEnum, static_cast<triton::PaddingOption>(op.getPadding()));
    if (!constantFill)
      return op.emitOpError("padding nan is only supported for floating-point "
                            "TME descriptor element types");

    SmallVector<Value> strides(4, b.i64_val(0));
    for (unsigned i = 0; i < rank - 1 && i < 4; i++) {
      Value s = strideOps[rank - 2 - i];
      strides[i] = b.mul(s, b.i64_val(elemSize));
    }

    Value baseAddr = b.ptrtoint(i64_ty, basePtr);

    Value writerPred = buildTMEIssueOnlyPredicate(loc, rewriter);
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock =
        rewriter.splitBlock(entryBlock, rewriter.getInsertionPoint());
    Block *writerBlock = rewriter.createBlock(continuationBlock);
    rewriter.setInsertionPointToEnd(entryBlock);
    LLVM::CondBrOp::create(rewriter, loc, writerPred, writerBlock,
                           continuationBlock);
    rewriter.setInsertionPointToStart(writerBlock);

    auto storeWord = [&](unsigned byteOffset, Value val) {
      Value addr = b.gep(descBuf.getType(), rewriter.getI8Type(), descBuf,
                         b.i32_val(byteOffset));
      LLVM::StoreOp::create(rewriter, loc, val, addr);
    };

    for (unsigned i = 0; i < 5; i++) {
      Value dim = (i < rank) ? shapeOps[rank - 1 - i] : b.i32_val(1);
      storeWord(i * 4, dim);
    }

    {
      Value s0l = b.trunc(i32_ty, strides[0]);
      Value s0h = b.trunc(i32_ty, b.lshr(strides[0], b.i64_val(32)));
      Value s1l = b.trunc(i32_ty, strides[1]);
      Value s1h = b.trunc(i32_ty, b.lshr(strides[1], b.i64_val(32)));
      storeWord(20, b.shl(b.and_(s0l, b.i32_val(0xFFFF)), b.i32_val(16)));
      storeWord(24, b.or_(b.or_(b.lshr(s0l, b.i32_val(16)),
                                b.shl(s0h, b.i32_val(16))),
                          b.shl(b.and_(s1l, b.i32_val(0xFF)), b.i32_val(24))));
      storeWord(28,
                b.or_(b.lshr(s1l, b.i32_val(8)), b.shl(s1h, b.i32_val(24))));
    }

    {
      Value s2l = b.trunc(i32_ty, strides[2]);
      Value s2h = b.trunc(i32_ty, b.lshr(strides[2], b.i64_val(32)));
      Value s3l = b.trunc(i32_ty, strides[3]);
      Value s3h = b.trunc(i32_ty, b.lshr(strides[3], b.i64_val(32)));
      storeWord(32, s2l);
      storeWord(36, b.or_(s2h, b.shl(s3l, b.i32_val(8))));
      storeWord(
          40, b.or_(b.or_(b.lshr(s3l, b.i32_val(24)), b.shl(s3h, b.i32_val(8))),
                    b.i32_val(elemTypeEnum << 24)));
    }

    storeWord(44, b.i32_val(0));

    storeWord(48, b.trunc(i32_ty, baseAddr));
    storeWord(52, b.trunc(i32_ty, b.lshr(baseAddr, b.i64_val(32))));

    uint32_t fillLow = static_cast<uint32_t>(*constantFill & 0xffffffffULL);
    uint32_t fillHigh = static_cast<uint32_t>(*constantFill >> 32);
    storeWord(56, b.i32_val(static_cast<int32_t>(fillLow)));
    storeWord(60, b.i32_val(static_cast<int32_t>(fillHigh)));

    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.membar.gl",
                                    TypeRange{}, {});
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.musa.tme.desc.iv.context", TypeRange{}, {});

    LLVM::BrOp::create(rewriter, loc, continuationBlock);
    rewriter.setInsertionPointToStart(continuationBlock);

    b.barrier(triton::gpu::AddrSpace::None);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReinterpretTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::musa::ReinterpretTensorDescOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::musa::ReinterpretTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultType,
                                                       adaptor.getRawDesc());
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateMUSAOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo) {
  patterns
      .add<SquadDotOpConversion, SquadDotWaitOpConversion, WmmaDotOpConversion,
           WmmaDotWaitOpConversion, BarRecordOpConversion,
           InitArrivalOpConversion, BarrierAddTransOpConversion,
           ArriveBarrierOpConversion, ArriveBarrierNoRetOpConversion,
           WaitBarrierOpConversion, AsyncTMECopyLocalToGlobalOpConversion,
           TMEStoreCommitOpConversion, TMEStoreReadWaitOpConversion,
           TMEEncodeDescriptorOpConversion, ReinterpretTensorDescOpConversion>(
          typeConverter, benefit);
  patterns.add<AsyncTMECopyGlobalToLocalOpConversion>(typeConverter, benefit,
                                                      targetInfo);
#ifdef __TLE__
  patterns.add<WarpArriveBarrierOpConversion>(typeConverter, benefit);
#endif // __TLE__
}
