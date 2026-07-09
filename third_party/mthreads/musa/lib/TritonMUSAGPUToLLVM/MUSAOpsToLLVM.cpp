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

namespace {

StringRef getTMELoadIntrinsicName(unsigned rank) {
  switch (rank) {
  case 1:
    return "llvm.musa.tme.ld.tile.1d";
  case 2:
    return "llvm.musa.tme.ld.tile.2d";
  case 3:
    return "llvm.musa.tme.ld.tile.3d";
  case 4:
    return "llvm.musa.tme.ld.tile.4d";
  case 5:
    return "llvm.musa.tme.ld.tile.5d";
  default:
    return {};
  }
}

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

Value buildTMEIssuePredicate(Value userPred, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadId = getThreadId(rewriter, loc);
  Value issuerPred = b.icmp_eq(threadId, b.i32_val(0));
  return b.and_(userPred, issuerPred);
}

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
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
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
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
    SmallVector<Value> operands = {adaptor.getBarId()};
    emitPredicatedVoidIntrinsic(rewriter, loc, launchPred,
                                "llvm.musa.async.arrive.none.phaseid",
                                operands);
    rewriter.eraseOp(op);
    return success();
  }
};

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
  using ConvertOpToLLVMPattern<
      triton::musa::AsyncTMECopyGlobalToLocalOp>::ConvertOpToLLVMPattern;

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
        getTMELoadIntrinsicName(static_cast<unsigned>(coordRank));
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
    Value descAddr = normalizeTMEDescriptorAddr(
        adaptor.getDesc(), op.getDesc().getType(), loc, rewriter);

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

    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.barrier0",
                                    TypeRange{}, {});

    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
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
    for (const auto &segment : loadSegments) {
      SmallVector<Value> operands = {
          adaptor.getBarId(), segment.dstAddr,  descAddr,
          segment.blockDim,   segment.blockPos, swizzleGranularity,
          swizzleStride,      swizzleLine,      prefetchSize,
          innerPersistence,   outerPersistence, cachePolicy};
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, TypeRange{},
                                      operands);
    }
    LLVM::BrOp::create(rewriter, loc, afterCall);
    rewriter.setInsertionPointToStart(afterCall);
    rewriter.eraseOp(op);
    return success();
  }
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
    Value descAddr = normalizeTMEDescriptorAddr(
        adaptor.getDesc(), op.getDesc().getType(), loc, rewriter);

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
    Value launchPred = buildTMEIssuePredicate(adaptor.getPred(), loc, rewriter);
    SmallVector<Value> operands = {
        srcAddr,     descAddr,           blockDim,
        blockPos,    swizzleGranularity, swizzleStride,
        swizzleLine, innerPersistence,   outerPersistence,
        cachePolicy};

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCall =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCall);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, launchPred, trueBlock, afterCall);
    rewriter.setInsertionPointToStart(trueBlock);
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

} // namespace

void mlir::triton::MUSA::populateMUSAOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns
      .add<SquadDotOpConversion, SquadDotWaitOpConversion, WmmaDotOpConversion,
           WmmaDotWaitOpConversion, BarRecordOpConversion,
           InitArrivalOpConversion, BarrierAddTransOpConversion,
           ArriveBarrierOpConversion, ArriveBarrierNoRetOpConversion,
           WaitBarrierOpConversion, AsyncTMECopyGlobalToLocalOpConversion,
           AsyncTMECopyLocalToGlobalOpConversion, TMEStoreCommitOpConversion,
           TMEStoreReadWaitOpConversion>(typeConverter, benefit);
}
