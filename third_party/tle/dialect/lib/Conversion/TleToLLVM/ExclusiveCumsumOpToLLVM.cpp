#include "tle/dialect/include/Conversion/TleToLLVM/ExclusiveCumsumOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>

namespace {

using namespace mlir;
namespace tle = mlir::triton::tle;

static Value createZeroConstant(Location loc,
                                ConversionPatternRewriter &rewriter, Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return LLVM::ConstantOp::create(rewriter, loc, ty,
                                    rewriter.getIntegerAttr(intTy, 0));
  }
  if (auto floatTy = dyn_cast<FloatType>(ty)) {
    return LLVM::ConstantOp::create(rewriter, loc, ty,
                                    rewriter.getFloatAttr(floatTy, 0.0));
  }
  return Value();
}

static Value createAdd(Location loc, ConversionPatternRewriter &rewriter,
                       Value lhs, Value rhs, Type elemTy) {
  if (isa<FloatType>(elemTy))
    return LLVM::FAddOp::create(rewriter, loc, lhs, rhs);
  if (isa<IntegerType>(elemTy))
    return LLVM::AddOp::create(rewriter, loc, lhs, rhs);
  return Value();
}

static Value createSub(Location loc, ConversionPatternRewriter &rewriter,
                       Value lhs, Value rhs, Type elemTy) {
  if (isa<FloatType>(elemTy))
    return LLVM::FSubOp::create(rewriter, loc, lhs, rhs);
  if (isa<IntegerType>(elemTy))
    return LLVM::SubOp::create(rewriter, loc, lhs, rhs);
  return Value();
}

static Value castToI32(Location loc, ConversionPatternRewriter &rewriter,
                       Value idx) {
  auto i32Ty = rewriter.getI32Type();
  auto idxTy = dyn_cast<IntegerType>(idx.getType());
  if (!idxTy)
    return Value();
  if (idxTy.getWidth() == 32)
    return idx;
  if (idxTy.getWidth() > 32)
    return LLVM::TruncOp::create(rewriter, loc, i32Ty, idx);
  return LLVM::ZExtOp::create(rewriter, loc, i32Ty, idx);
}

static Value getSharedElemPtr(Location loc, TritonLLVMOpBuilder &b, Value base,
                              Type elemTy, Value idx) {
  return b.gep(base.getType(), elemTy, base, idx);
}

static std::pair<Block *, Block *>
createIfThenBlocks(ConversionPatternRewriter &rewriter, Location loc,
                   Value condition) {
  Block *prevBlock = rewriter.getInsertionBlock();
  Block *ifBlock = rewriter.splitBlock(prevBlock, rewriter.getInsertionPoint());
  Block *thenBlock = rewriter.splitBlock(ifBlock, ifBlock->begin());
  rewriter.setInsertionPointToEnd(ifBlock);
  cf::BranchOp::create(rewriter, loc, thenBlock);
  rewriter.setInsertionPointToEnd(prevBlock);
  cf::CondBranchOp::create(rewriter, loc, condition, ifBlock, thenBlock);
  return {ifBlock, thenBlock};
}

static Value branchSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Value condition, Value trueValue, Value falseValue) {
  Block *prevBlock = rewriter.getInsertionBlock();
  Block *ifBlock = rewriter.splitBlock(prevBlock, rewriter.getInsertionPoint());
  Block *mergeBlock = rewriter.splitBlock(ifBlock, ifBlock->begin());
  mergeBlock->addArgument(trueValue.getType(), loc);
  rewriter.setInsertionPointToEnd(ifBlock);
  cf::BranchOp::create(rewriter, loc, mergeBlock, ValueRange{trueValue});
  rewriter.setInsertionPointToEnd(prevBlock);
  cf::CondBranchOp::create(rewriter, loc, condition, ifBlock, mergeBlock,
                           ValueRange{falseValue});
  rewriter.setInsertionPointToStart(mergeBlock);
  return mergeBlock->getArgument(0);
}

static Value createWarpScanStepI32(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value val, int offset) {
  auto intTy = dyn_cast<IntegerType>(val.getType());
  if (!intTy || intTy.getWidth() != 32)
    return Value();

  mlir::triton::PTXBuilder ptxBuilder;
  auto *out = ptxBuilder.newOperand("=r", /*init=*/false);
  auto *in = ptxBuilder.newOperand(val, "r");
  auto i32Ty = rewriter.getI32Type();
  auto *offsetOpr = ptxBuilder.newOperand(
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(offset)),
      "r");
  auto *clampOpr = ptxBuilder.newOperand(
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(0)),
      "r");
  auto *maskOpr = ptxBuilder.newOperand(
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(-1)),
      "r");
  std::string ptx = "{\n"
                    "  .reg .s32 r0;\n"
                    "  .reg .pred p;\n"
                    "  shfl.sync.up.b32 r0|p, $1, $2, $3, $4;\n"
                    "  @p add.s32 r0, r0, $1;\n"
                    "  mov.s32 $0, r0;\n"
                    "}\n";
  auto &shflScan = *ptxBuilder.create(ptx);
  shflScan({out, in, offsetOpr, clampOpr, maskOpr},
           /*onlyAttachMLIRArgs=*/true);
  return ptxBuilder.launch(rewriter, loc, val.getType(),
                           /*hasSideEffects=*/false);
}

struct ExclusiveCumsumOpConversion
    : public ConvertOpToLLVMPattern<tle::ExclusiveCumsumOp> {
  ExclusiveCumsumOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(tle::ExclusiveCumsumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy || srcTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expects rank-1 tensor source");
    int64_t axisExtent = srcTy.getShape()[0];
    if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
      return rewriter.notifyMatchFailure(
          op, "expects static, positive axis extent");

    Location loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type elemTy = srcTy.getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);
    if (!isa<IntegerType, FloatType>(llvmElemTy)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported element type for cumsum");
    }

    auto mod = op->getParentOfType<ModuleOp>();
    const int threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    const int numWarps = triton::gpu::lookupNumWarps(op);
    const int numThreadsPerCTA = threadsPerWarp * numWarps;

    Value zero = createZeroConstant(loc, rewriter, llvmElemTy);
    if (!zero)
      return rewriter.notifyMatchFailure(op, "failed to materialize zero");

    auto inputVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto inputIndices =
        emitIndices(loc, rewriter, targetInfo, srcTy.getEncoding(), srcTy,
                    /*withCTAOffset=*/false);
    if (inputVals.size() != inputIndices.size())
      return rewriter.notifyMatchFailure(op, "value/index size mismatch");

    Value baseSharedMem =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = getThreadId(rewriter, loc);
    Value trueVal = b.true_val();

    auto getElemPtr = [&](Value logicalIndex) -> Value {
      return getSharedElemPtr(loc, b, baseSharedMem, llvmElemTy, logicalIndex);
    };

    // TRT/CUB-aligned fastpath for topk histogram rounds:
    // - rank-1, one element per thread
    // - no reverse remapping
    // - full CTA participates (axisExtent == threads per CTA)
    // This avoids the extra shared-memory store/load round-trip used by the
    // generic logical-index path.
    if (!op.getReverse() && inputVals.size() == 1 &&
        axisExtent == static_cast<int64_t>(numThreadsPerCTA) &&
        threadsPerWarp == 32 && numWarps > 0 && numWarps <= 32) {
      Value laneId = b.and_(threadId, b.i32_val(threadsPerWarp - 1));
      Value warpId = b.lshr(threadId, b.i32_val(5));
      Value orderedVal = inputVals.front();
      Value scanVal = orderedVal;
      for (int offset = 1; offset < threadsPerWarp; offset <<= 1) {
        if (Value scanStep =
                createWarpScanStepI32(loc, rewriter, scanVal, offset)) {
          scanVal = scanStep;
        } else {
          Value shfl = targetInfo.shuffleUp(rewriter, loc, scanVal, offset);
          Value hasPred = b.icmp_sge(laneId, b.i32_val(offset));
          Value combined = createAdd(loc, rewriter, scanVal, shfl, llvmElemTy);
          if (!combined)
            return rewriter.notifyMatchFailure(op,
                                               "unsupported add in warp scan");
          scanVal = branchSelect(rewriter, loc, hasPred, combined, scanVal);
        }
      }

      Value warpTotal =
          targetInfo.shuffleIdx(rewriter, loc, scanVal, threadsPerWarp - 1);
      Value isWarpTail = b.icmp_eq(laneId, b.i32_val(threadsPerWarp - 1));
      Value warpSlotBase = b.i32_val(static_cast<int32_t>(axisExtent));
      Value warpSlot = b.add(warpSlotBase, warpId);
      Value warpSlotPtr = getElemPtr(warpSlot);
      auto [tailStoreBlock, tailContBlock] =
          createIfThenBlocks(rewriter, loc, isWarpTail);
      rewriter.setInsertionPointToStart(tailStoreBlock);
      targetInfo.storeShared(rewriter, loc, warpSlotPtr, warpTotal, trueVal);
      rewriter.setInsertionPointToStart(tailContBlock);

      targetInfo.barrier(loc, rewriter);

      Value totalSlot = b.add(warpSlotBase, b.i32_val(numWarps));
      Value totalPtrFast = getElemPtr(totalSlot);
      Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
      auto [ifBlock, thenBlock] = createIfThenBlocks(rewriter, loc, isThread0);
      rewriter.setInsertionPointToStart(ifBlock);
      Value running = zero;
      for (int w = 0; w < numWarps; ++w) {
        Value slot = b.add(warpSlotBase, b.i32_val(w));
        Value slotPtr = getElemPtr(slot);
        Value warpSum =
            targetInfo.loadShared(rewriter, loc, slotPtr, llvmElemTy, trueVal);
        targetInfo.storeShared(rewriter, loc, slotPtr, running, trueVal);
        Value next = createAdd(loc, rewriter, running, warpSum, llvmElemTy);
        if (!next) {
          return rewriter.notifyMatchFailure(
              op, "unsupported add in block-prefix scan");
        }
        running = next;
      }
      targetInfo.storeShared(rewriter, loc, totalPtrFast, running, trueVal);
      rewriter.setInsertionPointToStart(thenBlock);
      targetInfo.barrier(loc, rewriter);

      Value blockPrefixPtr = getElemPtr(warpSlot);
      Value blockPrefix = targetInfo.loadShared(rewriter, loc, blockPrefixPtr,
                                                llvmElemTy, trueVal);
      Value inclusiveOrdered =
          createAdd(loc, rewriter, scanVal, blockPrefix, llvmElemTy);
      Value exclusiveOrdered =
          createSub(loc, rewriter, inclusiveOrdered, orderedVal, llvmElemTy);
      if (!exclusiveOrdered)
        return rewriter.notifyMatchFailure(
            op, "unsupported sub in ordered exclusive");

      Value exclusiveRes =
          packLLElements(loc, typeConverter,
                         SmallVector<Value>{exclusiveOrdered}, rewriter, srcTy);
      Value totalRes = targetInfo.loadShared(rewriter, loc, totalPtrFast,
                                             llvmElemTy, trueVal);
      rewriter.replaceOp(op, ValueRange{exclusiveRes, totalRes});
      return success();
    }

    Value nMinus1 = b.i32_val(static_cast<int32_t>(axisExtent - 1));
    auto getLogicalIndex = [&](Value idx) -> Value {
      if (!op.getReverse())
        return idx;
      return b.sub(nMinus1, idx);
    };

    for (auto [val, idxVec] : llvm::zip_equal(inputVals, inputIndices)) {
      if (idxVec.size() != 1)
        return rewriter.notifyMatchFailure(op, "expects rank-1 indices");
      Value idx = castToI32(loc, rewriter, idxVec[0]);
      if (!idx)
        return rewriter.notifyMatchFailure(op, "index must be integer");
      Value logicalIndex = getLogicalIndex(idx);
      Value ptr = getElemPtr(logicalIndex);
      targetInfo.storeShared(rewriter, loc, ptr, val, trueVal);
    }

    // Ensure all logical-index stores are visible before scan reads.
    targetInfo.barrier(loc, rewriter);

    // Fast path (TRT-style): one logical element per thread order, then
    // warp-scan + shared-memory cross-warp prefix scan.
    // This is the dominant configuration for topk histogram threshold search.
    if (inputVals.size() == 1 && axisExtent <= numThreadsPerCTA &&
        threadsPerWarp == 32 && numWarps > 0 && numWarps <= 32) {
      Value axisExtentVal = b.i32_val(static_cast<int32_t>(axisExtent));
      Value laneId = b.and_(threadId, b.i32_val(threadsPerWarp - 1));
      Value warpId = b.lshr(threadId, b.i32_val(5));
      Value activeOrdered = b.icmp_ult(threadId, axisExtentVal);

      Value orderedPtr = getElemPtr(threadId);
      Value orderedVal = targetInfo.loadShared(rewriter, loc, orderedPtr,
                                               llvmElemTy, activeOrdered);
      orderedVal = b.select(activeOrdered, orderedVal, zero);

      Value scanVal = orderedVal;
      for (int offset = 1; offset < threadsPerWarp; offset <<= 1) {
        if (Value scanStep =
                createWarpScanStepI32(loc, rewriter, scanVal, offset)) {
          scanVal = scanStep;
        } else {
          Value shfl = targetInfo.shuffleUp(rewriter, loc, scanVal, offset);
          Value hasPred = b.icmp_sge(laneId, b.i32_val(offset));
          Value combined = createAdd(loc, rewriter, scanVal, shfl, llvmElemTy);
          if (!combined)
            return rewriter.notifyMatchFailure(op,
                                               "unsupported add in warp scan");
          scanVal = branchSelect(rewriter, loc, hasPred, combined, scanVal);
        }
      }

      Value warpTotal =
          targetInfo.shuffleIdx(rewriter, loc, scanVal, threadsPerWarp - 1);
      Value isWarpTail = b.icmp_eq(laneId, b.i32_val(threadsPerWarp - 1));
      Value warpSlotBase = b.i32_val(static_cast<int32_t>(axisExtent));
      Value warpSlot = b.add(warpSlotBase, warpId);
      Value warpSlotPtr = getElemPtr(warpSlot);
      auto [tailStoreBlock, tailContBlock] =
          createIfThenBlocks(rewriter, loc, isWarpTail);
      rewriter.setInsertionPointToStart(tailStoreBlock);
      targetInfo.storeShared(rewriter, loc, warpSlotPtr, warpTotal, trueVal);
      rewriter.setInsertionPointToStart(tailContBlock);

      targetInfo.barrier(loc, rewriter);

      Value totalSlot = b.add(warpSlotBase, b.i32_val(numWarps));
      Value totalPtrFast = getElemPtr(totalSlot);
      Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
      auto [ifBlock, thenBlock] = createIfThenBlocks(rewriter, loc, isThread0);
      rewriter.setInsertionPointToStart(ifBlock);
      Value running = zero;
      for (int w = 0; w < numWarps; ++w) {
        Value slot = b.add(warpSlotBase, b.i32_val(w));
        Value slotPtr = getElemPtr(slot);
        Value warpSum =
            targetInfo.loadShared(rewriter, loc, slotPtr, llvmElemTy, trueVal);
        targetInfo.storeShared(rewriter, loc, slotPtr, running, trueVal);
        Value next = createAdd(loc, rewriter, running, warpSum, llvmElemTy);
        if (!next) {
          return rewriter.notifyMatchFailure(
              op, "unsupported add in block-prefix scan");
        }
        running = next;
      }
      targetInfo.storeShared(rewriter, loc, totalPtrFast, running, trueVal);
      rewriter.setInsertionPointToStart(thenBlock);

      targetInfo.barrier(loc, rewriter);

      Value blockPrefixPtr = getElemPtr(warpSlot);
      Value blockPrefix = targetInfo.loadShared(rewriter, loc, blockPrefixPtr,
                                                llvmElemTy, trueVal);
      Value inclusiveOrdered =
          createAdd(loc, rewriter, scanVal, blockPrefix, llvmElemTy);
      Value exclusiveOrdered =
          createSub(loc, rewriter, inclusiveOrdered, orderedVal, llvmElemTy);
      if (!exclusiveOrdered)
        return rewriter.notifyMatchFailure(
            op, "unsupported sub in ordered exclusive");
      targetInfo.storeShared(rewriter, loc, orderedPtr, exclusiveOrdered,
                             activeOrdered);
      // The gather below reads by logical index (reverse remap may read values
      // produced by different threads/warps). Ensure all ordered exclusive
      // stores are visible before any thread starts loading gathered values.
      targetInfo.barrier(loc, rewriter);

      SmallVector<Value> exclusiveVals;
      exclusiveVals.reserve(inputVals.size());
      for (auto idxVec : inputIndices) {
        Value idx = castToI32(loc, rewriter, idxVec[0]);
        Value logicalIndex = getLogicalIndex(idx);
        Value ptr = getElemPtr(logicalIndex);
        exclusiveVals.push_back(
            targetInfo.loadShared(rewriter, loc, ptr, llvmElemTy, trueVal));
      }
      Value exclusiveRes =
          packLLElements(loc, typeConverter, exclusiveVals, rewriter, srcTy);
      Value totalRes = targetInfo.loadShared(rewriter, loc, totalPtrFast,
                                             llvmElemTy, trueVal);
      rewriter.replaceOp(op, ValueRange{exclusiveRes, totalRes});
      return success();
    }

    // Fallback path: generic serial scan in shared memory by thread-0.
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Type i8PtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), targetInfo.getSharedAddressSpace());
    unsigned elemBytes = static_cast<unsigned>(
        std::max<int>(1, srcTy.getElementTypeBitWidth() / 8));
    int64_t totalByteOffset =
        static_cast<int64_t>(axisExtent) * static_cast<int64_t>(elemBytes);
    if (totalByteOffset > std::numeric_limits<int32_t>::max()) {
      return rewriter.notifyMatchFailure(op,
                                         "shared scratch offset exceeds i32");
    }
    Value baseI8 = b.bitcast(baseSharedMem, i8PtrTy);
    Value totalOffsetBytes = b.i32_val(static_cast<int32_t>(totalByteOffset));
    Value totalPtrI8 = b.gep(i8PtrTy, i8_ty, baseI8, totalOffsetBytes);
    Value totalPtr = b.bitcast(totalPtrI8, baseSharedMem.getType());

    Value running = zero;
    for (int64_t i = 0; i < axisExtent; ++i) {
      Value idx = b.i32_val(static_cast<int32_t>(i));
      Value ptr = getElemPtr(idx);
      Value inVal =
          targetInfo.loadShared(rewriter, loc, ptr, llvmElemTy, isThread0);
      targetInfo.storeShared(rewriter, loc, ptr, running, isThread0);
      Value next = createAdd(loc, rewriter, running, inVal, llvmElemTy);
      if (!next)
        return rewriter.notifyMatchFailure(op,
                                           "unsupported add for element type");
      running = next;
    }
    targetInfo.storeShared(rewriter, loc, totalPtr, running, isThread0);

    targetInfo.barrier(loc, rewriter);

    SmallVector<Value> exclusiveVals;
    exclusiveVals.reserve(inputVals.size());
    for (auto idxVec : inputIndices) {
      Value idx = castToI32(loc, rewriter, idxVec[0]);
      Value logicalIndex = getLogicalIndex(idx);
      Value ptr = getElemPtr(logicalIndex);
      exclusiveVals.push_back(
          targetInfo.loadShared(rewriter, loc, ptr, llvmElemTy, trueVal));
    }
    Value exclusiveRes =
        packLLElements(loc, typeConverter, exclusiveVals, rewriter, srcTy);
    Value totalRes =
        targetInfo.loadShared(rewriter, loc, totalPtr, llvmElemTy, trueVal);

    rewriter.replaceOp(op, ValueRange{exclusiveRes, totalRes});
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void tle::populateExclusiveCumsumOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ExclusiveCumsumOpConversion>(typeConverter, targetInfo, benefit);
}
