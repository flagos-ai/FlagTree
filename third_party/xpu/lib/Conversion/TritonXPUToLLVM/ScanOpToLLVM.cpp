//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

using ::mlir::triton::gpu::getTotalElemsPerThread;

inline SmallVector<Value>
inlineCombineBlock(ConversionPatternRewriter &rewriter, Block &combineBlock,
                   Block *insertionBlock, Block::iterator insertionPoint,
                   ValueRange combineArgs) {
  auto returnOp = combineBlock.getTerminator();
  rewriter.inlineBlockBefore(&combineBlock, insertionBlock, insertionPoint,
                             combineArgs);

  auto results = SmallVector<Value>(returnOp->getOperands());

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
  return results;
}

inline SmallVector<Value> applyCombineOp(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Region &combineOp, ValueRange acc,
                                         ValueRange cur, Value pred = {}) {
  // Allows for passing an uninitialized acc and use cur as the neutral element
  if (acc.size() == 0) {
    return cur;
  }
  assert(cur.size() == acc.size());

  // Create a new copy of the combine block, and try to speculatively inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();

  rewriter.cloneRegionBefore(combineOp, parent,
                             std::next(currentBlock->getIterator()));
  Block &newCombine = *currentBlock->getNextNode();

  llvm::SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  auto isRegionSpeculatable =
      std::all_of(newCombine.begin(), newCombine.end(),
                  [](auto &op) { return isSpeculatable(&op); });

  if (!pred || isRegionSpeculatable) {
    // Fast path, region has no side effects so we can unconditionally execute
    return inlineCombineBlock(rewriter, newCombine, currentBlock,
                              rewriter.getInsertionPoint(), combineArgs);
  }

  // Slow case, create an if to only execute region when pred is true
  // #currentBlock
  // if (pred) {
  //   #newCombine
  //   results = combineOp(cur, acc)
  //   yield results
  // } else {
  //    yield undef
  // }
  // #thenBlock
  Block *thenBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  auto returnOp = newCombine.getTerminator();
  auto results = SmallVector<Value>(returnOp->getOperands());

  rewriter.setInsertionPointToEnd(currentBlock);
  SmallVector<Value> thenBlockArgs;
  thenBlockArgs.reserve(results.size());
  for (auto result : results) {
    auto ty = result.getType();
    auto undef = rewriter.create<LLVM::UndefOp>(loc, ty);
    thenBlockArgs.push_back(undef);
    thenBlock->addArgument(ty, loc);
  }
  rewriter.create<cf::CondBranchOp>(loc, pred, &newCombine, combineArgs,
                                    thenBlock, thenBlockArgs);

  // Split a block after the call.
  rewriter.setInsertionPointToEnd(&newCombine);
  rewriter.replaceOpWithNewOp<cf::BranchOp>(returnOp, thenBlock, results);
  rewriter.setInsertionPointToStart(thenBlock);
  return SmallVector<Value>(thenBlock->getArguments());
}

// apply combine region to acc and cur and accumulate it into acc
static SmallVector<Value> accumulate(ScanLoweringHelper &helper,
                                     ConversionPatternRewriter &rewriter,
                                     ValueRange acc, ValueRange cur,
                                     Value pred = {}) {
  auto loc = helper.getXPULoc();
  auto &combineOp = helper.getXPUCombineOp();
  return applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
}

// Scan a contiguous elements within a thread and update `srcValues` in place.
static void
scanThreadContiguousElements(SmallVector<SmallVector<Value>> &srcValues,
                             ConversionPatternRewriter &rewriter,
                             ScanLoweringHelper &helper) {
  // Depending on layout contiguous elements along axis dim may not be
  // contiguous in srcValues. Keep track of what elements belong to the same
  // chunk of contiguous elements.
  SmallVector<SmallVector<Value>> accs(srcValues.size());
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    // Change this into emitOffsetForLayout?
    unsigned accIndex = srcIndex;

    if (srcIndex == 0) {
      accs[srcIndex] =
          accumulate(helper, rewriter, accs[srcIndex], srcValues[srcIndex]);
    } else {
      accs[srcIndex] =
          accumulate(helper, rewriter, accs[srcIndex - 1], srcValues[srcIndex]);
    }
    srcValues[srcIndex] = accs[accIndex];
  }
}

// Read the partial reductions from shared memory from each chunk of contiguous
// elements for each warp and parallel scan. Then combine the partial reduction
// with the right elements. Within a given contiguous element chunk we update
// all the elements by accumulating the value from the last element of the
// reduced value from the previous lane.
static void applyLastElemScanOnSM(SmallVector<SmallVector<Value>> &srcValues,
                                  ConversionPatternRewriter &rewriter,
                                  const TargetInfoBase &targetInfo,
                                  ScanLoweringHelper &helper,
                                  SmallVector<Value> smemBases,
                                  SmallVector<Value> accSmemBases,
                                  SmallVector<Type> smemTypes, Value groupId,
                                  Value laneId, triton::xpu::ScanOp op) {
  Location loc = helper.getXPULoc();

  // Scan SM Last Elem Per Thread
  Value threadId = getThreadId(rewriter, loc);
  Value zero = i32_val(0);
  Value coreIdInGroupZero = icmp_eq(threadId, zero);
  unsigned groupSizeInt = 64;

  auto thenBlock = rewriter.splitBlock(rewriter.getInsertionBlock(),
                                       rewriter.getInsertionPoint());
  auto mergeBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());
  rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
  rewriter.create<LLVM::CondBrOp>(loc, coreIdInGroupZero, thenBlock,
                                  mergeBlock);

  rewriter.setInsertionPointToStart(thenBlock);

  SmallVector<SmallVector<Value>> readValues(groupSizeInt);
  // readValues[i][j]: the j-th operand's core_i lastScanElem
  for (unsigned readOffset = 0; readOffset < groupSizeInt; ++readOffset) {
    for (unsigned operandIdx = 0; operandIdx < (helper.getXPUNumOperands() - 1);
         ++operandIdx) { // skip loopIndex
      auto elemTy = smemTypes[operandIdx];
      Value readPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                          smemBases[operandIdx], i32_val(readOffset));
      auto readVal = load_sm(elemTy, readPtr);
      readValues[readOffset].push_back(readVal);
    }
  }

  // auto dump2DSmallVector = [](const SmallVector<SmallVector<Value>>
  // &srcValues,
  //                             std::string str) {
  //   assert(srcValues.size() > 0 && srcValues[0].size() > 0);
  //   llvm::errs() << "shape: " << srcValues.size() << "x" <<
  //   srcValues[0].size()
  //                << "\n";
  //   for (auto [i, srcValue] : llvm::enumerate(srcValues)) {
  //     llvm::errs() << str << "[" << i << "]: ";

  //     for (auto [j, srcVal] : llvm::enumerate(srcValue)) {
  //       llvm::errs() << srcVal << ", ";
  //     }
  //     llvm::errs() << "\n ";
  //   }
  // };

  //   llvm::errs() << "\n [After readValues] Dump readValues:\n";
  //   dump2DSmallVector(readValues, "readValues");

  SmallVector<SmallVector<Value>> accs(groupSizeInt);

  std::string calFunc = "_ZN3xpu10printFloatEfi";

  for (unsigned srcIdx = 0; srcIdx < groupSizeInt; ++srcIdx) {
    if (srcIdx == 0) { // core0 dont't need to acc
      accs[srcIdx] =
          accumulate(helper, rewriter, accs[srcIdx], readValues[srcIdx]);

      //   ValueRange operandValueRange({accs[srcIdx][0], i32_val(srcIdx)});
      //   mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
      //                                     operandValueRange, loc);

    } else if (srcIdx == 1) { // core1 assign accSmem[0] value directly
      accs[srcIdx] = accs[srcIdx - 1];
      //   accs[srcIdx] = readValues[srcIdx - 1];

      //   ValueRange operandValueRange({accs[srcIdx][0], i32_val(srcIdx)});
      //   mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
      //                                     operandValueRange, loc);
    } else {
      accs[srcIdx] = accumulate(helper, rewriter, accs[srcIdx - 1],
                                readValues[srcIdx - 1]);

      //   ValueRange operandValueRange(
      //       {accs[srcIdx - 1][0], i32_val(100 + srcIdx)});
      //   mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
      //                                     operandValueRange, loc);

      //   ValueRange operandValueRange_1(
      //       {readValues[srcIdx][0], i32_val(200 + srcIdx)});
      //   mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
      //                                     operandValueRange_1, loc);

      //   ValueRange operandValueRange_2({accs[srcIdx][0], i32_val(300 +
      //   srcIdx)}); mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
      //                                     operandValueRange_1, loc);
    }
  }

  // [dump] dump accs result
  if (/*dump_Out_SMValue*/ false) {
    auto operandIdx = 1; // 第一个输入

    for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
      auto elemTy = accs[core_id_offset][operandIdx].getType();

      std::string calFunc;
      if (elemTy.isF32()) {
        calFunc = "_ZN3xpu15printFloat_specEfiii";
      } else if (elemTy.isInteger(32)) {
        calFunc = "_ZN3xpu13printInt_specEiiii";
      } else if (elemTy.isInteger(64)) {
        calFunc = "_ZN3xpu15printInt64_specEliii";
      }

      ValueRange operandValueRange(
          {accs[core_id_offset][operandIdx], /*cluster_id*/ i32_val(0),
           /*core_id*/ i32_val(0),
           /*custom_id*/ i32_val(200 + core_id_offset)});
      mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                        operandValueRange, loc);
    }
  }

  //   llvm::errs() << "\n [After accumulate readValues] Dump accs:\n";
  //   dump2DSmallVector(accs, "accs");

  for (unsigned writeOffset = 0; writeOffset < groupSizeInt; ++writeOffset) {
    for (unsigned operandIdx = 0; operandIdx < (helper.getXPUNumOperands() - 1);
         ++operandIdx) { // skip loopIndex
      auto elemTy = smemTypes[operandIdx];
      Value writePtrs = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                            accSmemBases[operandIdx],
                            /*writeOffset*/ i32_val(writeOffset));
      //   llvm::errs() << "[writePtrs-" << writeOffset << "]" << writePtrs <<
      //   "\n";
      store_sm(accs[writeOffset][operandIdx], writePtrs);
    }
  }

  // [dump][sm] operand-0: sm_acc[0]-sm_acc[63]   or  operand-1
  // sm_acc[64]-sm_acc[127] sm_acc = sm[128]
  if (/*dump_Out_SMValue*/ false) {
    auto operandIdx = 1; // 第一个输入

    for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
      auto elemTy = smemTypes[operandIdx];
      Value loadPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                          accSmemBases[operandIdx], i32_val(core_id_offset));
      Value loadVal = load_sm(elemTy, loadPtr);

      std::string calFunc;
      if (elemTy.isF32()) {
        calFunc = "_ZN3xpu15printFloat_specEfiii";
      } else if (elemTy.isInteger(32)) {
        calFunc = "_ZN3xpu13printInt_specEiiii";
      } else if (elemTy.isInteger(64)) {
        calFunc = "_ZN3xpu15printInt64_specEliii";
      }

      ValueRange operandValueRange(
          {loadVal, /*cluster_id*/ i32_val(0), /*core_id*/ i32_val(0),
           /*custom_id*/ i32_val(600 + core_id_offset)});
      mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                        operandValueRange, loc);
    }
  }

  rewriter.create<LLVM::BrOp>(loc, mergeBlock);
  rewriter.setInsertionPointToStart(mergeBlock);
}

static void applyCoreElemScanWithSMAcc(
    SmallVector<SmallVector<Value>> &srcValues,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    ScanLoweringHelper &helper, SmallVector<Value> smemBases,
    SmallVector<Value> accSmemBases, SmallVector<Type> smemTypes, Value groupId,
    Value laneId, triton::xpu::ScanOp op) {
  Location loc = helper.getXPULoc();
  Value threadId = getThreadId(rewriter, loc);

  Value zero = i32_val(0);
  Value coreIdInGroupZero = icmp_eq(threadId, zero);

  //   auto thenBlock = rewriter.splitBlock(rewriter.getInsertionBlock(),
  //                                        rewriter.getInsertionPoint());
  //   auto mergeBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());
  //   rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
  //   rewriter.create<LLVM::CondBrOp>(loc, coreIdInGroupZero, thenBlock,
  //                                   mergeBlock);

  //   rewriter.setInsertionPointToStart(thenBlock);

  // Load the SM Val Per CoreTile
  SmallVector<Value> cur_core_acc_val(helper.getXPUNumOperands() - 1);
  for (unsigned operandIdx = 0; operandIdx < (helper.getXPUNumOperands() - 1);
       ++operandIdx) { // skip loopIndex
    auto elemTy = smemTypes[operandIdx];
    Value readCurCoreAccValOnSMPtrs =
        gep(ptr_ty(rewriter.getContext(), 2), elemTy, accSmemBases[operandIdx],
            /*readOffset*/ threadId);
    // llvm::errs() << "[readCurCoreAccValOnSMPtrs-" << operandIdx << "]"
    //              << readCurCoreAccValOnSMPtrs << "\n";
    cur_core_acc_val[operandIdx] = load_sm(elemTy, readCurCoreAccValOnSMPtrs);
  }

  // [dump][lm] core load sm[coreId] data to lm
  if (/*dump_Out_SMValue*/ false) {
    auto operandIdx = 1;
    auto elemTy = cur_core_acc_val[operandIdx].getType();

    std::string calFunc;
    if (elemTy.isF32()) {
      calFunc = "_ZN3xpu10printFloatEfi";
    } else if (elemTy.isInteger(32)) {
      calFunc = "_ZN3xpu8printIntEii";
    } else if (elemTy.isInteger(64)) {
      calFunc = "_ZN3xpu10printInt64Eli";
    }

    ValueRange operandValueRange({cur_core_acc_val[operandIdx],
                                  /*custom_id*/ i32_val(300)});
    mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op, operandValueRange,
                                      loc);
  }

  // auto dump1DSmallVector = [](const SmallVector<Value> &results,
  //                             std::string str) {
  //   assert(results.size() > 0);
  //   llvm::errs() << "shape: " << results.size() << "\n";
  //   llvm::errs() << str << ": ";
  //   for (auto [i, srcValue] : llvm::enumerate(results)) {
  //     llvm::errs() << srcValue << ", ";
  //   }
  //   llvm::errs() << "\n ";
  // };

  //   llvm::errs()
  //       << "\n [After readCurCoreAccValOnSMPtrs] Dump cur_core_acc_val:\n";
  //   dump1DSmallVector(cur_core_acc_val, "cur_core_acc_val");

  //  Add the SM Acc Val Per CoreTile
  // final accumulate(cur_core_acc_val & srcValue)

  //   for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
  //     srcValues[srcIndex] =
  //         accumulate(helper, rewriter, cur_core_acc_val,
  //         srcValues[srcIndex]);
  //   }

  SmallVector<SmallVector<Value>> newSrcValues(srcValues.size());
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    newSrcValues[srcIndex] =
        accumulate(helper, rewriter, cur_core_acc_val, srcValues[srcIndex]);
  }

  // [dump][lm] final srcValues
  if (/*dump_Out_SMValue*/ false) {
    auto coreElemIdx = 0;
    auto operandIdx = 0;
    auto elemTy = newSrcValues[coreElemIdx][operandIdx].getType();

    std::string calFunc;
    if (elemTy.isF32()) {
      calFunc = "_ZN3xpu10printFloatEfi";
    } else if (elemTy.isInteger(32)) {
      calFunc = "_ZN3xpu8printIntEii";
    } else if (elemTy.isInteger(64)) {
      calFunc = "_ZN3xpu10printInt64Eli";
    }

    ValueRange operandValueRange({newSrcValues[coreElemIdx][operandIdx],
                                  /*custom_id*/ i32_val(3000)});
    mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op, operandValueRange,
                                      loc);
  }

  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    for (unsigned operandIdx = 0; operandIdx < (helper.getXPUNumOperands() - 1);
         ++operandIdx) { // skip loopIndex
      srcValues[srcIndex][operandIdx] =
          select(coreIdInGroupZero, srcValues[srcIndex][operandIdx],
                 newSrcValues[srcIndex][operandIdx]);
    }
  }

  //   rewriter.create<LLVM::BrOp>(loc, mergeBlock);
  //   rewriter.setInsertionPointToStart(mergeBlock);
}

namespace {

struct XPUScanOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ScanOp> {

  XPUScanOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ScanOp>(converter, benefit),
        targetInfo(targetInfo) {}

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(triton::xpu::ScanOp op, int i) const {
    auto ty = getElementTypeOrSelf(op.getInputTypes()[i].getElementType());
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  std::pair<SmallVector<Value>, SmallVector<Value>>
  getSmemBases(triton::xpu::ScanOp op, unsigned elems,
               ConversionPatternRewriter &rewriter) const {
    ScanLoweringHelper helper(op);
    SmallVector<int64_t> offsets;

    auto prevSMOffset =
        helper.getScanId() == 0
            ? 0 // [TODO]: find reduceOp and replace the last endOffset
            : helper.getSMOffsets(helper.getScanId() - 1)->endOffset;

    auto loc = op.getLoc();
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    // std::vector<unsigned> indices(op.getNumOperands() - 1); // skip loopIndex
    // std::iota(indices.begin(), indices.end(), 0);

    // std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
    //   if (i == op.getNumOperands() - 1 ||
    //       j == op.getNumOperands() - 1) { // skip loopIndex
    //     return false;
    //   }
    //   return op.getElementTypes()[i].getIntOrFloatBitWidth() >
    //          op.getElementTypes()[j].getIntOrFloatBitWidth();
    // });

    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[0] =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    // add prev reduceOp used sm bytes offset
    indexToBase[0] =
        gep(ptr_ty(rewriter.getContext(), 2), getElementType(op, 0),
            indexToBase[0], i32_val(prevSMOffset));

    offsets.push_back((getElementType(op, 0).getIntOrFloatBitWidth() * elems) /
                      8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBase[i] =
          gep(ptr_ty(rewriter.getContext(), 2), getElementType(op, i),
              indexToBase[i - 1], i32_val(elems));
      offsets.push_back(
          (getElementType(op, i).getIntOrFloatBitWidth() * elems) / 8);
    }

    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands() - 1);     // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      smemBases[i] = indexToBase[i];
    }

    // accCacheSmemBases[k] is the base pointer for the k-th operand which
    // is the prev accmulate result
    SmallVector<Value> accCacheSmemBases(op.getNumOperands() -
                                         1); // skip loopIndex
    std::map<unsigned, Value> indexToBaseForAccCache;
    indexToBaseForAccCache[0] = gep(ptr_ty(rewriter.getContext(), 2),
                                    getElementType(op, op.getNumOperands() - 2),
                                    smemBases.back(), i32_val(elems));
    offsets.push_back((getElementType(op, 0).getIntOrFloatBitWidth() * elems) /
                      8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBaseForAccCache[i] =
          gep(ptr_ty(rewriter.getContext(), 2), getElementType(op, i),
              indexToBaseForAccCache[i - 1], i32_val(elems));
      offsets.push_back(
          (getElementType(op, i).getIntOrFloatBitWidth() * elems) / 8);
    }

    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      accCacheSmemBases[i] = indexToBaseForAccCache[i];
    }

    helper.setSMOffsets(helper.getScanId(), offsets);
    return {smemBases, accCacheSmemBases};
  }

  // For each set of contiguous elements within a thread we store the partial
  // reduction into shared memory. Each parallel scan and each group will
  // store its own partial reductions. The shared memory is organized as
  // follow:
  static void storeGroupAccumulator(SmallVector<SmallVector<Value>> &srcValues,
                                    ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper, Value laneId,
                                    Value groupId, SmallVector<Value> smemBases,
                                    SmallVector<Type> smemTypes) {
    Location loc = helper.getXPULoc();
    Value threadId = getThreadId(rewriter, loc);

    for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
      // Only consider the last element of each contiguous chunk of elements.
      if (srcIndex != srcValues.size() - 1)
        continue;

      auto lastElement = srcValues[srcIndex];

      for (unsigned i = 0; i < lastElement.size();
           ++i) { // lastElement.size() == operand size
        Value writePtr =
            gep(ptr_ty(rewriter.getContext(), 2), smemTypes[i], smemBases[i],
                /*writeOffset*/ threadId);
        store_sm(lastElement[i], writePtr);
      }
    }
  }

  LogicalResult
  matchAndRewrite(triton::xpu::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ScanLoweringHelper helper(op);
    auto loc = helper.getXPULoc();

    Value threadId = getThreadId(rewriter, loc);
    unsigned groupSizeInt = helper.getIntraGroupSizeWithUniqueData();
    Value groupSize = i32_val(groupSizeInt);
    Value groupId = udiv(threadId, groupSize);
    Value laneId = urem(threadId, groupSize);

    // dim-0: coreElems
    // dim-1: operandNums
    auto srcValues =
        unpackInputs(loc, op, adaptor, rewriter, *getTypeConverter());

    auto dump2DSmallVector =
        [](const SmallVector<SmallVector<Value>> &srcValues, std::string str) {
          assert(srcValues.size() > 0 && srcValues[0].size() > 0);
          llvm::errs() << "shape: " << srcValues.size() << "x"
                       << srcValues[0].size() << "\n";
          for (auto [i, srcValue] : llvm::enumerate(srcValues)) {
            llvm::errs() << str << "[" << i << "]: ";

            for (auto [j, srcVal] : llvm::enumerate(srcValue)) {
              llvm::errs() << srcVal << ", ";
            }
            llvm::errs() << "\n ";
          }
        };

    // llvm::errs() << "\n [After unpackInputs] Dump srcValues:\n";
    // dump2DSmallVector(srcValues, "srcValues");

    // [dump][lm] operand-0: src[:][0]   or  operand-1 src[:][1]
    if (/*dump_srcValue*/ false) {
      auto operandIdx = 1;
      std::string calFunc;
      auto elemTy = srcValues[0][operandIdx].getType();

      for (size_t cnt = 0; cnt < srcValues.size(); ++cnt) {
        if (elemTy.isF32()) {
          calFunc = "_ZN3xpu10printFloatEfi";
        } else if (elemTy.isInteger(32)) {
          calFunc = "_ZN3xpu8printIntEii";
        } else if (elemTy.isInteger(64)) {
          calFunc = "_ZN3xpu10printInt64Eli";
        }
        ValueRange operandValueRange(
            {srcValues[cnt][operandIdx], i32_val(cnt + 400)});
        mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                          operandValueRange, loc);
      }
    }

    if (op.getReverse()) {
      llvm_unreachable("wait support reverse");
    }

    // Scan contiguous elements in a thread and update `srcValues`.
    scanThreadContiguousElements(srcValues, rewriter, helper);
    // llvm::errs() << "\n After scanThreadContiguousElements:"
    //              << op->getParentOfType<ModuleOp>() << "\n";
    // llvm::errs() << "\n [After scanThreadContiguousElements] Dump
    // srcValues:\n"; dump2DSmallVector(srcValues, "srcValues");

    if (!helper.isCoreSynchronous()) {
      // llvm::errs() << "\n [Before getSmemBases]:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";
      // helper.dumpSMOffsets();

      auto elems = helper.getScratchSizeInElemsXPU();
      auto [smemBases, accSmemBases] = getSmemBases(op, elems, rewriter);

      // llvm::errs() << "\n [After getSmemBases]:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";
      // helper.dumpSMOffsets();

      SmallVector<Type> smemTypes(op.getNumOperands() -
                                  1); // skip skip loopIndex
      for (unsigned i = 0; i < (op.getNumOperands() - 1);
           ++i) { // skip skip loopIndex
        smemTypes[i] = getElementType(op, i);
      }

      xpu_barrier(); // Deal with Cumcum(Reduce+Scan) Case

      storeGroupAccumulator(srcValues, rewriter, helper, laneId, groupId,
                            smemBases, smemTypes);

      // [dump][sm] operand-0: sm[0]-sm[63]   or  operand-1 sm[64]-sm[127]
      if (/*dump_In_SMValue*/ false) {
        auto operandIdx = 1; // 第一个输入

        for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
          auto elemTy = smemTypes[operandIdx];
          Value loadPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                              smemBases[operandIdx], i32_val(core_id_offset));
          Value loadVal = load_sm(elemTy, loadPtr);

          std::string calFunc;
          if (elemTy.isF32()) {
            calFunc = "_ZN3xpu15printFloat_specEfiii";
          } else if (elemTy.isInteger(32)) {
            calFunc = "_ZN3xpu13printInt_specEiiii";
          } else if (elemTy.isInteger(64)) {
            calFunc = "_ZN3xpu15printInt64_specEliii";
          }

          ValueRange operandValueRange(
              {loadVal, /*cluster_id*/ i32_val(0), /*core_id*/ i32_val(0),
               /*custom_id*/ i32_val(500 + core_id_offset)});
          mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                            operandValueRange, loc);
        }
      }

      // llvm::errs() << "\n After storeGroupAccumulator:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";
      //   llvm::errs() << "\n [After storeGroupAccumulator] Dump srcValues:\n";
      //   dump2DSmallVector(srcValues, "srcValues");

      xpu_barrier();

      // Read back the partial reduction of each warp and accumulate them
      // based on warpId.
      applyLastElemScanOnSM(srcValues, rewriter, targetInfo, helper, smemBases,
                            accSmemBases, smemTypes, groupId, laneId, op);
      // llvm::errs() << "\n After applyLastElemScanOnSM:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";

      xpu_barrier();

      // [dump][sm] operand-0: sm_acc[0]-sm_acc[63]   or  operand-1
      // sm_acc[64]-sm_acc[127] sm_acc = sm[128]
      if (/*dump_Out_SMValue*/ false) {
        auto operandIdx = 1; // 第一个输入

        for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
          auto elemTy = smemTypes[operandIdx];
          Value loadPtr =
              gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                  accSmemBases[operandIdx], i32_val(core_id_offset));
          Value loadVal = load_sm(elemTy, loadPtr);

          std::string calFunc;
          if (elemTy.isF32()) {
            calFunc = "_ZN3xpu15printFloat_specEfiii";
          } else if (elemTy.isInteger(32)) {
            calFunc = "_ZN3xpu13printInt_specEiiii";
          } else if (elemTy.isInteger(64)) {
            calFunc = "_ZN3xpu15printInt64_specEliii";
          }

          ValueRange operandValueRange(
              {loadVal, /*cluster_id*/ i32_val(0), /*core_id*/ i32_val(0),
               /*custom_id*/ i32_val(600 + core_id_offset)});
          mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                            operandValueRange, loc);
        }
      }
      xpu_barrier();

      // llvm::errs() << "\n Before applyCoreElemScanWithSMAcc:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";
      // Then update each chunk of contiguous elements by
      // adding the accumulated value from the previous lane.
      applyCoreElemScanWithSMAcc(srcValues, rewriter, targetInfo, helper,
                                 smemBases, accSmemBases, smemTypes, groupId,
                                 laneId, op);
      // llvm::errs() << "\n After applyCoreElemScanWithSMAcc:\n"
      //              << op->getParentOfType<ModuleOp>() << "\n";

      // dump operand-0: sm_acc[0]-sm_acc[63]   or  operand-1
      // sm_acc[64]-sm_acc[127] sm_acc = sm[128]
      if (/*dump_Out_SMValue*/ false) {
        auto operandIdx = 0; // 第一个输入

        for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
          auto elemTy = smemTypes[operandIdx];
          Value loadPtr =
              gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                  accSmemBases[operandIdx], i32_val(core_id_offset));
          Value loadVal = load_sm(elemTy, loadPtr);

          std::string calFunc;
          if (elemTy.isF32()) {
            calFunc = "_ZN3xpu15printFloat_specEfiii";
          } else if (elemTy.isInteger(32)) {
            calFunc = "_ZN3xpu13printInt_specEiiii";
          } else if (elemTy.isInteger(64)) {
            calFunc = "_ZN3xpu15printInt64_specEliii";
          }

          ValueRange operandValueRange(
              {loadVal, /*cluster_id*/ i32_val(0), /*core_id*/ i32_val(0),
               /*custom_id*/ i32_val(700 + core_id_offset)});
          mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                            operandValueRange, loc);
        }
      }

      //   llvm::errs() << "\n [After applyCoreElemScanWithSMAcc] Dump
      //   srcValues:\n"; dump2DSmallVector(srcValues, "srcValues");

      xpu_barrier();
    }

    auto transpose = [](const SmallVector<SmallVector<Value>> &v) {
      assert(v.size() > 0 && v[0].size() > 0);
      auto ret = SmallVector<SmallVector<Value>>(v[0].size(),
                                                 SmallVector<Value>(v.size()));
      for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v[0].size(); ++j) {
          ret[j][i] = v[i][j];
        }
      }
      return ret;
    };

    auto valuesTransposed = transpose(srcValues);
    // llvm::errs() << "\n After transpose:" << op->getParentOfType<ModuleOp>()
    //              << "\n";
    // llvm::errs() << "\n [After transpose] Dump valuesTransposed:\n";
    // dump2DSmallVector(valuesTransposed, "valuesTransposed");

    SmallVector<Value> results(op.getNumOperands() - 1); // skip loopIndex
    if (op.getReverse()) {
      llvm_unreachable("wait support reverse");
    }

    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto resultTy = dyn_cast<RankedTensorType>(op.getResult()[i].getType());
      results[i] = packLLElements(loc, getTypeConverter(), valuesTransposed[i],
                                  rewriter, resultTy);
    }

    auto dump1DSmallVector = [](const SmallVector<Value> &results,
                                std::string str) {
      assert(results.size() > 0);
      llvm::errs() << "shape: " << results.size() << "\n";
      llvm::errs() << str << ": ";
      for (auto [i, srcValue] : llvm::enumerate(results)) {
        llvm::errs() << srcValue << ", ";
      }
      llvm::errs() << "\n ";
    };

    // llvm::errs() << "\n Before replaceOp:" << op->getParentOfType<ModuleOp>()
    //              << "\n";
    // llvm::errs() << "\n [Before replaceOp] Dump results:\n";
    // dump1DSmallVector(results, "results");

    rewriter.replaceOp(op, results);

    // llvm::errs() << "\n After replaceOp:" << op->getParentOfType<ModuleOp>()
    //              << "\n";
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::xpu::ScanOp op,
               triton::xpu::ScanOpAdaptor adaptor,
               ConversionPatternRewriter &rewriter,
               const LLVMTypeConverter &converter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }
};

} // namespace

void mlir::triton::xpu::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<XPUScanOpConversion>(typeConverter, targetInfo, benefit);
}
