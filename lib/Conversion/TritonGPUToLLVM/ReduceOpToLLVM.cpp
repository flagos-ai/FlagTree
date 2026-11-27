/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ReduceScanCommon.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(helper.isReduceWithinCTA() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();

#ifdef __TLE__
    if (succeeded(
            tryLowerCtaOrReductionFastpath(op, adaptor, helper, rewriter))) {
      return success();
    }
#endif

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    if (canDirectGatherInterWarp(op, helper, smemShape)) {
      syncInterWarp(rewriter, loc, op, helper);
      directGatherAndPackResult(helper, smemShape, smemBases, rewriter);
      return success();
    }

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    accumulatePartialReductions(helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

#ifdef __TLE__
  bool isOrReductionToScalarI1(triton::ReduceOp op) const {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return false;
    if (!op.getResult()[0].getType().isInteger(1))
      return false;
    auto *combine = op.getSingleCombiner();
    return combine && isa<arith::OrIOp>(combine);
  }

  LogicalResult
  tryLowerCtaOrReductionFastpath(triton::ReduceOp op, OpAdaptor adaptor,
                                 ReduceOpHelper &helper,
                                 ConversionPatternRewriter &rewriter) const {
    if (helper.isWarpSynchronous() || !isOrReductionToScalarI1(op))
      return failure();

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::optional<Value> threadAny;
    for (unsigned i = 0; i < srcValues.size(); ++i) {
      Value pred = srcValues[i][0];
      if (!pred.getType().isInteger(1)) {
        Type predTy = pred.getType();
        pred = b.icmp_ne(pred, b.int_val(predTy.getIntOrFloatBitWidth(), 0));
      }
      threadAny = threadAny ? b.or_(*threadAny, pred) : pred;
    }
    if (!threadAny)
      return failure();

    std::optional<Value> ctaAny =
        targetInfo.ctaReduceOrPredicate(rewriter, loc, *threadAny);
    if (!ctaAny)
      return failure();

    rewriter.replaceOp(op, *ctaAny);
    return success();
  }
#endif

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier();
  }

  bool canDirectGatherInterWarp(triton::ReduceOp op, ReduceOpHelper &helper,
                                ArrayRef<unsigned> smemShape) const {
    if (!targetInfo.preferDirectInterWarpReduceGather())
      return false;
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    if (sizeInterWarps != 2 && sizeInterWarps != 4 && sizeInterWarps != 8)
      return false;
    if (smemShape[op.getAxis()] != sizeInterWarps)
      return false;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Type elemTy = getElementType(op, i);
      if (!elemTy.isIntOrFloat())
        return false;
      unsigned bw = elemTy.getIntOrFloatBitWidth();
      if (bw != 16 && bw != 32 && bw != 64)
        return false;
      if ((sizeInterWarps * bw) % 32 != 0)
        return false;
    }
    return true;
  }

  void syncInterWarp(ConversionPatternRewriter &rewriter, Location loc,
                     triton::ReduceOp op, ReduceOpHelper &helper) const {
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned numWarps = triton::gpu::lookupNumWarps(op);
    unsigned axis = op.getAxis();

    // Use a subgroup-scoped shared-memory barrier when warp subgroups access
    // disjoint slices of the reduction scratch buffer and physical named
    // barrier ids [1 .. numSubgroups] are free.
    if (sizeInterWarps > 1 && sizeInterWarps < numWarps &&
        llvm::isPowerOf2_32(sizeInterWarps) && llvm::isPowerOf2_32(numWarps)) {
      unsigned numSubgroups = numWarps / sizeInterWarps;
      unsigned k = llvm::Log2_32(sizeInterWarps);
      auto func = op->getParentOfType<FunctionOpInterface>();
      bool hasNamedOrWS = false;
      if (func) {
        func.walk([&](Operation *child) {
          StringRef name = child->getName().getStringRef();
          if (name == "ttg.warp_specialize" || name.contains("barrier_named"))
            hasNamedOrWS = true;
        });
      }
      if (func && !hasNamedOrWS && 1 + numSubgroups <= 16 &&
          isa<RankedTensorType>(op.getResult()[0].getType())) {
        auto srcTy = cast<RankedTensorType>(op.getInputTypes()[0]);
        auto srcLL = mlir::triton::gpu::toLinearLayout(srcTy);
        auto kWarp = rewriter.getStringAttr("warp");
        // Subgroup barriers require every warp to be representative (no
        // broadcast warp bits) and upper warp bits to index disjoint slices.
        if (srcLL.hasInDim(kWarp) &&
            srcLL.getFreeVariableMasks().lookup(kWarp) == 0) {
          const auto &warpBases = srcLL.getBases().lookup(kWarp);
          unsigned rank = srcTy.getRank();
          SmallVector<unsigned> subgroupMask(rank, 0);
          SmallVector<unsigned> innerMask(rank, 0);
          bool disjoint = (warpBases.size() == llvm::Log2_32(numWarps));
          for (unsigned bit = 0; disjoint && bit < warpBases.size(); ++bit) {
            const auto &basis = warpBases[bit];
            if (bit < k) {
              if (basis[axis] == 0)
                disjoint = false;
              for (unsigned d = 0; d < rank; ++d)
                if (d != axis && basis[d] != 0)
                  disjoint = false;
            } else {
              if (basis[axis] != 0)
                disjoint = false;
              bool hasNonAxis = false;
              for (unsigned d = 0; d < rank; ++d) {
                if (d != axis && basis[d] != 0) {
                  hasNonAxis = true;
                  subgroupMask[d] |= static_cast<unsigned>(basis[d]);
                }
              }
              if (!hasNonAxis)
                disjoint = false;
            }
          }
          if (disjoint) {
            for (const auto &entry : srcLL.getBases()) {
              if (entry.first == kWarp)
                continue;
              for (const auto &basis : entry.second)
                for (unsigned d = 0; d < rank; ++d)
                  innerMask[d] |= static_cast<unsigned>(basis[d]);
            }
            for (unsigned d = 0; d < rank; ++d) {
              if (d != axis && (subgroupMask[d] & innerMask[d]) != 0) {
                disjoint = false;
                break;
              }
            }
          }
          if (disjoint) {
            auto b = TritonLLVMOpBuilder(loc, rewriter);
            auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
            Value subgroupId = b.lshr(warpId, b.i32_val(k));
            if (targetInfo.subgroupBarrier(loc, rewriter, subgroupId,
                                           sizeInterWarps * numLanes))
              return;
          }
        }
      }
    }
    targetInfo.barrier(loc, rewriter, /*isWarpSync=*/false);
  }

  // Directly vector-load all sizeInterWarps contiguous partials from SMEM and
  // combine them locally in registers, eliminating the second warp-0 shuffle,
  // SMEM write-back, and second CTA barrier.
  void directGatherAndPackResult(ReduceOpHelper &helper,
                                 SmallVector<unsigned> smemShape,
                                 SmallVector<Value> &smemBases,
                                 ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned axis = op.getAxis();
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    unsigned numOps = op.getNumOperands();

    auto gatherAtOffset = [&](Value readOffset) -> SmallVector<Value> {
      SmallVector<SmallVector<Value>> laneAccs(sizeInterWarps);
      for (unsigned w = 0; w < sizeInterWarps; ++w)
        laneAccs[w].reserve(numOps);

      for (unsigned i = 0; i < numOps; ++i) {
        Type elemTy = getElementType(op, i);
        unsigned bw = elemTy.getIntOrFloatBitWidth();
        unsigned vecWidth = std::min<unsigned>(sizeInterWarps, 128 / bw);
        unsigned numVecs = sizeInterWarps / vecWidth;
        for (unsigned v = 0; v < numVecs; ++v) {
          Value off = (v == 0) ? readOffset
                               : b.add(readOffset, b.i32_val(v * vecWidth));
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], off);
          if (vecWidth > 1) {
            auto vecTy =
                VectorType::get({static_cast<int64_t>(vecWidth)}, elemTy);
            Value vec = b.load(vecTy, readPtr);
            for (unsigned u = 0; u < vecWidth; ++u)
              laneAccs[v * vecWidth + u].push_back(
                  b.extract_element(vec, b.i32_val(u)));
          } else {
            laneAccs[v].push_back(b.load(elemTy, readPtr));
          }
        }
      }
      // Binary-tree combine matching warpReduce order: (0 op N/2), ...
      for (unsigned step = sizeInterWarps / 2; step >= 1; step >>= 1) {
        for (unsigned w = 0; w < step; ++w) {
          accumulate(loc, rewriter, op.getCombineOp(), laneAccs[w],
                     laneAccs[w + step]);
        }
      }
      return laneAccs[0];
    };

    SmallVector<Value> results(numOps);
    if (auto resultTy0 =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto resultLayout = cast<SliceEncodingAttr>(resultTy0.getEncoding());
      unsigned resultElems = getTotalElemsPerThread(resultTy0);
      auto resultIndices =
          emitIndices(loc, rewriter, targetInfo, resultLayout, resultTy0, true);
      auto resultShape = resultTy0.getShape();
      assert(resultIndices.size() == resultElems);

      SmallVector<SmallVector<Value>> allResultVals(
          numOps, SmallVector<Value>(resultElems));
      for (unsigned j = 0; j < resultElems; ++j) {
        SmallVector<Value> readIdx = resultIndices[j];
        readIdx.insert(readIdx.begin() + axis, b.i32_val(0));
        for (size_t resultIdx = 0, resultDim = resultShape.size();
             resultIdx < resultDim; ++resultIdx) {
          auto smemIdx = resultIdx < axis ? resultIdx : resultIdx + 1;
          if (resultShape[resultIdx] > smemShape[smemIdx]) {
            readIdx[smemIdx] =
                b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
          }
        }
        Value readOffset =
            linearize(rewriter, loc, readIdx, smemShape, smemOrder);
        auto combined = gatherAtOffset(readOffset);
        for (unsigned i = 0; i < numOps; ++i)
          allResultVals[i][j] = combined[i];
      }
      for (unsigned i = 0; i < numOps; ++i) {
        auto resTy = cast<RankedTensorType>(op.getResult()[i].getType());
        results[i] = packLLElements(loc, getTypeConverter(), allResultVals[i],
                                    rewriter, resTy);
      }
    } else {
      auto combined = gatherAtOffset(b.i32_val(0));
      for (unsigned i = 0; i < numOps; ++i)
        results[i] = combined[i];
    }
    rewriter.replaceOp(op, results);
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumulate over
    // those.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    // reduce within threads
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout =
        mlir::cast<DistributedEncodingTrait>(helper.getSrcLayout());
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto srcShape = helper.getSrcShape();
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, srcLayout, srcShape, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, srcLayout, srcShape, kWarp, warpId);

    Value laneIdAxis = multiDimLaneId[axis];
    Value laneZero = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value write =
        b.and_(b.and_(isRepresentativeLane, isRepresentativeWarp), laneZero);

    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], write);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = numLanes * numWarps;

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(numLanes);
    Value laneId = b.urem(threadId, warpSize);
    Value zero = b.i32_val(0);

    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = b.icmp_slt(threadId, b.i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                       threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                 threadIsNeeded);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = b.urem(laneId, b.i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          b.icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = b.and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = b.add(readOffset, b.i32_val(numThreads));
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              // When srcShape smaller than src sizePerThread, only srcShape
              // elements is accumulated in smem. Modulo smemShape effectively
              // replicates srcShape elements to src sizePerThread.
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = b.load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = b.load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
