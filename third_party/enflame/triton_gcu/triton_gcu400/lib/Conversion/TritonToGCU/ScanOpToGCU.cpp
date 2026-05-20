/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "Utility.h"

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

// ===----------------------------------------------------------------------===
// Vectorized combiner helpers (ported from GCU 300).
// Vectorize the scalar combine region of tt.scan into SIMD vector ops.
// ===----------------------------------------------------------------------===
SmallVector<Value> vectorizeCombineOpWithoutTerminator(Location loc,
                                                       OpBuilder &builder,
                                                       Region &combineOp,
                                                       ValueRange operands,
                                                       unsigned vectorLength) {
  IRMapping map;
  for (auto [arg, operand] : llvm::zip(combineOp.getArguments(), operands)) {
    map.map(arg, operand);
  }
  for (auto &o : combineOp.back().without_terminator()) {
    for (auto operand : o.getOperands()) {
      if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
        if (!map.lookupOrNull(operand)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(constantOp);
          if (operand.getType().isInteger(1)) {
            auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue());
            auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
            if ((boolAttr && !boolAttr.getValue()) ||
                (integerAttr && integerAttr.getValue().isZero())) {
              map.map(operand,
                      builder.create<vector::ConstantMaskOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          operand.getType()),
                          DenseI64ArrayAttr::get(builder.getContext(),
                                                 ArrayRef<int64_t>{0})));
            } else {
              map.map(
                  operand,
                  builder.create<vector::ConstantMaskOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      operand.getType()),
                      DenseI64ArrayAttr::get(builder.getContext(),
                                             ArrayRef<int64_t>{vectorLength})));
            }
          } else {
            map.map(operand,
                    builder.create<vector::BroadcastOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        operand.getType()),
                        operand));
          }
        }
      }
    }
    Operation *newOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(o)) {
      auto condition = selectOp.getCondition();
      auto mapValue = map.lookup(condition);
      if (cast<VectorType>(mapValue.getType()).getElementType().isInteger(8)) {
        map.map(condition,
                builder
                    .create<gcu::VectorConvertOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        mapValue)
                    .getResult(0));
        newOp = builder.clone(o, map);
        map.map(condition, mapValue);
      } else {
        newOp = builder.clone(o, map);
      }
    } else {
      newOp = builder.clone(o, map);
    }
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(newOp);
    if (typeInterface &&
        succeeded(typeInterface.inferReturnTypes(
            newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
            newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
            newOp->getRegions(), resultTypes))) {
      for (auto [resultType, result, newResult] :
           llvm::zip(resultTypes, o.getResults(), newOp->getResults())) {
        newResult.setType(resultType);
        map.map(result, newResult);
      }
    } else {
      for (auto [result, newResult] :
           llvm::zip(o.getResults(), newOp->getResults())) {
        auto vectorTy =
            VectorType::get(ArrayRef<int64_t>{vectorLength}, result.getType());
        newResult.setType(vectorTy);
        map.map(result, newResult);
      }
    }
  }
  auto terminatorOprands = llvm::to_vector(llvm::map_range(
      llvm::cast<triton::ScanReturnOp>(combineOp.back().getTerminator())
          .getResult(),
      [&](auto v) {
        auto mappingValue = map.lookupOrNull(v);
        assert(mappingValue != nullptr);
        return mappingValue;
      }));
  return terminatorOprands;
}

// Apply combine region to scalar operands (no vectorization).
// operands = [lhs0, ..., lhsN, rhs0, ..., rhsN] matching block args.
SmallVector<Value> scalarCombineOpWithoutTerminator(Location loc,
                                                    OpBuilder &builder,
                                                    Region &combineOp,
                                                    ValueRange operands) {
  IRMapping map;
  for (auto [arg, operand] : llvm::zip(combineOp.getArguments(), operands))
    map.map(arg, operand);
  for (auto &o : combineOp.back().without_terminator()) {
    auto *newOp = builder.clone(o, map);
    for (auto [result, newResult] :
         llvm::zip(o.getResults(), newOp->getResults()))
      map.map(result, newResult);
  }
  auto terminatorOprands = llvm::to_vector(llvm::map_range(
      llvm::cast<triton::ScanReturnOp>(combineOp.back().getTerminator())
          .getResult(),
      [&](auto v) {
        auto mappingValue = map.lookupOrNull(v);
        assert(mappingValue != nullptr);
        return mappingValue;
      }));
  return terminatorOprands;
}

void vectorizeCombineOpTerminator(Location loc, OpBuilder &builder,
                                  ValueRange operands) {
  builder.create<triton::ScanReturnOp>(loc, operands);
}

// ===----------------------------------------------------------------------===
// Shared helpers for TTScanOpLowering and TleExclusiveCumsumLowering.
//
// Decision flow (4-step gate):
//   1. foldTo3DScanShape     - fold per-thread shapes into a 3D scan shape.
//   2. isContiguousLayout    - check that the blocked encoding fully covers
//                              every tensor dimension without holes.
//   3. isDegenerateScan      - check whether the scan axis is too narrow for
//                              efficient multi-SIP parallel execution.
//   4. mustRunOnMasterThread - final gate: true  -> gather onto master thread
//                                                   and scan serially.
//                                          false -> scan in-place on each warp.
// ===----------------------------------------------------------------------===

// Fold per-thread element counts into a 3D shape {dim0, dim1, dim2}, placing
// the scan axis into one of the three dimensions.  Non-scan dimensions are
// collapsed (multiplied) into the remaining slots.
//
//   scanInOutDims[out] will always be 1 for the dimension that is unused.
//   scanAxis          is the index (0-2) where the scan happens.
void foldTo3DScanShape(const SmallVector<unsigned> &numElems, unsigned axis,
                       std::array<int64_t, 3> &scanInOutDims,
                       int64_t &scanAxis) {
  scanInOutDims = {1, 1, 1};
  scanAxis = 2;
  for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
    if (static_cast<unsigned>(i) == axis) {
      if (scanInOutDims[j] == 1)
        scanInOutDims[j] = numElems[i];
      else
        scanInOutDims[--j] = numElems[i];
      scanAxis = j;
      --j;
    } else {
      scanInOutDims[j] *= numElems[i];
    }
  }
}

// Returns true when every dimension satisfies
//   dim == elems_per_thread × threads_per_warp × warps_per_cta
// i.e., the blocked layout is a simple "one value per logical element" mapping
// without holes or redundant replication.
bool isContiguousLayout(RankedTensorType tensorType) {
  auto encodingAttr = tensorType.getEncoding();
  auto warpsPerCTA = triton::gcu::getWarpsPerCTA(encodingAttr);
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(tensorType);
  auto elementsPerThread = triton::gcu::getElemsPerThread(tensorType);
  for (auto [dim, elems, threads, warps] :
       llvm::zip(tensorType.getShape(), elementsPerThread, threadsPerWarp,
                 warpsPerCTA)) {
    if (dim != elems * threads * warps)
      return false;
  }
  return true;
}

// Returns true when the scan axis is too narrow for efficient parallel
// execution across SIPs.  "Too narrow" means the non-scan dimension is smaller
// than the vector length, or the scan dimension itself has fewer than 8
// elements - a heuristic threshold below which serial master-thread execution
// is cheaper than coordinating multiple SIPs.
bool isDegenerateScan(const DenseSet<unsigned> &slicedAxies,
                      const std::array<int64_t, 3> &scanInOutDims,
                      int64_t scanAxis, unsigned vectorLength) {
  if (slicedAxies.empty())
    return false;
  if (scanAxis == 2 &&
      (scanInOutDims[1] < vectorLength || scanInOutDims[2] < 8))
    return true;
  if (scanAxis == 1 &&
      (scanInOutDims[2] < vectorLength || scanInOutDims[1] < 8))
    return true;
  return false;
}

// Decides whether the scan must be serialised on the master thread.
// Returns true when any of these holds:
//   - The scan axis is degenerate (too narrow for parallel execution).
//   - The scan dimension is sliced across multiple SIPs and those SIPs
//     need to coordinate.
//   - The blocked layout is not contiguous (handled safely by serialising).
bool mustRunOnMasterThread(RankedTensorType inputType, unsigned axis,
                           const std::array<int64_t, 3> &scanInOutDims,
                           int64_t scanAxis, unsigned vectorLength) {
  auto slicedAxies = getSlicedAxies(inputType);

  bool isContiguous = isContiguousLayout(inputType);
  bool isScanDimSplit = slicedAxies.count(axis);
  bool isDegenerate =
      isDegenerateScan(slicedAxies, scanInOutDims, scanAxis, vectorLength);
  return isDegenerate || isScanDimSplit || !isContiguous;
}

// Returns true when 1D reshape is applicable.
// On success sets N = total scan elements, M = ceil(N/VL), tail = N % VL.
bool isOneDimReshapeCandidate(const std::array<int64_t, 3> &scanInOutDims,
                              int64_t scanAxis, int64_t VL, int64_t &N,
                              int64_t &M, int64_t &tail) {
  constexpr int64_t minM = 2;
  N = 0;
  if (scanAxis == 2 && scanInOutDims[0] == 1 && scanInOutDims[1] == 1 &&
      scanInOutDims[2] >= VL * minM) {
    N = scanInOutDims[2];
  } else if (scanAxis == 1 && scanInOutDims[0] == 1 && scanInOutDims[2] == 1 &&
             scanInOutDims[1] >= VL * minM) {
    N = scanInOutDims[1];
  } else {
    return false;
  }
  M = N / VL;
  tail = N % VL;
  return true;
}

// ===----------------------------------------------------------------------===
// Common 1D-reshape scan framework.
//
// Reshapes N elements into [numBlocks, VL] (VL = vectorLength).
//  - Iterates M full blocks calling processBlock for intra-block work.
//  - And handles the tail (N % VL) with a scalar fallback.
//
// Inter-block carry is propagated via scf::ForOp iter args.
//
// processBlock(b, loc, vec, carry, isFirstBlock) -> {updatedVec, newCarry}
//   vec          - VL-wide vector loaded from inMemRef
//   carry        - accumulated value from the prev block.
//   isFirstBlock - true  -> the callback is processing the very first block;
//                           inclusive scan skips cross-block carry fusion,
//                           exclusive scan applies the identity element.
//                  false -> normal block with valid cross-block carry.
//
// Returns the final carry (total).
//
// inMemRef and outMemRef may be the same (in‑place inclusive scan) or
// different (exclusive scan with separate src / exclusive buffers).
// ===----------------------------------------------------------------------===
using ProcessBlockFn = std::function<std::pair<Value, Value>(
    OpBuilder &, Location, Value /*vec*/, Value /*carry*/,
    bool /*isFirstBlock*/)>;
using ProcessScalarFn = std::function<std::pair<Value, Value>(
    OpBuilder &, Location, Value /*acc*/, Value /*elem*/)>;

Value applyOneDimReshapeScanCommon(OpBuilder &rewriter, Location loc,
                                   Value inMemRef, Value outMemRef, int64_t N,
                                   int64_t VL, bool reverse, Type elemTy,
                                   Value initVal, ProcessBlockFn processBlock,
                                   ProcessScalarFn combineScalar = nullptr) {
  int64_t M = N / VL;
  int64_t tail = N % VL;
  int64_t numBlocks = M + (tail > 0 ? 1 : 0);
  auto vecTy = VectorType::get(VL, elemTy);

  // - Reshape in / out as [numBlocks, VL]
  auto flatTy = MemRefType::get(ArrayRef<int64_t>{numBlocks, VL}, elemTy);
  Value inFlat = rewriter.create<memref::ReinterpretCastOp>(
      loc, flatTy, inMemRef, ValueRange{}, ValueRange{}, ValueRange{},
      ArrayRef<int64_t>{0}, ArrayRef<int64_t>{numBlocks, VL},
      ArrayRef<int64_t>{VL, 1});

  Value outFlat = inFlat;
  if (inMemRef != outMemRef) {
    outFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatTy, outMemRef, ValueRange{}, ValueRange{}, ValueRange{},
        ArrayRef<int64_t>{0}, ArrayRef<int64_t>{numBlocks, VL},
        ArrayRef<int64_t>{VL, 1});
  }

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto mConst = rewriter.create<arith::ConstantIndexOp>(loc, M);

  Value carry;
  assert(M >= 2 && "M must be at least 2");

  // NOTE: The exclusive (initVal) and inclusive (!initVal) branches are kept
  // separate intentionally.  The exclusive path uses a single unified ForOp
  // over all M blocks (isFirstBlock=false for every iteration), which enables
  // the compiler to unroll the loop body uniformly.  Merging the two branches
  // by pulling block 0 out of the loop (as the inclusive path does) would
  // prevent this unroll optimization and measurably degrade exclusive scan
  // performance.
  if (initVal) {
    // Exclusive scan: single unified ForOp over all M blocks.
    auto mM1Const = rewriter.create<arith::ConstantIndexOp>(loc, M - 1);
    auto forOp = rewriter.create<scf::ForOp>(
        loc, zero, mConst, one, ValueRange{initVal},
        [&](OpBuilder &b, Location loc, Value blkIdx, ValueRange iterArgs) {
          Value blkCarry = iterArgs[0];
          Value blkI =
              reverse ? b.create<arith::SubIOp>(loc, mM1Const, blkIdx) : blkIdx;
          Value vec = b.create<vector::LoadOp>(loc, vecTy, inFlat,
                                               ValueRange{blkI, zero});
          auto [newVec, newCarry] =
              processBlock(b, loc, vec, blkCarry, /*isFirstBlock=*/false);
          b.create<vector::StoreOp>(loc, newVec, outFlat,
                                    ValueRange{blkI, zero});
          vectorizeCombineOpTerminator(loc, b, ValueRange(newCarry));
        });
    carry = forOp.getResult(0);
  } else {
    // Inclusive scan: the first block has no external carry.
    // - the first block
    int64_t firstBlkIdx = reverse ? M - 1 : 0;
    auto blkI = rewriter.create<arith::ConstantIndexOp>(loc, firstBlkIdx);
    Value vec = rewriter.create<vector::LoadOp>(loc, vecTy, inFlat,
                                                ValueRange{blkI, zero});
    auto [newVec, newCarry] =
        processBlock(rewriter, loc, vec, /*carry=*/Value(),
                     /*isFirstBlock=*/true);
    rewriter.create<vector::StoreOp>(loc, newVec, outFlat,
                                     ValueRange{blkI, zero});
    carry = newCarry;

    // - the remaining M-1 blocks, with carry fused in the first lane.
    Value lb = reverse ? zero : one;
    Value ub = reverse ? rewriter.create<arith::ConstantIndexOp>(loc, M - 1)
                       : rewriter.create<arith::ConstantIndexOp>(loc, M);
    auto mMinusTwo = rewriter.create<arith::ConstantIndexOp>(loc, M - 2);
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lb, ub, one, ValueRange{carry},
        [&](OpBuilder &b, Location loc, Value blkIdx, ValueRange iterArgs) {
          Value blkCarry = iterArgs[0];
          Value blkI = reverse ? b.create<arith::SubIOp>(loc, mMinusTwo, blkIdx)
                               : blkIdx;
          Value vec = b.create<vector::LoadOp>(loc, vecTy, inFlat,
                                               ValueRange{blkI, zero});
          auto [newVec, newCarry] =
              processBlock(b, loc, vec, blkCarry, /*isFirstBlock=*/false);
          b.create<vector::StoreOp>(loc, newVec, outFlat,
                                    ValueRange{blkI, zero});
          vectorizeCombineOpTerminator(loc, b, ValueRange(newCarry));
        });
    carry = forOp.getResult(0);
  }

  // - Tail handling: scalar combine via combineScalar callback.
  //   combineScalar returns {storedValue, newCarry} so that both inclusive
  //   and exclusive semantics work correctly.
  if (tail > 0) {
    assert(combineScalar && "tail > 0 requires combineScalar callback");
    auto tailFor = rewriter.create<scf::ForOp>(
        loc, zero, rewriter.create<arith::ConstantIndexOp>(loc, tail), one,
        ValueRange{carry},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          Value acc = iterArgs[0];
          Value rowIdx = b.create<arith::ConstantIndexOp>(loc, M);
          Value colIdx =
              reverse ? b.create<arith::SubIOp>(
                            loc,
                            b.create<arith::ConstantIndexOp>(loc, tail - 1), iv)
                      : iv;
          Value elem =
              b.create<memref::LoadOp>(loc, inFlat, ValueRange{rowIdx, colIdx});
          auto [stored, newAcc] = combineScalar(b, loc, acc, elem);
          b.create<memref::StoreOp>(loc, stored, outFlat,
                                    ValueRange{rowIdx, colIdx});
          vectorizeCombineOpTerminator(loc, b, ValueRange(newAcc));
        });
    return tailFor.getResult(0);
  }

  return carry;
}

struct TTScanOpLowering : SharedConversionPattern<triton::ScanOp> {
  using SharedConversionPattern::SharedConversionPattern;

  void applyScan(triton::ScanOp op, OpBuilder &rewriter,
                 ArrayRef<Value> outputs, ArrayRef<Value> inputs, Type type,
                 bool reverse) const {
    auto axis = op.getAxis();
    auto loc = op.getLoc();
    auto numElems = triton::gcu::getElemsPerThread(type);
    auto numOutput = outputs.size();
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(type);
    auto tag = pTagPool.getPrivateSyncTagInfo(op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // initialize outputs by inputs
    for (unsigned i = 0; i < numOutput; ++i) {
      rewriter.create<memref::DmaStartOp>(
          loc, inputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          outputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems),
          tag.getTag(), ValueRange{tag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
    }

    std::array<int64_t, 3> scanInOutDims = {1, 1, 1};
    int64_t scanAxis = 2;
    for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
      if (static_cast<unsigned>(i) == axis) {
        if (scanInOutDims[j] == 1) {
          scanInOutDims[j] = numElems[i];
        } else {
          scanInOutDims[--j] = numElems[i];
        }
        scanAxis = j;
        --j;
      } else {
        scanInOutDims[j] *= numElems[i];
      }
    }
    SmallVector<Value, 4> outs;
    llvm::transform(outputs, std::back_inserter(outs), [&](auto output) {
      return rewriter.create<memref::ReinterpretCastOp>(
          loc,
          MemRefType::get(scanInOutDims,
                          cast<MemRefType>(output.getType()).getElementType()),
          output, ValueRange{}, ValueRange{}, ValueRange{},
          ArrayRef<int64_t>{0},
          ArrayRef<int64_t>{scanInOutDims[0], scanInOutDims[1],
                            scanInOutDims[2]},
          ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                            scanInOutDims[2], 1});
    });
    // 1D specialized path: when the scan axis occupies the only "large"
    // dimension and the vectorize axis is equal to 1, reshape so the scan axis
    // becomes dim1 and the vectorize axis becomes dim2. After scan, apply lane
    // postfix to combine independent per-lane scans.
    if (succeeded(applyOneDimReshapeScan(op, rewriter, outs, scanInOutDims,
                                         scanAxis, reverse))) {
      return;
    }
    if (succeeded(applyGeneralScan(op, rewriter, outs, scanInOutDims, scanAxis,
                                   reverse))) {
      return;
    }
    return applyScanFallback(op, rewriter, outs, scanInOutDims, scanAxis,
                             reverse);
  }

  LogicalResult
  applyOneDimReshapeScan(triton::ScanOp op, OpBuilder &rewriter,
                         ArrayRef<Value> outputs,
                         const std::array<int64_t, 3> &scanInOutDims,
                         int64_t scanAxis, bool reverse) const {
    unsigned maxBpe = 4;
    unsigned minBpe = 4;
    auto numOutput = outputs.size();
    for (auto output : outputs) {
      auto elTy = cast<MemRefType>(output.getType()).getElementType();
      auto bpe = mlir::triton::gcu::getBpe(elTy);
      maxBpe = std::max(maxBpe, bpe);
      minBpe = std::min(minBpe, bpe);
    }
    if (maxBpe < 4)
      maxBpe = 4;
    int64_t VL = oaccSizeInBytes / minBpe;

    int64_t N = 0, M = 0, tail = 0;
    if (!isOneDimReshapeCandidate(scanInOutDims, scanAxis, VL, N, M, tail))
      return failure();
    if (numOutput != 1)
      return failure();

    auto loc = op.getLoc();
    auto &combineOp = op.getCombineOp();
    auto elTy = cast<MemRefType>(outputs[0].getType()).getElementType();

    // Check if the combine op is add — enables Hillis-Steele O(log2(VL)).
    bool isAddOp = false;
    if (combineOp.hasOneBlock()) {
      auto &block = combineOp.front();
      if (llvm::hasSingleElement(block.without_terminator())) {
        auto &frontOp = block.front();
        isAddOp = isa<arith::AddIOp>(frontOp) || isa<arith::AddFOp>(frontOp);
      }
    }

    ProcessBlockFn inclusiveScan;
    if (isAddOp) {
      // Hillis-Steele O(log2(VL)) inclusive scan for add combine.
      // Phase 1: fuse cross-block carry into firstLane (scalar insert).
      // Phase 2: log2(VL) steps of vector_shift + vectorized combine.
      inclusiveScan = [&](OpBuilder &b, Location loc, Value vec, Value carry,
                          bool isFirstBlock) {
        auto shiftDir = reverse ? gcu::VectorShiftDirection::LEFT
                                : gcu::VectorShiftDirection::RIGHT;

        if (!isFirstBlock) {
          int64_t firstLane = reverse ? VL - 1 : 0;
          Value firstElem = b.create<vector::ExtractOp>(
              loc, vec, ArrayRef<int64_t>{firstLane});
          SmallVector<Value, 2> args = {carry, firstElem};
          auto res = scalarCombineOpWithoutTerminator(loc, b, combineOp, args);
          vec = b.create<vector::InsertOp>(loc, res[0], vec,
                                           ArrayRef<int64_t>{firstLane});
        }

        for (int64_t stride = 1; stride < VL; stride *= 2) {
          Value shiftVal =
              b.create<arith::ConstantIntOp>(loc, b.getI32Type(), stride);
          Value shifted =
              b.create<gcu::VectorShiftOp>(loc, shiftDir, vec, shiftVal);
          auto res = vectorizeCombineOpWithoutTerminator(
              loc, b, combineOp, ValueRange{shifted, vec}, VL);
          vec = res[0];
        }

        int64_t lastLane = reverse ? 0 : VL - 1;
        Value acc =
            b.create<vector::ExtractOp>(loc, vec, ArrayRef<int64_t>{lastLane});
        return std::make_pair(vec, acc);
      };
    } else {
      // Serial O(VL) inclusive scan for non-add combine ops.
      // Phase 1: fuse cross-block carry into firstLane (or use firstElem
      //          directly when isFirstBlock -- correct for any monoid).
      // Phase 2: chain-combine lanes 1..VL-1 with the running accumulator.
      inclusiveScan = [&](OpBuilder &b, Location loc, Value vec, Value carry,
                          bool isFirstBlock) {
        Value acc;
        int64_t firstLane = reverse ? VL - 1 : 0;
        Value firstElem =
            b.create<vector::ExtractOp>(loc, vec, ArrayRef<int64_t>{firstLane});
        if (isFirstBlock) {
          acc = firstElem;
        } else {
          SmallVector<Value, 2> args = {carry, firstElem};
          auto res = scalarCombineOpWithoutTerminator(loc, b, combineOp, args);
          acc = res[0];
          vec = b.create<vector::InsertOp>(loc, acc, vec,
                                           ArrayRef<int64_t>{firstLane});
        }

        for (int64_t k = 1; k < VL; ++k) {
          int64_t lane = reverse ? VL - 1 - k : k;
          Value elem =
              b.create<vector::ExtractOp>(loc, vec, ArrayRef<int64_t>{lane});
          SmallVector<Value, 2> args = {acc, elem};
          auto res = scalarCombineOpWithoutTerminator(loc, b, combineOp, args);
          acc = res[0];
          vec = b.create<vector::InsertOp>(loc, acc, vec,
                                           ArrayRef<int64_t>{lane});
        }
        return std::make_pair(vec, acc);
      };
    }

    // Scalar combine for tail elements: inclusive -> {newAcc, newAcc}.
    ProcessScalarFn scalarCombine = [&](OpBuilder &b, Location loc, Value acc,
                                        Value elem) {
      SmallVector<Value, 2> args = {acc, elem};
      auto res = scalarCombineOpWithoutTerminator(loc, b, combineOp, args);
      return std::make_pair(res[0], res[0]);
    };

    // Inclusive scan: the first block has no external carry.  The framework
    // passes isFirstBlock=true, so the callback ignores the initVal entirely
    // (first lane untouched — correct for any monoid).  initVal is only used
    // as a syntactic placeholder to satisfy the ProcessBlockFn signature.
    applyOneDimReshapeScanCommon(rewriter, loc, outputs[0], outputs[0], N, VL,
                                 reverse, elTy, Value(), inclusiveScan,
                                 scalarCombine);
    return success();
  }

  LogicalResult applyGeneralScan(triton::ScanOp op, OpBuilder &rewriter,
                                 ArrayRef<Value> outputs,
                                 const std::array<int64_t, 3> &scanInOutDims,
                                 int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    int64_t vectorizeAxis;
    if (scanAxis == 2) {
      assert(scanInOutDims[0] == 1);
      vectorizeAxis = 1;
    } else {
      assert(scanAxis == 1);
      vectorizeAxis = scanInOutDims[0] > scanInOutDims[2] ? 0 : 2;
    }

    unsigned maxBpe = 4;
    unsigned minBpe = 4;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() && !elementType.isInteger(64)) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    // for vector step i32
    if (maxBpe < 4) {
      maxBpe = 4;
    }
    auto numOacc = maxBpe / minBpe;
    if (numOacc > 4) {
      return failure();
    }

    unsigned vectorLength = oaccSizeInBytes / minBpe;
    if (scanInOutDims[vectorizeAxis] < vectorLength) {
      return failure();
    }
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<VectorType, 4> vectorTypes;
    llvm::transform(
        outputs, std::back_inserter(vectorTypes), [vectorLength](auto output) {
          auto elementTy = cast<MemRefType>(output.getType()).getElementType();
          return VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        });

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    std::array<int64_t, 3> loopcnt = scanInOutDims;
    if (loopcnt[vectorizeAxis] % vectorLength != 0) {
      llvm_unreachable("invalid datalayout");
    }
    loopcnt[vectorizeAxis] /= vectorLength;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[2])};
    SmallVector<Value, 4> step(scanInOutDims.size(), one);

    auto maskType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI1Type());
    Value mask = rewriter.create<vector::ConstantMaskOp>(
        loc, maskType,
        DenseI64ArrayAttr::get(rewriter.getContext(),
                               ArrayRef<int64_t>{vectorLength}));
    unsigned strideOnVectorizeAxis =
        std::accumulate(scanInOutDims.begin() + vectorizeAxis + 1,
                        scanInOutDims.end(), 1, std::multiplies<unsigned>());
    auto vecTy =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI32Type());
    auto indexVec = rewriter.create<arith::MulIOp>(
        loc,
        rewriter
            .create<gcu::VectorStepOp>(
                loc, vecTy, rewriter.create<arith::ConstantIntOp>(loc, 0, 32))
            .getResult(),
        rewriter.create<vector::BroadcastOp>(
            loc, vecTy,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(strideOnVectorizeAxis))));

    SmallVector<Value, 4> passThruValues;
    for (unsigned i = 0; i < numOutput; ++i) {
      passThruValues.push_back(rewriter.create<vector::BroadcastOp>(
          loc, vectorTypes[i],
          rewriter.create<arith::ConstantOp>(
              loc, vectorTypes[i].getElementType(),
              rewriter.getZeroAttr(vectorTypes[i].getElementType()))));
    }

    scf::buildLoopNest(
        rewriter, loc,
        ArrayRef<Value>(lbs.begin(), lbs.begin() + vectorizeAxis),
        ArrayRef<Value>(ubs.begin(), ubs.begin() + vectorizeAxis),
        ArrayRef<Value>(step.begin(), step.begin() + vectorizeAxis),
        [&](OpBuilder &builder, Location loc, ValueRange outerIters) {
          scf::buildLoopNest(
              rewriter, loc,
              ArrayRef<Value>(lbs.begin() + vectorizeAxis, lbs.end()),
              ArrayRef<Value>(ubs.begin() + vectorizeAxis, ubs.end()),
              ArrayRef<Value>(step.begin() + vectorizeAxis, step.end()),
              [&](OpBuilder &builder, Location loc, ValueRange innerIters) {
                SmallVector<Value, 4> inputIndices;
                SmallVector<Value, 4> outputIndices;

                SmallVector<Type, 4> resultElemTypes;
                SmallVector<Value, 4> operands;
                SmallVector<Value, 4> ivs;
                for (auto iv : outerIters) {
                  ivs.push_back(iv);
                }
                for (auto iv : innerIters) {
                  ivs.push_back(iv);
                }
                if (reverse) {
                  ivs[scanAxis] = builder.create<arith::SubIOp>(
                      loc,
                      builder.create<arith::ConstantIndexOp>(
                          loc, scanInOutDims[scanAxis] - 1),
                      ivs[scanAxis]);
                }
                for (unsigned i = 0; i < ivs.size(); ++i) {
                  if (i == vectorizeAxis) {
                    outputIndices.push_back(builder.create<arith::MulIOp>(
                        loc, ivs[i],
                        rewriter.create<arith::ConstantIndexOp>(loc,
                                                                vectorLength)));
                  } else {
                    outputIndices.push_back(ivs[i]);
                  }
                  if (i == scanAxis) {
                    if (reverse) {
                      inputIndices.push_back(builder.create<arith::AddIOp>(
                          loc, outputIndices[i], one));
                    } else {
                      inputIndices.push_back(builder.create<arith::SubIOp>(
                          loc, outputIndices[i], one));
                    }
                  } else {
                    inputIndices.push_back(outputIndices[i]);
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], inputIndices, indexVec,
                      mask, passThruValues[i]));
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], outputIndices, indexVec,
                      mask, passThruValues[i]));
                  resultElemTypes.push_back(vectorTypes[i]);
                }

                auto executeRegionOp =
                    builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
                executeRegionOp.getRegion().emplaceBlock();
                IRMapping map;
                for (auto [arg, operand] :
                     llvm::zip(op.getCombineOp().getArguments(), operands)) {
                  map.map(arg, operand);
                }
                {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointToStart(
                      &executeRegionOp.getRegion().back());
                  for (auto &o : op.getCombineOp().back()) {
                    for (auto operand : o.getOperands()) {
                      if (auto constantOp =
                              operand.getDefiningOp<arith::ConstantOp>()) {
                        if (!map.lookupOrNull(operand)) {
                          OpBuilder::InsertionGuard guard(builder);
                          builder.setInsertionPointAfter(constantOp);
                          map.map(operand,
                                  builder.create<vector::BroadcastOp>(
                                      loc,
                                      VectorType::get(
                                          ArrayRef<int64_t>{vectorLength},
                                          operand.getType()),
                                      operand));
                        }
                      }
                    }
                    auto newO = builder.clone(o, map);
                    for (auto [result, newResult] :
                         llvm::zip(o.getResults(), newO->getResults())) {
                      auto vectorTy = VectorType::get(
                          ArrayRef<int64_t>{vectorLength}, result.getType());
                      newResult.setType(vectorTy);
                      map.map(result, newResult);
                    }
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  builder.create<vector::ScatterOp>(
                      loc, outputs[i], outputIndices, indexVec, mask,
                      executeRegionOp.getResult(i));
                }
              });
        });
    return success();
  }

  void applyScanFallback(triton::ScanOp op, OpBuilder &rewriter,
                         ArrayRef<Value> outputs,
                         const std::array<int64_t, 3> &scanInOutDims,
                         int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[2])};

    scf::buildLoopNest(
        rewriter, loc, lbs, ubs,
        SmallVector<Value, 4>(scanInOutDims.size(), one),
        [&](OpBuilder &builder, Location loc, ValueRange iters) {
          SmallVector<Value, 4> outputIters(iters.begin(), iters.end());
          if (reverse) {
            outputIters[scanAxis] = builder.create<arith::SubIOp>(
                loc,
                builder.create<arith::ConstantIndexOp>(
                    loc, scanInOutDims[scanAxis] - 1),
                outputIters[scanAxis]);
          }

          SmallVector<Value, 4> operands;
          SmallVector<Type, 4> resultElemTypes;
          SmallVector<Value, 4> inputIters(outputIters.begin(),
                                           outputIters.end());
          if (reverse) {
            inputIters[scanAxis] =
                builder.create<arith::AddIOp>(loc, one, inputIters[scanAxis]);
          } else {
            inputIters[scanAxis] =
                builder.create<arith::SubIOp>(loc, inputIters[scanAxis], one);
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], inputIters));
          }
          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], outputIters));
            resultElemTypes.push_back(operands.back().getType());
          }

          auto executeRegion =
              builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
          executeRegion.getRegion().emplaceBlock();
          IRMapping map;
          for (auto [arg, operand] :
               llvm::zip(op.getCombineOp().getArguments(), operands)) {
            map.map(arg, operand);
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&executeRegion.getRegion().back());
            for (auto &o : op.getCombineOp().back()) {
              auto newO = builder.clone(o, map);
              for (auto [result, newResult] :
                   llvm::zip(o.getResults(), newO->getResults())) {
                map.map(result, newResult);
              }
            }
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            builder.create<memref::StoreOp>(loc, executeRegion.getResult(i),
                                            outputs[i], outputIters);
          }
        });

    doMemFence(rewriter, op);
  }

  // Block scan with loop unrolling (ported from GCU 300).
  // Uses vector::LoadOp/StoreOp for contiguous memory access instead of
  // gather/scatter, and applies a Blelloch-style blocked prefix sum.
  void applyVectorizationImpl(triton::ScanOp op, OpBuilder &rewriter,
                              ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                              const std::array<int64_t, 3> &scanInOutDims,
                              int64_t scanAxis, unsigned vectorLength,
                              bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    assert(scanAxis == 1);
    auto loopLimit = scanInOutDims[1];
    auto vLength = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    constexpr int loopUnrollTime = 16;
    auto loopCnt = loopLimit > loopUnrollTime ? loopUnrollTime : loopLimit;
    auto loopCntValue = rewriter.create<arith::ConstantIndexOp>(loc, loopCnt);
    SmallVector<VectorType, 4> vectorTypes;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      vectorTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementType));
    }
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto &combineOp = op.getCombineOp();
    rewriter.create<scf::ForOp>(
        loc, zero,
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[0]), one,
        ValueRange{},
        [&](OpBuilder &builder, Location loc, Value iter0,
            ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zero,
              builder.create<arith::ConstantIndexOp>(loc, scanInOutDims[2]),
              vLength, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iter2,
                  ValueRange iterArgs) {
                SmallVector<Value, loopUnrollTime> vectorList;
                for (unsigned i = 0; i < loopCnt; ++i) {
                  Value scanIndex;
                  if (!reverse) {
                    scanIndex = builder.create<arith::ConstantIndexOp>(loc, i);
                  } else {
                    scanIndex = builder.create<arith::ConstantIndexOp>(
                        loc, scanInOutDims[1] - 1 - i);
                  }
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList.emplace_back(builder.create<vector::LoadOp>(
                        loc, vectorTypes[j], inputs[j],
                        ValueRange{iter0, scanIndex, iter2}));
                  }
                }
                // Pair scan within the first block
                SmallVector<Value, 4> args(numOutput * 2);
                for (unsigned i = 0; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[(i * 2) * numOutput + j];
                    args[numOutput + j] =
                        vectorList[(i * 2 + 1) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2 + 1) * numOutput + j] = combinOut[j];
                  }
                }
                // Cross-pair accumulation
                for (unsigned i = 1; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[((i - 1) * 2 + 1) * numOutput + j];
                    args[numOutput + j] =
                        vectorList[(i * 2 + 1) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2 + 1) * numOutput + j] = combinOut[j];
                  }
                }
                // Update even positions
                for (unsigned i = 1; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[((i - 1) * 2 + 1) * numOutput + j];
                    args[numOutput + j] = vectorList[(i * 2) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2) * numOutput + j] = combinOut[j];
                  }
                }
                // Store first block results
                for (unsigned i = 0; i < loopCnt; ++i) {
                  Value scanIndex;
                  if (!reverse) {
                    scanIndex = builder.create<arith::ConstantIndexOp>(loc, i);
                  } else {
                    scanIndex = builder.create<arith::ConstantIndexOp>(
                        loc, scanInOutDims[1] - 1 - i);
                  }
                  for (unsigned j = 0; j < numOutput; ++j) {
                    builder.create<vector::StoreOp>(
                        loc, vectorList[i * numOutput + j], outputs[j],
                        ValueRange{iter0, scanIndex, iter2});
                  }
                }
                SmallVector<Value, 4> blockOut(numOutput);
                for (unsigned j = 0; j < numOutput; ++j) {
                  blockOut[j] = vectorList[(loopCnt - 1) * numOutput + j];
                }
                // Subsequent blocks: carry over last result from previous block
                builder.create<scf::ForOp>(
                    loc, loopCntValue,
                    builder.create<arith::ConstantIndexOp>(loc,
                                                           scanInOutDims[1]),
                    loopCntValue, blockOut,
                    [&](OpBuilder &builder, Location loc, Value iter1,
                        ValueRange lastBlockOut) {
                      SmallVector<Value, loopUnrollTime> vectorListInner;
                      for (unsigned i = 0; i < loopCnt; ++i) {
                        Value scanIndex = builder.create<arith::AddIOp>(
                            loc, builder.create<arith::ConstantIndexOp>(loc, i),
                            iter1);
                        if (reverse) {
                          scanIndex = builder.create<arith::SubIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, scanInOutDims[1] - 1),
                              scanIndex);
                        }
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner.emplace_back(
                              builder.create<vector::LoadOp>(
                                  loc, vectorTypes[j], inputs[j],
                                  ValueRange{iter0, scanIndex, iter2}));
                        }
                      }
                      SmallVector<Value, 4> args(numOutput * 2);
                      // Pair scan
                      for (unsigned i = 0; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] = vectorListInner[(i * 2) * numOutput + j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2 + 1) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2 + 1) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      // Combine with last block's tail
                      for (unsigned j = 0; j < numOutput; ++j) {
                        args[j] = lastBlockOut[j];
                        args[numOutput + j] =
                            vectorListInner[(1) * numOutput + j];
                      }
                      auto combinOut = vectorizeCombineOpWithoutTerminator(
                          loc, builder, combineOp, args, vectorLength);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        vectorListInner[(1) * numOutput + j] = combinOut[j];
                      }
                      // Cross-pair accumulation
                      for (unsigned i = 1; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] =
                              vectorListInner[((i - 1) * 2 + 1) * numOutput +
                                              j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2 + 1) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2 + 1) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      // Update even positions (carry from previous block)
                      for (unsigned j = 0; j < numOutput; ++j) {
                        args[j] = lastBlockOut[j];
                        args[numOutput + j] = vectorListInner[j];
                      }
                      combinOut = vectorizeCombineOpWithoutTerminator(
                          loc, builder, combineOp, args, vectorLength);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        vectorListInner[j] = combinOut[j];
                      }
                      for (unsigned i = 1; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] =
                              vectorListInner[((i - 1) * 2 + 1) * numOutput +
                                              j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      // Store block results
                      for (unsigned i = 0; i < loopCnt; ++i) {
                        Value scanIndex = builder.create<arith::AddIOp>(
                            loc, builder.create<arith::ConstantIndexOp>(loc, i),
                            iter1);
                        if (reverse) {
                          scanIndex = builder.create<arith::SubIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, scanInOutDims[1] - 1),
                              scanIndex);
                        }
                        for (unsigned j = 0; j < numOutput; ++j) {
                          builder.create<vector::StoreOp>(
                              loc, vectorListInner[i * numOutput + j],
                              outputs[j], ValueRange{iter0, scanIndex, iter2});
                        }
                      }
                      SmallVector<Value, 4> blockOut(numOutput);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        blockOut[j] =
                            vectorListInner[(loopCnt - 1) * numOutput + j];
                      }
                      vectorizeCombineOpTerminator(loc, builder, blockOut);
                    });
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  }

  // Entry point for vectorized scan (ported from GCU 300).
  // Handles axis==1 directly, axis==2 via transpose->scan->transpose-back.
  LogicalResult applyVectorizationScan(triton::ScanOp op, OpBuilder &rewriter,
                                       ArrayRef<Value> outputs,
                                       ArrayRef<Value> inputs,
                                       std::array<int64_t, 3> scanInOutDims,
                                       int64_t scanAxis, unsigned vectorLength,
                                       bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = op.getResults().size();
    unsigned maxBpe = 1;
    unsigned minBpe = 4;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32()) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    auto numVacc = maxBpe / minBpe;
    if (numVacc > 4) {
      return failure();
    }
    if (scanAxis == 1 && scanInOutDims[2] >= vectorLength &&
        scanInOutDims[1] >= 8) {
      SmallVector<Value, 4> sanInputs;
      SmallVector<Value, 4> sanOutputs;
      llvm::transform(inputs, std::back_inserter(sanInputs), [&](auto input) {
        return rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(scanInOutDims,
                            cast<MemRefType>(input.getType()).getElementType()),
            input, ValueRange{}, ValueRange{}, ValueRange{},
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
            ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                              scanInOutDims[2], 1});
      });
      llvm::transform(
          outputs, std::back_inserter(sanOutputs), [&](auto output) {
            return rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(
                    scanInOutDims,
                    cast<MemRefType>(output.getType()).getElementType()),
                output, ValueRange{}, ValueRange{}, ValueRange{},
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
                ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                                  scanInOutDims[2], 1});
          });
      applyVectorizationImpl(op, rewriter, sanOutputs, sanInputs, scanInOutDims,
                             scanAxis, vectorLength, op.getReverse());
      return success();
    } else if (scanAxis == 2 && scanInOutDims[1] >= vectorLength &&
               scanInOutDims[2] >= 8) {
      // Transpose dim1↔dim2, scan on axis==1, then transpose back.
      SmallVector<Value, 4> sanInputs;
      SmallVector<Value, 4> sanOutputs;
      llvm::transform(inputs, std::back_inserter(sanInputs), [&](auto input) {
        return rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(scanInOutDims,
                            cast<MemRefType>(input.getType()).getElementType()),
            input, ValueRange{}, ValueRange{}, ValueRange{},
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
            ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                              scanInOutDims[2], 1});
      });
      llvm::transform(
          outputs, std::back_inserter(sanOutputs), [&](auto output) {
            return rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(
                    scanInOutDims,
                    cast<MemRefType>(output.getType()).getElementType()),
                output, ValueRange{}, ValueRange{}, ValueRange{},
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
                ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                                  scanInOutDims[2], 1});
          });
      std::array<int64_t, 3> transposeLayout = {0, 2, 1};
      auto tag = pTagPool.getPrivateSyncTagInfo(op);
      int64_t dim0 = scanInOutDims[transposeLayout[0]];
      int64_t dim1 = scanInOutDims[transposeLayout[1]];
      int64_t dim2 = scanInOutDims[transposeLayout[2]];
      scanInOutDims[0] = dim0;
      scanInOutDims[1] = dim1;
      scanInOutDims[2] = dim2;
      scanAxis = 1; // after transpose, scan axis moves to 1
      SmallVector<Value, 3> transposeLayoutValue;
      llvm::transform(transposeLayout, std::back_inserter(transposeLayoutValue),
                      [&](auto dim) {
                        return rewriter.create<arith::ConstantIntOp>(
                            loc, rewriter.getI32Type(), dim);
                      });
      SmallVector<Value, 2> tmpBuffers;
      llvm::transform(
          sanInputs, std::back_inserter(tmpBuffers), [&](auto input) {
            auto memrefTy = cast<MemRefType>(input.getType());
            auto elementTy = memrefTy.getElementType();
            auto tmpBuffer = rewriter.create<memref::AllocOp>(
                loc,
                MemRefType::get(ArrayRef<int64_t>{scanInOutDims}, elementTy));
            rewriter.create<memref_ext::TransposeStartOp>(
                loc, tmpBuffer, input, transposeLayoutValue, tag.getTag(),
                ValueRange{tag.getIdx()});
            rewriter.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()},
                rewriter.create<arith::ConstantIndexOp>(
                    loc, memrefTy.getNumElements()));
            return tmpBuffer;
          });
      SmallVector<Value, 4> sanTransOutputs(tmpBuffers.begin(),
                                            tmpBuffers.end());
      sanInputs = SmallVector<Value, 4>(tmpBuffers.begin(), tmpBuffers.end());
      applyVectorizationImpl(op, rewriter, sanTransOutputs, sanInputs,
                             scanInOutDims, scanAxis, vectorLength,
                             op.getReverse());
      // Transpose back
      std::array<int64_t, 3> transposeOutLayout = {0, 2, 1};
      dim0 = scanInOutDims[transposeOutLayout[0]];
      dim1 = scanInOutDims[transposeOutLayout[1]];
      dim2 = scanInOutDims[transposeOutLayout[2]];
      scanInOutDims[0] = dim0;
      scanInOutDims[1] = dim1;
      scanInOutDims[2] = dim2;
      SmallVector<Value, 3> transposeOutLayoutValue;
      llvm::transform(transposeOutLayout,
                      std::back_inserter(transposeOutLayoutValue),
                      [&](auto dim) {
                        return rewriter.create<arith::ConstantIntOp>(
                            loc, rewriter.getI32Type(), dim);
                      });
      for (unsigned i = 0; i < numOutput; ++i) {
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, sanOutputs[i], sanTransOutputs[i], transposeOutLayoutValue,
            tag.getTag(), ValueRange{tag.getIdx()});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag.getTag(), ValueRange{tag.getIdx()},
            rewriter.create<arith::ConstantIndexOp>(
                loc, scanInOutDims[0] * scanInOutDims[1] * scanInOutDims[2]));
      }
      for (auto buffer : tmpBuffers) {
        rewriter.create<memref::DeallocOp>(loc, buffer);
      }
      return success();
    }
    return failure();
  }

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto inputType = dyn_cast<RankedTensorType>(op.getSrcs()[0].getType());

    auto numInput = op.getSrcs().size();
    auto numOutput = op.getResults().size();

    // create outputs
    SmallVector<Value, 4> outputs;
    SmallVector<Type, 4> outputElemTypes;
    SmallVector<std::pair<Operation *, int>, 4> lastUsers;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto resultType =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType(i)));
      auto elemType = resultType.getElementType();
      outputElemTypes.push_back(elemType);
      auto lastUser = userAnalysis.getLastUser(op.getResults()[i]);
      Value output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                 replaced2Origin, resultType);
      outputs.push_back(output);
      lastUsers.push_back(lastUser);
    }

    unsigned minBpe = 4;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto bpe = mlir::triton::gcu::getBpe(outputElemTypes[i]);
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    unsigned vectorLength = oaccSizeInBytes / minBpe;

    auto numElems = triton::gcu::getElemsPerThread(inputType);
    auto axis = op.getAxis();
    std::array<int64_t, 3> scanInOutDims;
    int64_t scanAxis;
    foldTo3DScanShape(numElems, axis, scanInOutDims, scanAxis);

    if (mustRunOnMasterThread(inputType, axis, scanInOutDims, scanAxis,
                              vectorLength)) {
      auto tag = pTagPool.getPrivateSyncTagInfo(op);

      // move to shared memory
      SmallVector<Value, 4> sharedInputs;
      for (unsigned i = 0; i < numInput; ++i) {
        sharedInputs.push_back(storeToSharedMem(
            rewriter, tag,
            dyn_cast<RankedTensorType>(op.getSrcs()[i].getType()),
            adaptor.getSrcs()[i], false, std::make_pair(op.getOperation(), -1),
            userAnalysis, replaced2Origin));
      }

      // load all shared memory to thread 0
      SmallVector<Value, 4> mergedInputs;
      RankedTensorType mergedInputType;
      for (unsigned i = 0; i < numInput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getSrcs()[i].getType());
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedInputType = tensorType;
        mergedInputs.push_back(loadFromSharedMem(
            rewriter, tag, tensorType, sharedInputs[i], true,
            std::make_pair(op.getOperation(), -1), std::make_pair(nullptr, -1),
            userAnalysis, replaced2Origin));
      }

      SmallVector<Value, 4> mergedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        auto resultType =
            dyn_cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        mergedOutputs.push_back(
            syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                        userAnalysis, replaced2Origin, resultType));
      }

      // computing in thread 0
      auto masterWarpId = getMasterThreadId(op.getOperation());
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            std::array<int64_t, 3> mergeInOutDims;
            int64_t mergeScanAxis;
            foldTo3DScanShape(
                SmallVector<unsigned>(mergedInputType.getShape().begin(),
                                      mergedInputType.getShape().end()),
                axis, mergeInOutDims, mergeScanAxis);
            // Try vectorized block scan first, fallback to original applyScan.
            if (!succeeded(applyVectorizationScan(
                    op, builder, mergedOutputs, mergedInputs, mergeInOutDims,
                    mergeScanAxis, vectorLength, op.getReverse()))) {
              applyScan(op, builder, mergedOutputs, mergedInputs,
                        mergedInputType, op.getReverse());
            }
            builder.create<scf::YieldOp>(loc);
          });

      // save back to shared memory
      SmallVector<Value, 4> mergedSharedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), outputElemTypes[i],
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedSharedOutputs.push_back(
            storeToSharedMem(rewriter, tag, tensorType, mergedOutputs[i], true,
                             std::make_pair(op.getOperation(), -1),
                             userAnalysis, replaced2Origin));
      }
      // load from shared memory
      for (unsigned i = 0; i < numOutput; ++i) {
        outputs[i] = loadFromSharedMem(
            rewriter, tag, op.getResultTypes()[i], mergedSharedOutputs[i],
            false, lastUsers[i], std::make_pair(nullptr, -1), userAnalysis,
            replaced2Origin);
      }
    } else {
      // Warp-local path: try vectorized block scan first.
      if (!succeeded(applyVectorizationScan(
              op, rewriter, outputs,
              SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                    adaptor.getSrcs().end()),
              scanInOutDims, scanAxis, vectorLength, op.getReverse()))) {
        applyScan(op, rewriter, outputs,
                  SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                        adaptor.getSrcs().end()),
                  inputType, op.getReverse());
      }
    }

    SmallVector<Value, 4> finalOutputs;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto output = outputs[i];
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(op.getResultTypes()[i]));
      if (resultType.getNumElements() !=
          dyn_cast<MemRefType>(output.getType()).getNumElements()) {
        return op.emitOpError("element number mismatch");
      }
      auto [strides, offset] = resultType.getStridesAndOffset();
      output = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, output, offset, resultType.getShape(), strides);
      finalOutputs.push_back(output);
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// tle.exclusive_cumsum -> GCU lowering.
// ===----------------------------------------------------------------------===

struct TleExclusiveCumsumLowering : SharedGenericConversionPattern {
  TleExclusiveCumsumLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateTagPool &pTagPool)
      : SharedGenericConversionPattern("tle.exclusive_cumsum", converter, ctx,
                                       userAnalysis, replaced2Origin,
                                       pTagPool) {}

private:
  static Value addCombine(OpBuilder &b, Location loc, Value lhs, Value rhs,
                          Type elemTy) {
    if (isa<FloatType>(elemTy))
      return b.create<arith::AddFOp>(loc, lhs, rhs);
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  }

public:
  // === Scan execution strategies ==========================================
  // Contract: given srcInput (memref with totalNumElems elements) and
  // exclusiveOut (same shape), produce the total scalar value.
  //
  // To add a new strategy (vectorised block scan, 2D reshape, etc.):
  //   1. Implement a method with this exact signature.
  //   2. Select it in matchAndRewrite based on element type / shape heuristics.
  //   3. Validate with python/test/tle/unit/test_tle_cumsum.py.
  // ========================================================================

  // Strategy 0: scalar sequential scan, O(N).  Baseline, always correct.
  Value runScalarExclusiveScan(Operation *op, OpBuilder &rewriter,
                               Value srcInput, Value exclusiveOut,
                               int64_t totalNumElems, bool reverse,
                               Type elemTy) const {
    auto loc = op->getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto flatRank = MemRefType::get({totalNumElems}, elemTy);
    auto srcFlat1d = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatRank, srcInput, ValueRange{}, ValueRange{}, ValueRange{},
        ArrayRef<int64_t>{0}, ArrayRef<int64_t>{totalNumElems},
        ArrayRef<int64_t>{1});
    auto excFlat1d = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatRank, exclusiveOut, ValueRange{}, ValueRange{}, ValueRange{},
        ArrayRef<int64_t>{0}, ArrayRef<int64_t>{totalNumElems},
        ArrayRef<int64_t>{1});

    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, elemTy, rewriter.getZeroAttr(elemTy));

    auto totalElemsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);

    if (!reverse) {
      // Forward exclusive scan with carried accumulator.
      // acc starts at 0; each iteration: exc[i]=acc, acc+=src[i].
      // Total = final acc after the loop (= sum of all src).
      auto forOp = rewriter.create<scf::ForOp>(
          loc, zero, totalElemsVal, one, ValueRange{zeroVal},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value acc = iterArgs[0];
            b.create<memref::StoreOp>(loc, acc, excFlat1d, ValueRange{iv});
            Value srcVal =
                b.create<memref::LoadOp>(loc, srcFlat1d, ValueRange{iv});
            Value newAcc = addCombine(b, loc, acc, srcVal, elemTy);
            vectorizeCombineOpTerminator(loc, b, ValueRange(newAcc));
          });
      return forOp.getResult(0);

    } else {
      // Reverse exclusive scan with carried accumulator.
      // acc starts at 0; iterate from N-1 down to 0:
      //   exc[i]=acc, acc+=src[i].
      auto forOp = rewriter.create<scf::ForOp>(
          loc, zero, totalElemsVal, one, ValueRange{zeroVal},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value acc = iterArgs[0];
            Value actualIdx = b.create<arith::SubIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, totalNumElems - 1),
                iv);
            b.create<memref::StoreOp>(loc, acc, excFlat1d,
                                      ValueRange{actualIdx});
            Value srcVal =
                b.create<memref::LoadOp>(loc, srcFlat1d, ValueRange{actualIdx});
            Value newAcc = addCombine(b, loc, acc, srcVal, elemTy);
            vectorizeCombineOpTerminator(loc, b, ValueRange(newAcc));
          });
      return forOp.getResult(0);
    }
  }

  // ===---------------------------------------------------------------------===
  // Hillis-Steele inclusive scan on a VL-wide vector using VectorShiftOp.
  // O(log2(VL)) steps instead of O(VL) serial combine.
  //
  // gcu.vector_shift fills vacated lanes with 0.  This is the identity for
  // add / or / xor, so those ops work directly.  For ops whose identity
  // is not 0, the shifted result must be patched before combining:
  //
  //   Op          Identity          Zero-fill OK?
  //   ----        --------          -------------
  //   add         0                 yes
  //   or          0                 yes
  //   xor         0                 yes
  //   mul         1                 NO
  //   max(si)     INT_MIN           NO
  //   min(si)     INT_MAX           NO
  //   max(ui)     0                 yes
  //   min(ui)     UINT_MAX          NO
  //   and         all-ones          NO
  //   max(f)      -inf              NO
  //   min(f)      +inf              NO
  //
  // To support a non-zero-identity op, patch the shifted vector:
  //   Value identity = b.create<arith::ConstantOp>(loc, elemTy, identityAttr);
  //   Value identityVec = b.create<vector::BroadcastOp>(loc, vecTy, identity);
  //   Value laneIdx = b.create<gcu::VectorStepOp>(loc, i32VecTy, zeroI32);
  //   Value strideVec =b.create<vector::BroadcastOp>(loc, i32VecTy, strideI32);
  //   // forward: vacated lanes are [0, stride); reverse: [VL-stride, VL)
  //   Value mask = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
  //                                        laneIdx, strideVec);
  //   shifted = b.create<arith::SelectOp>(loc, mask, identityVec, shifted);
  //
  // Precision note (floating-point):
  //   Hillis-Steele changes the association order of additions compared to
  //   the serial chain ((a0+a1)+a2)+... .  This introduces different rounding
  //   but is not systematically worse -- tree-structured addition actually
  //   reduces catastrophic cancellation when magnitudes vary widely.
  //   The same class of precision difference already exists in
  //   applyVectorizationImpl (Blelloch-style blocked prefix sum).
  // ===---------------------------------------------------------------------===
  Value hillisSteeleInclusiveScan(OpBuilder &b, Location loc, Value vec,
                                  int64_t VL, Type elemTy, bool reverse) const {
    auto vecTy = cast<VectorType>(vec.getType());
    auto shiftDir = reverse ? gcu::VectorShiftDirection::LEFT
                            : gcu::VectorShiftDirection::RIGHT;
    for (int64_t stride = 1; stride < VL; stride *= 2) {
      Value shiftVal =
          b.create<arith::ConstantIntOp>(loc, b.getI32Type(), stride);
      Value shifted =
          b.create<gcu::VectorShiftOp>(loc, shiftDir, vec, shiftVal);
      if (isa<FloatType>(elemTy))
        vec = b.create<arith::AddFOp>(loc, vecTy, shifted, vec);
      else
        vec = b.create<arith::AddIOp>(loc, vecTy, shifted, vec);
    }
    return vec;
  }

  // Strategy 1: vectorized exclusive scan via the common 1D-reshape framework.
  // Uses Hillis-Steele O(log2(VL)) parallel prefix sum.
  // Returns the total scalar (final carry), or null Value to
  // signal fallback to runScalarExclusiveScan.
  //
  // Per-block cost: log2(VL) shift+add (inclusive) + 1 extract + 1 scalar add
  // (carry) + 1 shift + 1 broadcast + 1 vector add (inclusive->exclusive).
  // This is O(log2(VL) + 2) vector ops, which is the minimum for producing
  // both the exclusive result and the propagated carry from a single block.
  Value runVectorizedExclusiveScan(Operation *op, OpBuilder &rewriter,
                                   Value srcInput, Value exclusiveOut,
                                   int64_t totalNumElems, bool reverse,
                                   Type elemTy) const {
    auto loc = op->getLoc();
    unsigned bpe = mlir::triton::gcu::getBpe(elemTy);
    unsigned minBpe = bpe < 4 ? 4 : bpe;
    int64_t VL = oaccSizeInBytes / minBpe;
    std::array<int64_t, 3> dims = {1, 1, totalNumElems};
    int64_t N, M, tail;
    if (!isOneDimReshapeCandidate(dims, /*scanAxis=*/2, VL, N, M, tail))
      return Value();

    ProcessBlockFn exclusiveScan = [&](OpBuilder &b, Location loc, Value vec,
                                       Value carry, bool /*isFirstBlock*/) {
      // 1) Hillis-Steele inclusive scan on raw input: O(log2(VL)) steps.
      //    inclusive[i] = sum(vec[0..i])  (block-local, no carry yet)
      Value inclusive =
          hillisSteeleInclusiveScan(b, loc, vec, VL, elemTy, reverse);

      // 2) The new carry for the next block = carry + sum(all block elements).
      int64_t lastLane = reverse ? 0 : VL - 1;
      Value blockSum = b.create<vector::ExtractOp>(loc, inclusive,
                                                   ArrayRef<int64_t>{lastLane});
      Value newCarry = addCombine(b, loc, carry, blockSum, elemTy);

      // 3) Convert inclusive -> exclusive by shifting right by 1 and inserting
      //    the incoming carry at position 0.
      //    exclusive[0] = carry, exclusive[i] = carry + inclusive[i-1]
      auto excShiftDir = reverse ? gcu::VectorShiftDirection::LEFT
                                 : gcu::VectorShiftDirection::RIGHT;
      Value oneShift = b.create<arith::ConstantIntOp>(loc, b.getI32Type(), 1);
      Value excVec =
          b.create<gcu::VectorShiftOp>(loc, excShiftDir, inclusive, oneShift);
      // Shifted-in lane is 0 from VectorShiftOp; add carry to all lanes.
      auto vecTy = cast<VectorType>(vec.getType());
      Value carryVec = b.create<vector::BroadcastOp>(loc, vecTy, carry);
      if (isa<FloatType>(elemTy))
        excVec = b.create<arith::AddFOp>(loc, vecTy, excVec, carryVec);
      else
        excVec = b.create<arith::AddIOp>(loc, vecTy, excVec, carryVec);

      return std::make_pair(excVec, newCarry);
    };
    ProcessScalarFn scalarCombine = [&](OpBuilder &b, Location loc, Value acc,
                                        Value elem) {
      return std::make_pair(acc, addCombine(b, loc, acc, elem, elemTy));
    };
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, elemTy, rewriter.getZeroAttr(elemTy));
    return applyOneDimReshapeScanCommon(rewriter, loc, srcInput, exclusiveOut,
                                        N, VL, reverse, elemTy, zeroVal,
                                        exclusiveScan, scalarCombine);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 2)
      return failure();

    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op))
      pTagPool.releaseMap(op);

    auto loc = op->getLoc();

    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    auto reverseAttr = op->getAttrOfType<BoolAttr>("reverse");
    if (!axisAttr || !reverseAttr)
      return failure();

    unsigned axis = axisAttr.getInt();
    bool reverse = reverseAttr.getValue();

    auto srcTensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!srcTensorTy)
      return failure();
    auto elemTy = srcTensorTy.getElementType();
    Value adaptorSrc = operands[0];

    unsigned bpe = mlir::triton::gcu::getBpe(elemTy);
    unsigned minBpe = bpe < 4 ? 4 : bpe;
    unsigned vectorLength = oaccSizeInBytes / minBpe;

    auto numElems = triton::gcu::getElemsPerThread(srcTensorTy);
    std::array<int64_t, 3> scanInOutDims;
    int64_t scanAxis;
    foldTo3DScanShape(numElems, axis, scanInOutDims, scanAxis);

    // Allocate per-thread outputs for exclusive result
    auto exclusiveTensorTy =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto exclusiveMemRefTy = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(exclusiveTensorTy));
    if (!exclusiveMemRefTy)
      return failure();
    auto exclusiveLastUser = userAnalysis.getLastUser(op->getResult(0));

    Value exclusiveOut =
        syncAllocOp(rewriter, loc, exclusiveLastUser, userAnalysis,
                    replaced2Origin, exclusiveMemRefTy);
    Value totalScalar;

    if (mustRunOnMasterThread(srcTensorTy, axis, scanInOutDims, scanAxis,
                              vectorLength)) {
      auto tag = pTagPool.getPrivateSyncTagInfo(op);

      // Store src to shared memory
      Value sharedSrc = storeToSharedMem(rewriter, tag, srcTensorTy, adaptorSrc,
                                         false, std::make_pair(op, -1),
                                         userAnalysis, replaced2Origin);

      // Load to thread 0
      auto mergedTensorTy = RankedTensorType::get(
          srcTensorTy.getShape(), elemTy,
          triton::gpu::getDefaultBlockedEncoding(
              getContext(), srcTensorTy.getShape(), 1, 1, 1));
      Value mergedSrc =
          loadFromSharedMem(rewriter, tag, mergedTensorTy, sharedSrc, true,
                            std::make_pair(op, -1), std::make_pair(nullptr, -1),
                            userAnalysis, replaced2Origin);

      // Allocate merged exclusive output on thread 0
      auto mergedMemRefTy =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(mergedTensorTy));
      Value mergedExclusive =
          syncAllocOp(rewriter, loc, std::make_pair(op, -1), userAnalysis,
                      replaced2Origin, mergedMemRefTy);

      int64_t totalMergedElems = 1;
      for (auto d : srcTensorTy.getShape())
        totalMergedElems *= d;

      // Allocate a 0-d memref to pass total through the scf::IfOp boundary
      auto totalScalarMemRef = MemRefType::get({}, elemTy);
      Value mergedTotal =
          syncAllocOp(rewriter, loc, std::make_pair(op, -1), userAnalysis,
                      replaced2Origin, totalScalarMemRef);

      // Execute on master thread only
      auto masterWarpId = getMasterThreadId(op);
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &b, Location loc) {
            Value tv =
                runVectorizedExclusiveScan(op, b, mergedSrc, mergedExclusive,
                                           totalMergedElems, reverse, elemTy);
            if (!tv)
              tv = runScalarExclusiveScan(op, b, mergedSrc, mergedExclusive,
                                          totalMergedElems, reverse, elemTy);
            b.create<memref::StoreOp>(loc, tv, mergedTotal);
            b.create<scf::YieldOp>(loc);
          });

      // Scatter exclusive result back via shared memory
      Value sharedExclusive = storeToSharedMem(
          rewriter, tag, mergedTensorTy, mergedExclusive, true,
          std::make_pair(op, -1), userAnalysis, replaced2Origin);
      exclusiveOut = loadFromSharedMem(
          rewriter, tag, exclusiveTensorTy, sharedExclusive, false,
          exclusiveLastUser, std::make_pair(nullptr, -1), userAnalysis,
          replaced2Origin);

      // Total: all threads read from the shared 0-d memref
      totalScalar = rewriter.create<memref::LoadOp>(loc, mergedTotal);
    } else {
      // Single-warp path: direct scan on per-thread data
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(srcTensorTy);
      totalScalar =
          runVectorizedExclusiveScan(op, rewriter, adaptorSrc, exclusiveOut,
                                     totalNumElems, reverse, elemTy);
      if (!totalScalar)
        totalScalar =
            runScalarExclusiveScan(op, rewriter, adaptorSrc, exclusiveOut,
                                   totalNumElems, reverse, elemTy);
    }

    // Build final outputs
    SmallVector<Value, 2> finalOutputs;
    {
      auto [strides, offset] = exclusiveMemRefTy.getStridesAndOffset();
      if (exclusiveMemRefTy.getNumElements() !=
          dyn_cast<MemRefType>(exclusiveOut.getType()).getNumElements()) {
        finalOutputs.push_back(rewriter.create<memref::ReinterpretCastOp>(
            loc, exclusiveMemRefTy, exclusiveOut, offset,
            exclusiveMemRefTy.getShape(), strides));
      } else {
        finalOutputs.push_back(exclusiveOut);
      }
    }
    finalOutputs.push_back(totalScalar);

    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }
};

} // namespace

void mlir::triton::populateScanOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TTScanOpLowering>(converter, patterns.getContext(), userAnalysis,
                                 replaced2Origin, pTagPool);
  patterns.add<TleExclusiveCumsumLowering>(converter, patterns.getContext(),
                                           userAnalysis, replaced2Origin,
                                           pTagPool);
}
