/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/PermutationAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

struct StoreCandidate {
  Operation *operation;
  StaticAccess access;
  RankedTensorType pointerType;
  RankedTensorType valueType;
  RankedTensorType offsetType;
  Type pointeeType;
  uint64_t elementBytes;
  int64_t elementCount;
  Value value;
  triton::CacheModifier cache;
  triton::EvictionPolicy evict;
  DictionaryAttr attributes;
};

struct StoreCoalescingRun {
  SmallVector<StoreCandidate, 4> addressOrderStores;
  SmallVector<Operation *, 4> programOrderStores;
  Operation *anchor = nullptr;
  int64_t firstOffset = 0;
  int64_t endExclusive = 0;
  int64_t totalElements = 0;
  uint64_t totalBytes = 0;
};

bool fitsI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

bool checkedAddI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs)
    return false;
  if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedSubI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs)
    return false;
  if (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs)
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> getByteWidth(Type type) {
  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    bitWidth = integerType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
  } else {
    return std::nullopt;
  }

  // i1 and sub-byte integer/float element types are deliberately excluded:
  // capacity is accounted in concrete addressable bytes, never rounded lanes.
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return static_cast<uint64_t>(bitWidth / 8);
}

std::optional<int64_t> getStaticElementCount(RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      type.getRank() < 1)
    return std::nullopt;

  int64_t elementCount = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension <= 0 ||
        (elementCount != 0 &&
         dimension > std::numeric_limits<int64_t>::max() / elementCount))
      return std::nullopt;
    elementCount *= dimension;
  }
  return elementCount;
}

bool isUnencodedStaticTensor(RankedTensorType type) {
  return getStaticElementCount(type).has_value();
}

bool isUnencodedStaticRankOneTensor(RankedTensorType type) {
  return isUnencodedStaticTensor(type) && type.getRank() == 1;
}

bool isDirectFunctionBodyStore(triton::StoreOp store, triton::FuncOp function,
                               Block *block) {
  if (!store || !function || !block || store->getBlock() != block ||
      store->getNumRegions() != 0)
    return false;
  if (block->getParent() != &function.getBody())
    return false;
  triton::FuncOp parent = store->getParentOfType<triton::FuncOp>();
  return parent && parent.getOperation() == function.getOperation();
}

bool hasExpectedStoreTypes(const StoreCandidate &candidate) {
  if (!isUnencodedStaticTensor(candidate.pointerType) ||
      !isUnencodedStaticTensor(candidate.valueType) ||
      !isUnencodedStaticTensor(candidate.offsetType))
    return false;
  if (candidate.pointerType.getShape() != candidate.valueType.getShape() ||
      candidate.pointerType.getShape() != candidate.offsetType.getShape() ||
      candidate.pointerType.getEncoding() !=
          candidate.valueType.getEncoding() ||
      candidate.pointerType.getEncoding() != candidate.offsetType.getEncoding())
    return false;

  auto pointerElement =
      dyn_cast<triton::PointerType>(candidate.pointerType.getElementType());
  if (!pointerElement || triton::isTensorPointerType(pointerElement) ||
      pointerElement.getPointeeType() != candidate.pointeeType ||
      candidate.valueType.getElementType() != candidate.pointeeType)
    return false;

  int64_t span = 0;
  if (!checkedSubI64(candidate.access.lastOffset, candidate.access.firstOffset,
                     span) ||
      !checkedAddI64(span, 1, span) || span <= 0 ||
      span != candidate.elementCount)
    return false;
  return true;
}

std::optional<StoreCandidate>
matchStore(triton::StoreOp store, triton::FuncOp function, Block *block) {
  if (!isDirectFunctionBodyStore(store, function, block) || store.getMask() ||
      !store.getBoundaryCheck().empty())
    return std::nullopt;

  auto pointerType = dyn_cast<RankedTensorType>(store.getPtr().getType());
  auto valueType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!pointerType || !valueType || !isUnencodedStaticTensor(pointerType) ||
      !isUnencodedStaticTensor(valueType))
    return std::nullopt;

  auto pointerElement =
      dyn_cast<triton::PointerType>(pointerType.getElementType());
  if (!pointerElement || triton::isTensorPointerType(pointerElement) ||
      pointerElement.getPointeeType() != valueType.getElementType() ||
      pointerType.getShape() != valueType.getShape() ||
      pointerType.getEncoding() != valueType.getEncoding())
    return std::nullopt;

  std::optional<uint64_t> elementBytes =
      getByteWidth(valueType.getElementType());
  if (!elementBytes)
    return std::nullopt;
  std::optional<int64_t> elementCount = getStaticElementCount(valueType);
  if (!elementCount)
    return std::nullopt;

  StaticAccessAnalysis accessAnalysis;
  StaticAccessProof proof = accessAnalysis.analyzeStore(store);
  if (!proof.isProven() || !proof.access->isLogicalRowMajorContiguous() ||
      proof.access->pointer != store.getPtr() || !proof.access->base ||
      proof.access->shape != pointerType.getShape() ||
      proof.access->elementCount != *elementCount)
    return std::nullopt;

  auto offsetType = dyn_cast<RankedTensorType>(proof.access->offset.getType());
  if (!offsetType)
    return std::nullopt;

  StoreCandidate candidate{
      store.getOperation(), std::move(*proof.access),
      pointerType,          valueType,
      offsetType,           pointerElement.getPointeeType(),
      *elementBytes,        *elementCount,
      store.getValue(),     store.getCache(),
      store.getEvict(),     store->getAttrDictionary()};
  if (!hasExpectedStoreTypes(candidate))
    return std::nullopt;
  return candidate;
}

bool canShareBucket(const StoreCandidate &lhs, const StoreCandidate &rhs) {
  // Base and dynamic-origin comparisons are deliberately SSA identity only.
  // This rule never consults alias analysis to coalesce output stores.
  return lhs.operation->getBlock() == rhs.operation->getBlock() &&
         lhs.access.base == rhs.access.base &&
         lhs.access.dynamicOrigin == rhs.access.dynamicOrigin &&
         lhs.pointeeType == rhs.pointeeType &&
         lhs.valueType.getElementType() == rhs.valueType.getElementType() &&
         lhs.pointerType.getEncoding() == rhs.pointerType.getEncoding() &&
         lhs.valueType.getEncoding() == rhs.valueType.getEncoding() &&
         lhs.cache == rhs.cache && lhs.evict == rhs.evict &&
         lhs.attributes == rhs.attributes;
}

bool isBeforeInProgramOrder(Operation *lhs, Operation *rhs) {
  return lhs && rhs && lhs != rhs && lhs->getBlock() == rhs->getBlock() &&
         lhs->isBeforeInBlock(rhs);
}

bool isLocalValueBeforeAnchor(Value value, Operation *anchor,
                              DominanceInfo &dominance) {
  if (!value || !anchor || !dominance.dominates(value, anchor))
    return false;

  if (auto blockArgument = dyn_cast<BlockArgument>(value))
    return blockArgument.getOwner() == anchor->getBlock();

  Operation *definition = value.getDefiningOp();
  return definition && definition->getBlock() == anchor->getBlock() &&
         definition->isBeforeInBlock(anchor);
}

bool valuesDominateAnchor(const StoreCoalescingRun &run,
                          triton::FuncOp function) {
  if (!run.anchor || !function)
    return false;

  DominanceInfo dominance(function.getOperation());
  for (const StoreCandidate &candidate : run.addressOrderStores) {
    if (!candidate.value || !candidate.access.base ||
        !isLocalValueBeforeAnchor(candidate.value, run.anchor, dominance) ||
        !isLocalValueBeforeAnchor(candidate.access.base, run.anchor,
                                  dominance) ||
        (candidate.access.dynamicOrigin &&
         !isLocalValueBeforeAnchor(candidate.access.dynamicOrigin, run.anchor,
                                   dominance)))
      return false;
  }
  return true;
}

std::optional<StoreCoalescingRun>
buildRun(ArrayRef<StoreCandidate> addressOrderStores, triton::FuncOp function,
         unsigned ubCapacityBytes,
         const EntryArgPointerAliasAnalysis *entryAliases) {
  if (!function || !entryAliases || ubCapacityBytes == 0 ||
      addressOrderStores.size() < 2)
    return std::nullopt;

  const StoreCandidate &first = addressOrderStores.front();
  const StoreCandidate &last = addressOrderStores.back();
  for (const StoreCandidate &candidate : addressOrderStores) {
    if (candidate.elementBytes != first.elementBytes)
      return std::nullopt;
  }
  int64_t endExclusive = 0;
  int64_t totalElements = 0;
  if (!checkedAddI64(last.access.lastOffset, 1, endExclusive) ||
      !checkedSubI64(endExclusive, first.access.firstOffset, totalElements) ||
      totalElements <= 0 || !fitsI32(first.access.firstOffset) ||
      !fitsI32(endExclusive))
    return std::nullopt;

  int64_t summedElements = 0;
  for (const StoreCandidate &candidate : addressOrderStores) {
    if (candidate.elementCount <= 0 ||
        !checkedAddI64(summedElements, candidate.elementCount, summedElements))
      return std::nullopt;
  }
  if (summedElements != totalElements)
    return std::nullopt;

  uint64_t totalBytes = 0;
  if (!checkedMulU64(static_cast<uint64_t>(totalElements), first.elementBytes,
                     totalBytes) ||
      totalBytes > static_cast<uint64_t>(ubCapacityBytes))
    return std::nullopt;

  StoreCoalescingRun run;
  run.addressOrderStores.append(addressOrderStores.begin(),
                                addressOrderStores.end());
  run.firstOffset = first.access.firstOffset;
  run.endExclusive = endExclusive;
  run.totalElements = totalElements;
  run.totalBytes = totalBytes;
  for (const StoreCandidate &candidate : run.addressOrderStores)
    run.programOrderStores.push_back(candidate.operation);
  std::stable_sort(run.programOrderStores.begin(), run.programOrderStores.end(),
                   isBeforeInProgramOrder);
  run.anchor = run.programOrderStores.back();

  if (!valuesDominateAnchor(run, function))
    return std::nullopt;

  // Keep the StaticAccess descriptions aligned with program order.  Address
  // order determines packed-value layout, while program order determines which
  // stores are delayed across a given intervening operation.
  SmallVector<StaticAccess, 4> programOrderAccesses;
  programOrderAccesses.reserve(run.programOrderStores.size());
  for (Operation *operation : run.programOrderStores) {
    auto candidate = std::find_if(run.addressOrderStores.begin(),
                                  run.addressOrderStores.end(),
                                  [operation](const StoreCandidate &store) {
                                    return store.operation == operation;
                                  });
    if (candidate == run.addressOrderStores.end())
      return std::nullopt;
    programOrderAccesses.push_back(candidate->access);
  }

  // The combined store is inserted at the program-order last store.  For the
  // gap (S[i - 1], S[i]), only S[0] ... S[i - 1] move from before the gap to
  // after it.  A load/store in the gap may overlap a later planned store whose
  // relative order is unchanged, but must be statically disjoint from every
  // delayed prefix store.  All other memory effects remain fail-closed.
  ProtectedIntervalAnalysis intervalAnalysis;
  for (size_t index = 1; index < run.programOrderStores.size(); ++index) {
    llvm::ArrayRef<StaticAccess> delayedAccesses(programOrderAccesses.data(),
                                                 index);
    if (!intervalAnalysis
             .proveNoConflictingLoadStoreEffects(
                 run.programOrderStores[index - 1],
                 run.programOrderStores[index], delayedAccesses, entryAliases)
             .isProven())
      return std::nullopt;
  }

  if (run.addressOrderStores.size() - 1 >
      static_cast<size_t>(std::numeric_limits<unsigned>::max()))
    return std::nullopt;
  return run;
}

SmallVector<StoreCoalescingRun, 4>
findRuns(triton::FuncOp function, unsigned ubCapacityBytes,
         const EntryArgPointerAliasAnalysis *entryAliases) {
  SmallVector<StoreCoalescingRun, 4> runs;
  if (!function || !entryAliases || ubCapacityBytes == 0)
    return runs;

  for (Block &block : function.getBody()) {
    SmallVector<StoreCandidate, 8> stores;
    for (Operation &operation : block) {
      auto store = dyn_cast<triton::StoreOp>(&operation);
      if (!store)
        continue;
      std::optional<StoreCandidate> candidate =
          matchStore(store, function, &block);
      if (candidate)
        stores.push_back(std::move(*candidate));
    }

    SmallVector<SmallVector<StoreCandidate, 4>, 4> buckets;
    for (StoreCandidate &candidate : stores) {
      auto bucket = std::find_if(
          buckets.begin(), buckets.end(), [&](const auto &existing) {
            return !existing.empty() &&
                   canShareBucket(existing.front(), candidate);
          });
      if (bucket == buckets.end()) {
        buckets.emplace_back();
        buckets.back().push_back(candidate);
      } else {
        bucket->push_back(candidate);
      }
    }

    for (SmallVector<StoreCandidate, 4> &bucket : buckets) {
      if (bucket.size() < 2)
        continue;

      std::stable_sort(
          bucket.begin(), bucket.end(),
          [](const StoreCandidate &lhs, const StoreCandidate &rhs) {
            if (lhs.access.firstOffset != rhs.access.firstOffset)
              return lhs.access.firstOffset < rhs.access.firstOffset;
            if (lhs.access.lastOffset != rhs.access.lastOffset)
              return lhs.access.lastOffset < rhs.access.lastOffset;
            return isBeforeInProgramOrder(lhs.operation, rhs.operation);
          });

      // An overlap invalidates every candidate in this bucket.  In particular,
      // do not try to salvage a later non-overlapping sub-run.
      bool hasOverlap = false;
      for (size_t index = 1; index < bucket.size(); ++index) {
        if (bucket[index].access.firstOffset <=
            bucket[index - 1].access.lastOffset) {
          hasOverlap = true;
          break;
        }
      }
      if (hasOverlap)
        continue;

      size_t runBegin = 0;
      for (size_t index = 1; index <= bucket.size(); ++index) {
        bool endsRun = index == bucket.size();
        if (!endsRun) {
          int64_t expectedNextOffset = 0;
          endsRun = !checkedAddI64(bucket[index - 1].access.lastOffset, 1,
                                   expectedNextOffset) ||
                    expectedNextOffset != bucket[index].access.firstOffset;
        }

        if (!endsRun)
          continue;
        if (index - runBegin >= 2) {
          std::optional<StoreCoalescingRun> run =
              buildRun(ArrayRef<StoreCandidate>(bucket).slice(runBegin,
                                                              index - runBegin),
                       function, ubCapacityBytes, entryAliases);
          if (run)
            runs.push_back(std::move(*run));
        }
        runBegin = index;
      }
    }
  }
  return runs;
}

bool hasSameAddressOrder(const StoreCoalescingRun &run,
                         ArrayRef<Operation *> operations) {
  if (run.addressOrderStores.size() != operations.size())
    return false;
  for (size_t index = 0; index < operations.size(); ++index) {
    if (run.addressOrderStores[index].operation != operations[index])
      return false;
  }
  return true;
}

std::optional<StoreCoalescingRun>
findRunByOperations(triton::FuncOp function, unsigned ubCapacityBytes,
                    const EntryArgPointerAliasAnalysis *entryAliases,
                    ArrayRef<Operation *> operations) {
  for (StoreCoalescingRun &run :
       findRuns(function, ubCapacityBytes, entryAliases)) {
    if (hasSameAddressOrder(run, operations))
      return run;
  }
  return std::nullopt;
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

bool recordVerifiedOperation(Operation *operation,
                             SmallVectorImpl<Operation *> &created) {
  if (!operation)
    return false;
  created.push_back(operation);
  return succeeded(mlir::verify(operation));
}

std::optional<Value>
materializeRowMajorFlatValue(IRRewriter &rewriter,
                             const StoreCandidate &candidate,
                             SmallVectorImpl<Operation *> &created) {
  SmallVector<int64_t, 1> flatShape = {candidate.elementCount};
  RankedTensorType flatType = candidate.valueType.clone(flatShape);
  if (!isUnencodedStaticRankOneTensor(flatType))
    return std::nullopt;
  if (candidate.valueType == flatType)
    return candidate.value;

  auto reshape = rewriter.create<triton::ReshapeOp>(
      candidate.operation->getLoc(), flatType, candidate.value);
  if (!recordVerifiedOperation(reshape.getOperation(), created))
    return std::nullopt;
  return reshape.getResult();
}

LogicalResult applyRun(IRRewriter &rewriter, const StoreCoalescingRun &run) {
  if (!run.anchor || run.addressOrderStores.size() < 2 ||
      run.programOrderStores.size() != run.addressOrderStores.size() ||
      run.totalElements <= 0 || !fitsI32(run.firstOffset) ||
      !fitsI32(run.endExclusive))
    return failure();

  const StoreCandidate &exemplar = run.addressOrderStores.front();
  SmallVector<int64_t, 1> packedShape = {run.totalElements};
  // clone() retains any type details carried by the original tensor.  V1 has
  // already required unencoded types.
  RankedTensorType packedValueType = exemplar.valueType.clone(packedShape);
  RankedTensorType packedPointerType = exemplar.pointerType.clone(packedShape);
  RankedTensorType packedOffsetType = exemplar.offsetType.clone(packedShape);
  if (!isUnencodedStaticRankOneTensor(packedValueType) ||
      !isUnencodedStaticRankOneTensor(packedPointerType) ||
      !isUnencodedStaticRankOneTensor(packedOffsetType) ||
      packedPointerType.getShape() != packedValueType.getShape() ||
      packedPointerType.getShape() != packedOffsetType.getShape())
    return failure();

  SmallVector<Operation *, 16> created;
  rewriter.setInsertionPoint(run.anchor);
  auto empty = rewriter.create<tensor::EmptyOp>(run.anchor->getLoc(),
                                                packedValueType, ValueRange{});
  if (!recordVerifiedOperation(empty.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  Value packedValue = empty.getResult();
  int64_t packedOffset = 0;
  for (const StoreCandidate &candidate : run.addressOrderStores) {
    const int64_t sliceLength = candidate.elementCount;
    if (sliceLength <= 0) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    std::optional<Value> flatValue =
        materializeRowMajorFlatValue(rewriter, candidate, created);
    if (!flatValue) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    SmallVector<OpFoldResult, 1> offsets = {
        rewriter.getIndexAttr(packedOffset)};
    SmallVector<OpFoldResult, 1> sizes = {rewriter.getIndexAttr(sliceLength)};
    SmallVector<OpFoldResult, 1> strides = {rewriter.getIndexAttr(1)};
    auto inserted = rewriter.create<tensor::InsertSliceOp>(
        run.anchor->getLoc(), *flatValue, packedValue, offsets, sizes, strides);
    if (!recordVerifiedOperation(inserted.getOperation(), created)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    packedValue = inserted.getResult();

    int64_t nextPackedOffset = 0;
    if (!checkedAddI64(packedOffset, sliceLength, nextPackedOffset)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    packedOffset = nextPackedOffset;
  }
  if (packedOffset != run.totalElements) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  auto range = rewriter.create<triton::MakeRangeOp>(
      run.anchor->getLoc(), packedOffsetType,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(run.firstOffset)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(run.endExclusive)));
  if (!recordVerifiedOperation(range.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  Value packedPointerOffset = range.getResult();
  if (exemplar.access.dynamicOrigin) {
    auto originSplat = rewriter.create<triton::SplatOp>(
        run.anchor->getLoc(), packedOffsetType, exemplar.access.dynamicOrigin);
    if (!recordVerifiedOperation(originSplat.getOperation(), created)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    auto offsetAdd = rewriter.create<arith::AddIOp>(
        run.anchor->getLoc(), originSplat.getResult(), range.getResult());
    if (!recordVerifiedOperation(offsetAdd.getOperation(), created)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    packedPointerOffset = offsetAdd.getResult();
  }

  auto baseSplat = rewriter.create<triton::SplatOp>(
      run.anchor->getLoc(), packedPointerType, exemplar.access.base);
  if (!recordVerifiedOperation(baseSplat.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }
  auto pointer = rewriter.create<triton::AddPtrOp>(
      run.anchor->getLoc(), packedPointerType, baseSplat.getResult(),
      packedPointerOffset);
  if (!recordVerifiedOperation(pointer.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  auto combinedStore = rewriter.create<triton::StoreOp>(
      run.anchor->getLoc(), pointer.getResult(), packedValue, exemplar.cache,
      exemplar.evict);
  // Bucket construction required complete attribute equality.  Copying the
  // full dictionary therefore preserves every store property and extra attr.
  combinedStore->setAttrs(exemplar.attributes);
  if (!recordVerifiedOperation(combinedStore.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  StaticAccessAnalysis accessAnalysis;
  StaticAccessProof rebuiltProof =
      accessAnalysis.analyzePointer(pointer.getResult());
  if (!rebuiltProof.isProven() || !rebuiltProof.access->isRankOneContiguous() ||
      rebuiltProof.access->base != exemplar.access.base ||
      rebuiltProof.access->dynamicOrigin != exemplar.access.dynamicOrigin ||
      rebuiltProof.access->firstOffset != run.firstOffset ||
      rebuiltProof.access->lastOffset != run.endExclusive - 1) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  // All operations have verified and the replacement pointer has re-proven
  // its static interval before any original memory effect is erased.
  for (Operation *operation : llvm::reverse(run.programOrderStores))
    rewriter.eraseOp(operation);
  return success();
}

class StoreCoalescingPlan final : public RewritePlan {
public:
  StoreCoalescingPlan(const StoreCoalescingRun &run, triton::FuncOp function,
                      const EntryArgPointerAliasAnalysis *entryAliases,
                      unsigned ubCapacityBytes, unsigned epoch)
      : function(function), anchor(run.anchor), entryAliases(entryAliases),
        ubCapacityBytes(ubCapacityBytes), epoch(epoch) {
    for (const StoreCandidate &candidate : run.addressOrderStores)
      addressOrderStores.push_back(candidate.operation);
  }

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::StoreCoalescing;
  }

  unsigned getBenefit() const override {
    return addressOrderStores.size() < 2
               ? 0
               : static_cast<unsigned>(addressOrderStores.size() - 1);
  }

  Operation *getAnchor() const override { return anchor; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !function || !anchor || !entryAliases ||
        context.getFunction().getOperation() != function.operator->() ||
        &context.getEntryArgPointerAliasAnalysis() != entryAliases)
      return failure();

    std::optional<StoreCoalescingRun> current = findRunByOperations(
        function, ubCapacityBytes, entryAliases, addressOrderStores);
    if (!current || current->anchor != anchor)
      return failure();
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    if (!function || !anchor || !entryAliases)
      return failure();
    std::optional<StoreCoalescingRun> current = findRunByOperations(
        function, ubCapacityBytes, entryAliases, addressOrderStores);
    if (!current || current->anchor != anchor)
      return failure();
    return applyRun(rewriter, *current);
  }

private:
  triton::FuncOp function;
  Operation *anchor;
  SmallVector<Operation *, 4> addressOrderStores;
  const EntryArgPointerAliasAnalysis *entryAliases;
  unsigned ubCapacityBytes;
  unsigned epoch;
};

class StoreCoalescingRule final : public GraphOptimizationRule {
public:
  StoreCoalescingRule(unsigned ubCapacityBytes, bool enabledForCompileMode)
      : ubCapacityBytes(ubCapacityBytes),
        enabledForCompileMode(enabledForCompileMode) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::StoreCoalescing;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return !enabledForCompileMode || ubCapacityBytes == 0
               ? AnalysisRequirement::None
               : AnalysisRequirement::EntryArgPointerAlias;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (!enabledForCompileMode || ubCapacityBytes == 0)
      return success();

    const EntryArgPointerAliasAnalysis *entryAliases =
        &context.getEntryArgPointerAliasAnalysis();
    for (const StoreCoalescingRun &run :
         findRuns(context.getFunction(), ubCapacityBytes, entryAliases)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] matched graph optimization rule "
                 << static_cast<unsigned>(getId()) << " ("
                 << getGraphOptimizationRuleName(getId()) << ") at "
                 << run.anchor->getLoc()
                 << ": stores=" << run.programOrderStores.size() << " elements="
                 << run.totalElements << " bytes=" << run.totalBytes << "\n");
      plans.push_back(std::make_unique<StoreCoalescingPlan>(
          run, context.getFunction(), entryAliases, ubCapacityBytes,
          context.getEpoch()));
    }
    return success();
  }

private:
  unsigned ubCapacityBytes;
  bool enabledForCompileMode;
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createStoreCoalescingRule(unsigned ubCapacityBytes,
                               const StoreCoalescingRuleOptions &options) {
  return std::make_unique<StoreCoalescingRule>(ubCapacityBytes,
                                               options.enabledForCompileMode);
}
