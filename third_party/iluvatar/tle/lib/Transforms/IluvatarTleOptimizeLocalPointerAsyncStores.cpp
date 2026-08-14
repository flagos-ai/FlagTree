// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLEOPTIMIZELOCALPOINTERASYNCSTORES
#include "Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

// Marks the AsyncCopyGlobalToLocalOp produced by fusing a tle.copy staging
// (global tt.load + local_pointers tt.store) so downstream passes can identify
// the origin if needed.
constexpr StringLiteral kAsyncStoreAttr = "iluvatar_tle.local_ptr_async_store";

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value stripStoreValueWrappers(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static bool isGlobalPointerTensor(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return false;
  auto ptrTy = dyn_cast<tt::PointerType>(tensorTy.getElementType());
  if (!ptrTy)
    return false;
  return ptrTy.getAddressSpace() == 1;
}

static Value stripIndexValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = cvt.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t> getConstantIntLike(Value value) {
  Value current = stripIndexValueWrappers(value);
  if (auto splat = current.getDefiningOp<tt::SplatOp>())
    return getConstantIntLike(splat.getSrc());
  if (auto cst = current.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }
  if (auto cst = current.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = current.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static bool matchRangeWithStaticOffset(Value value, int64_t extent,
                                       int64_t &offset) {
  Value current = stripIndexValueWrappers(value);
  if (auto range = current.getDefiningOp<tt::MakeRangeOp>()) {
    offset = range.getStart();
    return range.getEnd() - range.getStart() == extent;
  }

  auto add = current.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto tryMatch = [&](Value lhs, Value rhs) -> bool {
    Value lhsStripped = stripIndexValueWrappers(lhs);
    auto range = lhsStripped.getDefiningOp<tt::MakeRangeOp>();
    if (!range)
      return false;
    std::optional<int64_t> cst = getConstantIntLike(rhs);
    if (!cst)
      return false;
    offset = range.getStart() + *cst;
    return range.getEnd() - range.getStart() == extent;
  };

  return tryMatch(add.getLhs(), add.getRhs()) ||
         tryMatch(add.getRhs(), add.getLhs());
}

static bool matchFullIndexTensorForAxis(Value index, size_t axis,
                                        ArrayRef<int64_t> shape,
                                        int64_t &offset) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger())
    return false;
  if (indexTy.getShape() != shape)
    return false;

  Value current = stripIndexValueWrappers(index);
  if (shape.size() == 1)
    return matchRangeWithStaticOffset(current, shape.front(), offset);

  auto bcast = current.getDefiningOp<tt::BroadcastOp>();
  if (!bcast)
    return false;

  auto bcastSrcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
  if (!bcastSrcTy || bcastSrcTy.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    const int64_t expected = dim == axis ? dimSize : 1;
    if (bcastSrcTy.getShape()[dim] != expected)
      return false;
  }

  current = stripIndexValueWrappers(bcast.getSrc());
  while (auto expand = current.getDefiningOp<tt::ExpandDimsOp>())
    current = stripIndexValueWrappers(expand.getSrc());

  auto rangeTy = dyn_cast<RankedTensorType>(current.getType());
  if (!rangeTy || rangeTy.getRank() != 1)
    return false;
  if (rangeTy.getShape()[0] != shape[axis])
    return false;

  return matchRangeWithStaticOffset(current, shape[axis], offset);
}

struct StaticSubviewMatch {
  Value baseMemDesc;
  SmallVector<int32_t> offsets;
  RankedTensorType valueType;
};

struct AsyncStoreCandidate {
  tt::StoreOp store;
  tt::LoadOp load;
  StaticSubviewMatch match;
  Value originalStoreValue;
};

static std::optional<StaticSubviewMatch>
matchStaticSubviewMemDesc(tt::StoreOp store) {
  Value ptr = stripConvertLayouts(store.getPtr());
  auto localPointers = ptr.getDefiningOp<LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!valueTy || !ptrTy || !memDescTy)
    return std::nullopt;
  if (valueTy.getShape() != ptrTy.getShape())
    return std::nullopt;
  if (valueTy.getElementType() != memDescTy.getElementType())
    return std::nullopt;

  auto memDescShape = memDescTy.getShape();
  SmallVector<int32_t> offsets(memDescTy.getRank(), 0);
  auto indices = localPointers.getIndices();
  if (indices.empty()) {
    if (llvm::equal(valueTy.getShape(), memDescShape))
      return StaticSubviewMatch{localPointers.getSrc(), std::move(offsets),
                                valueTy};
    return std::nullopt;
  }
  if (indices.size() != static_cast<size_t>(memDescTy.getRank()))
    return std::nullopt;

  for (auto [axis, index] : llvm::enumerate(indices)) {
    int64_t offset = 0;
    if (!matchFullIndexTensorForAxis(index, axis, valueTy.getShape(), offset))
      return std::nullopt;
    if (offset < 0 || offset + valueTy.getShape()[axis] > memDescShape[axis])
      return std::nullopt;
    offsets[axis] = static_cast<int32_t>(offset);
  }

  return StaticSubviewMatch{localPointers.getSrc(), std::move(offsets),
                            valueTy};
}

static Value createSubviewForStore(OpBuilder &builder, Location loc,
                                   const StaticSubviewMatch &match) {
  auto memDescTy = cast<ttg::MemDescType>(match.baseMemDesc.getType());
  bool isFullView =
      llvm::equal(match.valueType.getShape(), memDescTy.getShape()) &&
      llvm::all_of(match.offsets, [](int32_t offset) { return offset == 0; });
  if (isFullView)
    return match.baseMemDesc;

  auto subTy = ttg::MemDescType::get(
      match.valueType.getShape(), match.valueType.getElementType(),
      memDescTy.getEncoding(), memDescTy.getMemorySpace(),
      memDescTy.getMutableMemory(), memDescTy.getAllocShape());
  return ttg::MemDescSubsliceOp::create(builder, loc, subTy, match.baseMemDesc,
                                        match.offsets);
}

static std::optional<AsyncStoreCandidate>
matchAsyncStoreCandidate(tt::StoreOp store) {
  if (!store.getBoundaryCheck().empty())
    return std::nullopt;
  // tle.copy never emits a mask; keep the fusion limited to that unmasked
  // staging pattern.
  if (store.getMask())
    return std::nullopt;

  Value strippedStoreValue = stripStoreValueWrappers(store.getValue());
  auto load = strippedStoreValue.getDefiningOp<tt::LoadOp>();
  if (!load || !load->hasOneUse())
    return std::nullopt;
  if (load.getIsVolatile() || load.getMask() || load.getOther())
    return std::nullopt;
  if (!isa<RankedTensorType>(load.getType()))
    return std::nullopt;
  if (!isGlobalPointerTensor(load.getPtr()))
    return std::nullopt;

  auto match = matchStaticSubviewMemDesc(store);
  if (!match)
    return std::nullopt;
  auto loadTy = cast<RankedTensorType>(load.getType());
  if (loadTy.getShape() != match->valueType.getShape())
    return std::nullopt;
  if (loadTy.getElementType() != match->valueType.getElementType())
    return std::nullopt;

  return AsyncStoreCandidate{store, load, std::move(*match), store.getValue()};
}

static bool mayOverlap(const StaticSubviewMatch &lhs,
                       const StaticSubviewMatch &rhs) {
  if (lhs.baseMemDesc != rhs.baseMemDesc)
    return false;

  for (auto [lhsOffset, rhsOffset, lhsSize, rhsSize] :
       llvm::zip_equal(lhs.offsets, rhs.offsets, lhs.valueType.getShape(),
                       rhs.valueType.getShape())) {
    int64_t lhsBegin = lhsOffset;
    int64_t lhsEnd = lhsBegin + lhsSize;
    int64_t rhsBegin = rhsOffset;
    int64_t rhsEnd = rhsBegin + rhsSize;
    if (lhsEnd <= rhsBegin || rhsEnd <= lhsBegin)
      return false;
  }
  return true;
}

// Only side-effect-free ops and additional global loads may sit between grouped
// async copies and their shared commit/wait; anything else forces the group to
// close so we never reorder past an unrelated memory effect.
static bool canInterleaveBeforeGroupedWait(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && isGlobalPointerTensor(load.getPtr());
  return false;
}

static void eraseDeadStoreValueWrappers(Value originalStoreValue,
                                        tt::LoadOp load) {
  for (Value current = originalStoreValue; current != load.getResult();) {
    Operation *def = current.getDefiningOp();
    auto cvt = dyn_cast_or_null<ttg::ConvertLayoutOp>(def);
    if (!cvt || !cvt->use_empty())
      break;
    current = cvt.getSrc();
    cvt.erase();
  }
  load.erase();
}

static void rewriteAsyncStoreGroup(ArrayRef<AsyncStoreCandidate *> group) {
  if (group.empty())
    return;

  SmallVector<Value> tokens;
  tokens.reserve(group.size());
  for (AsyncStoreCandidate *candidate : group) {
    tt::StoreOp store = candidate->store;
    OpBuilder builder(store);
    Value dst =
        createSubviewForStore(builder, store.getLoc(), candidate->match);
    // Uses the compatibility builder (inputStride = {}, contiguity = 1); the
    // later coalesce_async_copy pass refines vectorization.
    auto asyncCopy = ttg::AsyncCopyGlobalToLocalOp::create(
        builder, store.getLoc(), candidate->load.getPtr(), dst,
        candidate->load.getMask(), candidate->load.getOther(),
        candidate->load.getCache(), candidate->load.getEvict(),
        candidate->load.getIsVolatile());
    asyncCopy->setAttr(kAsyncStoreAttr, builder.getUnitAttr());
    tokens.push_back(asyncCopy.getToken());
  }

  tt::StoreOp lastStore = group.back()->store;
  OpBuilder builder(lastStore);
  builder.setInsertionPointAfter(lastStore);
  auto commit = ttg::AsyncCommitGroupOp::create(builder, lastStore.getLoc(),
                                                ValueRange(tokens));
  ttg::AsyncWaitOp::create(builder, lastStore.getLoc(), commit.getResult(), 0);

  for (AsyncStoreCandidate *candidate : group)
    candidate->store.erase();
  for (AsyncStoreCandidate *candidate : group)
    eraseDeadStoreValueWrappers(candidate->originalStoreValue, candidate->load);
}

class OptimizeLocalPointerAsyncStoresPass
    : public impl::TritonIluvatarTleOptimizeLocalPointerAsyncStoresBase<
          OptimizeLocalPointerAsyncStoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<Operation *> orderedStores;
    llvm::DenseMap<Operation *, AsyncStoreCandidate> candidates;
    module.walk([&](tt::StoreOp store) {
      orderedStores.push_back(store.getOperation());
      if (auto candidate = matchAsyncStoreCandidate(store))
        candidates.try_emplace(store.getOperation(), std::move(*candidate));
    });

    llvm::DenseSet<Operation *> processed;
    for (Operation *storeOp : orderedStores) {
      auto it = candidates.find(storeOp);
      if (it == candidates.end() || processed.contains(storeOp))
        continue;

      SmallVector<AsyncStoreCandidate *> group;
      group.push_back(&it->second);

      // Extend the group with immediately following async-store candidates that
      // do not overlap, allowing them to share one commit/wait.
      for (Operation *next = storeOp->getNextNode(); next;
           next = next->getNextNode()) {
        auto candidateIt = candidates.find(next);
        if (candidateIt != candidates.end()) {
          bool overlaps = llvm::any_of(group, [&](AsyncStoreCandidate *entry) {
            return mayOverlap(entry->match, candidateIt->second.match);
          });
          if (overlaps)
            break;
          group.push_back(&candidateIt->second);
          continue;
        }
        if (!canInterleaveBeforeGroupedWait(next))
          break;
      }

      for (AsyncStoreCandidate *candidate : group)
        processed.insert(candidate->store.getOperation());
      rewriteAsyncStoreGroup(group);
    }
  }
};

} // namespace
} // namespace mlir::triton::iluvatar_tle
