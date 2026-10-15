// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

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

static bool reachesDot(Value value) {
  for (Operation *user : value.getUsers()) {
    if (isa<tt::DotOpInterface>(user))
      return true;
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user))
      if (reachesDot(cvt.getResult()))
        return true;
  }
  return false;
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

static Value matchRowStrideMul(Value offset, unsigned contigDim) {
  Value current = stripIndexValueWrappers(offset);
  while (auto bcast = current.getDefiningOp<tt::BroadcastOp>())
    current = stripIndexValueWrappers(bcast.getSrc());
  auto mul = current.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return {};
  auto match = [&](Value index, Value stride) -> Value {
    auto expand =
        stripIndexValueWrappers(index).getDefiningOp<tt::ExpandDimsOp>();
    if (!expand || expand.getAxis() != contigDim)
      return {};
    auto splat = stripIndexValueWrappers(stride).getDefiningOp<tt::SplatOp>();
    if (!splat || !splat.getSrc().getType().isInteger())
      return {};
    return splat.getSrc();
  };
  if (Value stride = match(mul.getLhs(), mul.getRhs()))
    return stride;
  return match(mul.getRhs(), mul.getLhs());
}

// A null result means the address expression was not recognized, in which case
// the caller keeps the copy on the ordinary non-SME path.
static Value findRowStride(Value ptr, unsigned contigDim) {
  for (Value current = stripConvertLayouts(ptr); current;) {
    if (auto addptr = current.getDefiningOp<tt::AddPtrOp>()) {
      if (Value stride = matchRowStrideMul(addptr.getOffset(), contigDim))
        return stride;
      current = stripConvertLayouts(addptr.getPtr());
      continue;
    }
    if (auto bcast = current.getDefiningOp<tt::BroadcastOp>()) {
      current = stripConvertLayouts(bcast.getSrc());
      continue;
    }
    return {};
  }
  return {};
}

// Mirrors AccelerateMatmul's getUseSmeFlagFromPtr: the bit position in the
// launch-time mask is the kernel argument number of the base pointer, so the
// dtype/contiguity/64-byte-alignment checks already ran against the real
// tensor.
static bool isSmeEligibleBase(Value ptr, unsigned useSme) {
  for (Value current = ptr; current;) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (isa<tt::PointerType>(blockArg.getType()))
        return (1u << blockArg.getArgNumber()) & useSme;
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp)
        return false;
      auto init = forOp.getTiedLoopInit(blockArg);
      if (!init)
        return false;
      current = init->get();
      continue;
    }
    Operation *def = current.getDefiningOp();
    if (!def || def->getNumOperands() == 0)
      return false;
    // splat/broadcast/addptr/convert_layout all carry the base in operand 0.
    current = def->getOperand(0);
  }
  return false;
}

struct SmeCopyPlan {
  ttg::BlockedEncodingAttr ptrEncoding;
  Value rowStride;
};

static std::optional<SmeCopyPlan> planSmeCopy(AsyncStoreCandidate &candidate,
                                              Value dst, unsigned useSme) {
  auto dstTy = cast<ttg::MemDescType>(dst.getType());
  if (dstTy.getRank() != 2)
    return std::nullopt;
  // useTcu is what tle.gpu.alloc(nv_mma_shared_layout=True) selects, and it is
  // the shared encoding that describes SME's hardware write pattern, so the
  // local_load read-back side needs no adjustment.
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
  if (!shared || !shared.getUseTcu())
    return std::nullopt;
  // SME writes whole tiles; a subslice destination is not handled yet.
  if (dst != candidate.match.baseMemDesc)
    return std::nullopt;

  Type elemTy = dstTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  if (bitwidth != 8 && bitwidth != 16 && bitwidth != 32)
    return std::nullopt;

  auto ptrTy = dyn_cast<RankedTensorType>(candidate.load.getPtr().getType());
  if (!ptrTy)
    return std::nullopt;
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ptrTy.getEncoding());
  if (!blocked || blocked.getIsSme())
    return std::nullopt;
  if (shared.getOrder()[0] != blocked.getOrder()[0])
    return std::nullopt;
  if (!isSmeEligibleBase(candidate.load.getPtr(), useSme))
    return std::nullopt;

  unsigned contigDim = blocked.getOrder()[0];
  Value rowStride = findRowStride(candidate.load.getPtr(), contigDim);
  if (!rowStride)
    return std::nullopt;

  auto mod = candidate.load->getParentOfType<ModuleOp>();
  auto smeEnc = ttg::BlockedEncodingAttr::get(
      ptrTy.getContext(), /*isSme=*/true, /*smeMask=*/false,
      ttg::lookupNumWarps(mod), elemTy, ptrTy.getShape(), blocked.getOrder(),
      blocked.getSizePerThread(), blocked.getThreadsPerWarp(),
      blocked.getWarpsPerCTA(), ttg::TritonGPUDialect::getNumCTAs(mod));

  // One transfer covers 16 rows x 64 bytes, so a tile that is not an exact
  // multiple of what smeWarpsPerCTA covers would be only partially copied.
  auto smeWpt = smeEnc.getSmeWarpsPerCTA();
  if (smeWpt.size() != 2)
    return std::nullopt;
  SmallVector<unsigned, 2> tile({16, 16});
  tile[contigDim] = 512 / bitwidth;
  for (unsigned dim = 0; dim < 2; ++dim) {
    unsigned covered = smeWpt[dim] * tile[dim];
    if (covered == 0 || ptrTy.getShape()[dim] % covered != 0)
      return std::nullopt;
  }

  return SmeCopyPlan{smeEnc, rowStride};
}

static bool readsBackIntoDot(Value storePtr) {
  Value root = stripConvertLayouts(storePtr);
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> seen{root};
  while (!worklist.empty()) {
    for (Operation *user : worklist.pop_back_val().getUsers()) {
      if (auto reader = dyn_cast<tt::LoadOp>(user)) {
        if (reachesDot(reader.getResult()))
          return true;
        continue;
      }
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user))
        if (seen.insert(cvt.getResult()).second)
          worklist.push_back(cvt.getResult());
    }
  }
  return false;
}

static bool shouldDeferToPipeliner(AsyncStoreCandidate &candidate,
                                   int64_t numStages, unsigned useSme) {
  auto forOp = dyn_cast<scf::ForOp>(candidate.store->getParentOp());
  if (!forOp)
    return false;
  auto stageAttr = forOp->getAttrOfType<IntegerAttr>("tt.num_stages");
  if (stageAttr ? stageAttr.getInt() <= 1 : numStages <= 1)
    return false;

  // The buffer has to outlive the iteration and be written whole, which is what
  // promote's dominance and shape checks require of the staging it rewrites.
  Operation *baseDef = candidate.match.baseMemDesc.getDefiningOp();
  if (!baseDef || forOp->isAncestor(baseDef))
    return false;
  auto memDescTy =
      cast<ttg::MemDescType>(candidate.match.baseMemDesc.getType());
  if (candidate.match.valueType.getShape() != memDescTy.getShape())
    return false;
  if (!llvm::all_of(candidate.match.offsets,
                    [](int32_t offset) { return offset == 0; }))
    return false;

  auto shared =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(memDescTy.getEncoding());
  auto ptrTy = dyn_cast<RankedTensorType>(candidate.load.getPtr().getType());
  auto blocked =
      ptrTy ? dyn_cast<ttg::BlockedEncodingAttr>(ptrTy.getEncoding()) : nullptr;
  if (!shared || !blocked || shared.getOrder()[0] != blocked.getOrder()[0])
    return false;
  if (shared.getOrder()[0] == 0 &&
      !planSmeCopy(candidate, candidate.match.baseMemDesc, useSme))
    return false;

  tt::StoreOp store = candidate.store;
  return readsBackIntoDot(store.getPtr());
}

static void rewriteAsyncStoreGroup(ArrayRef<AsyncStoreCandidate *> group,
                                   unsigned useSme) {
  if (group.empty())
    return;

  SmallVector<Value> tokens;
  tokens.reserve(group.size());
  for (AsyncStoreCandidate *candidate : group) {
    tt::StoreOp store = candidate->store;
    OpBuilder builder(store);
    Value dst =
        createSubviewForStore(builder, store.getLoc(), candidate->match);

    std::optional<SmeCopyPlan> sme = planSmeCopy(*candidate, dst, useSme);

    ttg::AsyncCopyGlobalToLocalOp asyncCopy;
    if (sme) {
      Value ptr = candidate->load.getPtr();
      auto ptrTy = cast<RankedTensorType>(ptr.getType());
      auto smePtrTy = RankedTensorType::get(
          ptrTy.getShape(), ptrTy.getElementType(), sme->ptrEncoding);
      Value smePtr =
          ttg::ConvertLayoutOp::create(builder, ptr.getLoc(), smePtrTy, ptr);
      Value stride = sme->rowStride;
      if (stride.getType().isInteger(64))
        stride = arith::TruncIOp::create(builder, store.getLoc(),
                                         builder.getI32Type(), stride);
      asyncCopy = ttg::AsyncCopyGlobalToLocalOp::create(
          builder, store.getLoc(), smePtr, dst, candidate->load.getMask(),
          candidate->load.getOther(), stride, candidate->load.getCache(),
          candidate->load.getEvict(), candidate->load.getIsVolatile(),
          /*contiguity=*/1);
    } else {
      asyncCopy = ttg::AsyncCopyGlobalToLocalOp::create(
          builder, store.getLoc(), candidate->load.getPtr(), dst,
          candidate->load.getMask(), candidate->load.getOther(),
          candidate->load.getCache(), candidate->load.getEvict(),
          candidate->load.getIsVolatile());
    }
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

struct OptimizeLocalPointerAsyncStoresPass
    : public impl::TritonIluvatarTleOptimizeLocalPointerAsyncStoresBase<
          OptimizeLocalPointerAsyncStoresPass> {
  using impl::TritonIluvatarTleOptimizeLocalPointerAsyncStoresBase<
      OptimizeLocalPointerAsyncStoresPass>::
      TritonIluvatarTleOptimizeLocalPointerAsyncStoresBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<Operation *> orderedStores;
    llvm::DenseMap<Operation *, AsyncStoreCandidate> candidates;
    module.walk([&](tt::StoreOp store) {
      orderedStores.push_back(store.getOperation());
      auto candidate = matchAsyncStoreCandidate(store);
      if (!candidate || shouldDeferToPipeliner(*candidate, numStages, useSme))
        return;
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
      rewriteAsyncStoreGroup(group, useSme);
    }
  }
};

} // namespace
} // namespace mlir::triton::iluvatar_tle
