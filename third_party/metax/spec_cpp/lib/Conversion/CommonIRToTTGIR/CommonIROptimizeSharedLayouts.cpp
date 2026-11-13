#include "mlir-ext/Dialect/CommonIR/Analysis/AxisInfoExt.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/CommonIRToTTGIR/AsyncCopyUtils.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/SharedLayoutUtils.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIROPTIMIZESHAREDLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

using WriterLayoutMap = llvm::MapVector<Operation *, Attribute>;

ttg::BlockedEncodingAttr
getCopyCoalescedPreference(ModuleOp module, ttg::AsyncCopyGlobalToLocalOp copy,
                           ModuleAxisInfoAnalysis &axisInfo) {
  auto type = copy.getSrc().getType();
  return ttg::buildCoalescedEncoding(
      module.getContext(), axisInfo, copy, ttg::lookupNumWarps(copy),
      ttg::TritonGPUDialect::getThreadsPerWarp(module),
      ttg::getCTALayout(type.getEncoding()), ttg::getShapePerCTA(type));
}

bool comesFromLogicalTile(Value value) {
  // Coalesce may already have placed a conversion immediately after the tile.
  while (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
    value = convert.getSrc();
  return isa_and_nonnull<tile::ExtractTileOp, tile::InsertTileOp>(
      value.getDefiningOp());
}

// Writer preferences guide shared order; they are not applied to tensors here.
void collectCopyWriterPreferences(ModuleOp module,
                                  ModuleAxisInfoAnalysis &axisInfo,
                                  WriterLayoutMap &preferences) {
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
  module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
    if (comesFromLogicalTile(copy.getSrc()))
      copies.push_back(copy);
  });
  if (copies.empty())
    return;
  for (auto copy : copies) {
    if (tile::getExplicitMemoryEncoding(copy) ||
        llvm::any_of(copy->getOperands(), [](Value value) {
          return bool(tile::getExplicitValueEncoding(value));
        }))
      continue;
    preferences[copy] = getCopyCoalescedPreference(module, copy, axisInfo);
  }
}

// Ordinary stores inherit the existing coalesced load layout through views.
void collectLoadStoreLayouts(ModuleOp module, WriterLayoutMap &layouts) {
  TTGIRLayoutAdapter adapter;
  module.walk([&](ttg::LocalStoreOp store) {
    Value source = store.getSrc();
    SmallVector<Operation *> views;
    bool hasTile = false;
    // Follow only logical views. Loads stay effect boundaries: use their
    // existing coalesced encoding without cloning or moving the memory read.
    while (true) {
      if (Value input = adapter.getConversionSource(source)) {
        source = input;
        continue;
      }
      Operation *op = source.getDefiningOp();
      if (!isa_and_nonnull<tile::ExtractTileOp, triton::TransOp>(op))
        break;
      hasTile |= isa<tile::ExtractTileOp>(op);
      views.push_back(op);
      source = op->getOperand(0);
    }
    if (!hasTile)
      return;

    Attribute encoding;
    // Include prefetched loads carried across loop iterations. Conflicting
    // load encodings do not provide one unambiguous transport requirement.
    for (Value value : collectTensorLayoutComponent(source, adapter)) {
      auto load = value.getDefiningOp<triton::LoadOp>();
      if (!load)
        continue;
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !isa<ttg::BlockedEncodingAttr>(type.getEncoding()))
        return;
      if (encoding && encoding != type.getEncoding())
        return;
      encoding = type.getEncoding();
    }
    if (!encoding)
      return;
    for (Operation *view : llvm::reverse(views)) {
      if (isa<triton::TransOp>(view))
        encoding = mlir::inferDstEncoding(view, encoding);
      if (!encoding)
        return;
    }
    layouts[store] = encoding;
  });
}

// Express writer orders in allocation coordinates. Use neutral swizzle
// parameters only to reuse the existing slot/transpose projection rules.
llvm::DenseMap<Value, Attribute>
collectWriterOrders(ModuleOp module, const WriterLayoutMap &copyPreferences,
                    const WriterLayoutMap &storeLayouts) {
  llvm::DenseMap<Value, Attribute> orders;
  auto collect = [&](OpOperand &source, Value destination) {
    Value root = getSharedAllocation(destination);
    if (!root || root.getDefiningOp()->hasAttr("tle.gpu_layout"))
      return;
    auto type = dyn_cast<RankedTensorType>(source.get().getType());
    Attribute order;
    if (type) {
      Attribute encoding = type.getEncoding();
      Attribute requested = isa<ttg::LocalStoreOp>(source.getOwner())
                                ? storeLayouts.lookup(source.getOwner())
                                : copyPreferences.lookup(source.getOwner());
      if (requested)
        encoding = requested;
      if (isa<ttg::BlockedEncodingAttr>(encoding)) {
        type = type.cloneWithEncoding(encoding);
        auto contiguity = ttg::getContigPerThread(type);
        auto blocked = cast<ttg::BlockedEncodingAttr>(encoding);
        auto writerOrder = blocked.getOrder();
        // A scalar or equally contiguous writer supplies no clear preference.
        unsigned dim = writerOrder.front();
        bool unambiguous = contiguity[dim] > 1;
        for (unsigned other : writerOrder.drop_front())
          unambiguous &= contiguity[dim] > contiguity[other];
        if (unambiguous) {
          order = ttg::SwizzledSharedEncodingAttr::get(
              module.getContext(), 1, 1, 1, writerOrder,
              ttg::getCTALayout(encoding));
          for (Value view = destination; view != root && order;
               view = getSharedViewSource(view))
            order = getSharedSourceEncoding(view, order);
        }
      }
    }
    auto [it, inserted] = orders.try_emplace(root, order);
    if (!inserted && it->second != order)
      it->second = Attribute{};
  };
  module.walk([&](Operation *op) {
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
      collect(store->getOpOperand(0), store.getDst());
    else if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
      collect(copy->getOpOperand(0), copy.getResult());
    else if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (alloc.getSrc())
        collect(alloc->getOpOperand(0), alloc.getResult());
    }
  });
  return orders;
}

FailureOr<Attribute> inferSharedEncoding(ttg::LocalLoadOp load,
                                         Attribute operandEncoding,
                                         Attribute writerOrder) {
  auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(operandEncoding);
  auto mma =
      dot ? dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent()) : nullptr;
  auto shared = load.getSrc().getType();
  auto type = load.getType();
  if (!mma || mma.getVersionMajor() != 2 || mma.getIsATrans() ||
      mma.getIsBTrans() || load.getMmaMode() != -1 || shared.getRank() != 2 ||
      type.getShape() != shared.getShape() ||
      type.getElementType() != shared.getElementType() ||
      (!type.getElementType().isF16() && !type.getElementType().isBF16()))
    return failure();

  // Check microtile divisibility before constructing the shared encoding.
  auto elems = mma.getElemsPerThreadOrTrans(dot);
  auto atom = mma.getThreadShape(false);
  unsigned first = dot.getOpIdx() == 0 ? atom[0] : atom[2];
  unsigned second = dot.getOpIdx() == 0 ? atom[2] : atom[1];
  if (type.getShape()[0] % (first * elems[0]) ||
      type.getShape()[1] % (second * elems[1]))
    return failure();
  auto order = ttg::getOrderForDotOperand(dot.getOpIdx(), 2, /*kContig=*/true);
  if (auto preferred =
          dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(writerOrder))
    order.assign(preferred.getOrder().begin(), preferred.getOrder().end());
  return Attribute(ttg::SwizzledSharedEncodingAttr::get(
      load.getContext(), dot, type.getShape(), order, ttg::getCTALayout(dot),
      type.getElementType(), /*needTrans=*/false));
}

Attribute chooseCopyLayout(ttg::AsyncCopyGlobalToLocalOp copy,
                           ttg::MemDescType destination,
                           ModuleAxisInfoAnalysis &axisInfo,
                           const WriterLayoutMap &preferences,
                           unsigned minVec = 1) {
  auto shared =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(destination.getEncoding());
  auto source = copy.getSrc().getType();
  auto info = axisInfo.getAxisInfo(copy.getSrc());
  if (!shared || source.getRank() != 2 || !info ||
      source.getShape() != destination.getShape())
    return {};
  auto maxVec = getAsyncCopyMaxVectorSize(copy, destination, axisInfo,
                                          /*emitErrors=*/false);
  if (failed(maxVec))
    return {};
  if (Attribute preferred = preferences.lookup(copy))
    source = source.cloneWithEncoding(preferred);
  return chooseAsyncCopyLayout(source, destination, info, minVec, *maxVec);
}

LogicalResult checkSharedAccesses(const SharedLayoutUpdates &updates,
                                  ModuleAxisInfoAnalysis &axisInfo,
                                  const WriterLayoutMap &preferences,
                                  WriterLayoutMap &copyLayouts) {
  for (OpOperand *access : updates.accesses) {
    Operation *op = access->getOwner();
    auto shared = cast<ttg::MemDescType>(updates.getType(access->get()));
    if (ttg::getNumCTAs(shared.getEncoding()) != 1)
      return failure();
    if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      unsigned elemBytes = shared.getElementTypeBitWidth() / 8;
      Attribute encoding = chooseCopyLayout(copy, shared, axisInfo, preferences,
                                            std::max(4 / elemBytes, 1u));
      if (!encoding)
        return failure();
      // Keep the layout that made this shared candidate feasible. The tile
      // optimizer consumes the resulting conversions without choosing again.
      copyLayouts[copy] = encoding;
    } else if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (load.getMmaMode() != -1 ||
          load.getType().getElementType() != shared.getElementType())
        return failure();
    } else if (!isa<ttg::LocalStoreOp, ttg::LocalDeallocOp>(op)) {
      return failure();
    }
  }
  return success();
}

void optimizeSharedLayouts(ModuleOp module, ArrayRef<triton::DotOp> dots,
                           const WriterLayoutMap &copyPreferences,
                           const WriterLayoutMap &storeLayouts,
                           ModuleAxisInfoAnalysis &axisInfo,
                           WriterLayoutMap &selectedCopies) {
  TTGIRLayoutAdapter tensors;
  auto writerOrders =
      collectWriterOrders(module, copyPreferences, storeLayouts);
  // A null encoding marks an allocation whose consumers cannot agree on a
  // supported layout. Keep that allocation unchanged, including all its views.
  llvm::MapVector<Value, Attribute> encodings;
  for (triton::DotOp dot : dots) {
    for (unsigned index : {0u, 1u}) {
      Attribute operandEncoding =
          cast<RankedTensorType>(dot->getOperand(index).getType())
              .getEncoding();
      for (Value value :
           collectTensorLayoutComponent(dot->getOperand(index), tensors)) {
        auto load = value.getDefiningOp<ttg::LocalLoadOp>();
        if (!load)
          continue;
        Value root = getSharedAllocation(load.getSrc());
        if (!root || root.getDefiningOp()->hasAttr("tle.gpu_layout"))
          continue;
        Attribute writerOrder = writerOrders.lookup(root);
        SmallVector<Value> views;
        for (Value view = load.getSrc(); view != root;
             view = getSharedViewSource(view))
          views.push_back(view);
        for (Value view : llvm::reverse(views)) {
          if (!writerOrder)
            break;
          writerOrder = getSharedViewEncoding(view, writerOrder);
        }
        auto shared = inferSharedEncoding(load, operandEncoding, writerOrder);
        Attribute encoding = succeeded(shared) ? *shared : Attribute{};
        for (Value view : views) {
          if (!encoding)
            break;
          encoding = getSharedSourceEncoding(view, encoding);
        }
        auto [it, inserted] = encodings.try_emplace(root, encoding);
        if (!inserted && it->second != encoding)
          it->second = Attribute{};
      }
    }
  }
  if (encodings.empty())
    return;
  SmallVector<SharedLayoutUpdates> accepted;
  for (auto [root, encoding] : encodings) {
    if (!encoding)
      continue;
    SharedLayoutUpdates updates;
    WriterLayoutMap copyLayouts;
    if (failed(collectSharedLayoutUpdates(root, encoding, updates)) ||
        failed(checkSharedAccesses(updates, axisInfo, copyPreferences,
                                   copyLayouts)))
      continue;
    accepted.push_back(std::move(updates));
    for (auto entry : copyLayouts)
      selectedCopies.insert(entry);
  }
  for (const auto &updates : accepted)
    updates.apply();
}

struct CommonIROptimizeSharedLayoutsPass
    : impl::CommonIROptimizeSharedLayoutsBase<
          CommonIROptimizeSharedLayoutsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::DotOp> dots;
    module.walk([&](triton::DotOp dot) {
      auto mma = dyn_cast_or_null<gpu::MACAMmaEncodingAttr>(
          dot.getType().getEncoding());
      auto elementType = dot.getA().getType().getElementType();
      if (!mma || mma.getVersionMajor() != 2 || mma.getIsATrans() ||
          mma.getIsBTrans() || (!elementType.isF16() && !elementType.isBF16()))
        return;
      dots.push_back(dot);
    });
    // Writer preferences guide shared order even when logical tiles hide
    // global contiguity from the ordinary Coalesce pass.
    WriterLayoutMap copyPreferences, storeLayouts, copyLayouts;
    {
      // Reuse axis info until shared types change; reanalyze copies below.
      ModuleAxisInfoAnalysis axisInfo(module,
                                      tile::addCommonIRAxisInfoVisitors);
      collectCopyWriterPreferences(module, axisInfo, copyPreferences);
      collectLoadStoreLayouts(module, storeLayouts);
      optimizeSharedLayouts(module, dots, copyPreferences, storeLayouts,
                            axisInfo, copyLayouts);
    }

    // Adapt remaining copies to their fixed/default shared layouts.
    SmallVector<ttg::AsyncCopyGlobalToLocalOp> remainingCopies;
    module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
      if (!copyLayouts.contains(copy))
        remainingCopies.push_back(copy);
    });
    if (!remainingCopies.empty()) {
      ModuleAxisInfoAnalysis axisInfo(module,
                                      tile::addCommonIRAxisInfoVisitors);
      for (auto copy : remainingCopies) {
        Attribute encoding = chooseCopyLayout(copy, copy.getResult().getType(),
                                              axisInfo, copyPreferences);
        if (encoding)
          copyLayouts[copy] = encoding;
        // The final legalizer diagnoses unsupported copies.
      }
    }

    TTGIRLayoutAdapter tensors;
    auto convertOperand = [&](OpOperand &use, Attribute encoding) {
      Value value = use.get();
      auto type =
          cast<RankedTensorType>(value.getType()).cloneWithEncoding(encoding);
      use.set(tensors.convertLayout(value, type, use.getOwner()));
    };
    for (auto [copy, encoding] : copyLayouts)
      for (OpOperand &operand : copy->getOpOperands())
        if (isa<RankedTensorType>(operand.get().getType()))
          convertOperand(operand, encoding);
    // Preserve Coalesce's load layout by converting at the store boundary.
    for (auto [store, encoding] : storeLayouts)
      convertOperand(store->getOpOperand(0), encoding);
  }
};

} // namespace
} // namespace mlir::triton::metax
