// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "LayoutConversionUtils.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRCoalesceAsyncCopy.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorTileAxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIROPTIMIZESHAREDLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

// Stage allocation/view types until all affected accesses have been checked.
struct SharedLayoutUpdates {
  llvm::MapVector<Value, Type> types;
  SmallVector<OpOperand *> accesses;

  Type getType(Value value) const;
  void apply() const;
};

Value getSharedViewSource(Value value) {
  Operation *op = value.getDefiningOp();
  if (isa_and_nonnull<ttg::MemDescIndexOp, ttg::MemDescTransOp>(op))
    return op->getOperand(0);
  return {};
}

Attribute projectTranspose(Attribute encoding, ArrayRef<int64_t> shape,
                           ArrayRef<int32_t> order) {
  Attribute result;
  auto *interface =
      encoding.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>();
  if (!interface || failed(interface->inferTransOpEncoding(
                        encoding, shape, order, result, /*loc=*/{})))
    return {};
  return result;
}

// A pipeline slot adds/drops only the leading allocation dimension.
Attribute projectIndex(Attribute encoding, bool toRoot) {
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding);
  if (!shared || ttg::getNumCTAs(shared) != 1)
    return {};
  SmallVector<unsigned> order;
  if (toRoot) {
    for (unsigned dim : shared.getOrder())
      order.push_back(dim + 1);
    order.push_back(0);
  } else {
    if (shared.getOrder().back() != 0)
      return {};
    for (unsigned dim : shared.getOrder().drop_back())
      order.push_back(dim - 1);
  }
  return ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), shared.getVec(), shared.getPerPhase(),
      shared.getMaxPhase(), order,
      ttg::CTAEncodingAttr::getDefault(encoding.getContext(), order.size()));
}

Attribute getSharedSourceEncoding(Value view, Attribute encoding) {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, true);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getType().getShape(),
                            triton::inversePermutation(trans.getOrder()));
  return {};
}

Attribute getSharedViewEncoding(Value view, Attribute encoding) {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, false);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getSrc().getType().getShape(),
                            trans.getOrder());
  return {};
}

Type SharedLayoutUpdates::getType(Value value) const {
  Type type = types.lookup(value);
  return type ? type : value.getType();
}

void SharedLayoutUpdates::apply() const {
  for (auto [value, type] : types)
    value.setType(type);
}

Value getSharedAllocation(Value value) {
  while (value && !value.getDefiningOp<ttg::LocalAllocOp>())
    value = getSharedViewSource(value);
  return value;
}

// Does not mutate IR. Discard the updates if collection or validation fails.
LogicalResult collectSharedLayoutUpdates(Value allocation, Attribute encoding,
                                         SharedLayoutUpdates &updates) {
  if (!allocation.getDefiningOp<ttg::LocalAllocOp>())
    return failure();
  SmallVector<std::pair<Value, Attribute>> worklist{{allocation, encoding}};
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto [value, layout] = worklist[i];
    auto type = dyn_cast<ttg::MemDescType>(value.getType());
    if (!type || !isa<ttg::SharedEncodingTrait>(layout))
      return failure();
    auto replacement = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), layout, type.getMemorySpace(),
        type.getMutableMemory(), type.getAllocShape());
    auto [it, inserted] = updates.types.try_emplace(value, replacement);
    if (!inserted) {
      if (it->second != replacement)
        return failure();
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getNumResults() == 1 &&
          getSharedViewSource(user->getResult(0)) == value) {
        Value view = user->getResult(0);
        Attribute projected = getSharedViewEncoding(view, layout);
        if (!projected)
          return failure();
        worklist.emplace_back(view, projected);
      } else {
        updates.accesses.push_back(&use);
      }
    }
  }
  return success();
}

// Express writer orders in allocation coordinates. Use neutral swizzle
// parameters only to reuse the existing slot/transpose projection rules.
llvm::DenseMap<Value, Attribute>
collectWriterOrders(ModuleOp module, ModuleAxisInfoAnalysis &axisInfo) {
  llvm::DenseMap<Value, Attribute> orders;
  auto collect = [&](OpOperand &source, Value destination) {
    Value root = getSharedAllocation(destination);
    if (!root || root.getDefiningOp()->hasAttr("tle.gpu_layout"))
      return;
    auto type = cast<RankedTensorType>(source.get().getType());
    Operation *writer = source.getOwner();
    Attribute preferred;
    if (auto store = dyn_cast<ttg::LocalStoreOp>(writer))
      preferred = getTileStoreSourcePreference(store);
    else if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(writer))
      preferred = getAsyncCopySourcePreference(copy, axisInfo);
    if (preferred)
      type = type.cloneWithEncoding(preferred);
    Attribute order;
    if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())) {
      auto contiguity = ttg::getContigPerThread(type);
      auto writerOrder = blocked.getOrder();
      unsigned width = contiguity[writerOrder.front()];
      // A scalar or equally contiguous writer supplies no clear preference.
      if (width > 1 && llvm::all_of(writerOrder.drop_front(), [&](unsigned d) {
            return width > contiguity[d];
          })) {
        order = ttg::SwizzledSharedEncodingAttr::get(
            module.getContext(), 1, 1, 1, writerOrder, blocked.getCTALayout());
        for (Value view = destination; view != root && order;
             view = getSharedViewSource(view))
          order = getSharedSourceEncoding(view, order);
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

LogicalResult checkSharedAccesses(const SharedLayoutUpdates &updates,
                                  ModuleAxisInfoAnalysis &axisInfo) {
  for (OpOperand *access : updates.accesses) {
    Operation *op = access->getOwner();
    auto shared = cast<ttg::MemDescType>(updates.getType(access->get()));
    if (ttg::getNumCTAs(shared.getEncoding()) != 1)
      return failure();
    if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      if (!canWriteMmaSharedLayout(copy, shared, axisInfo))
        return failure();
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

SmallVector<ttg::LocalLoadOp> collectMmaOperandSharedLoads(Value operand) {
  SmallVector<ttg::LocalLoadOp> loads;
  llvm::SmallSetVector<Value, 4> sources;
  sources.insert(operand);
  for (unsigned i = 0; i < sources.size(); ++i) {
    Value value = sources[i];
    if (auto load = value.getDefiningOp<ttg::LocalLoadOp>())
      loads.push_back(load);
    else
      appendTensorSources(value, sources);
  }
  return loads;
}

void optimizeSharedLayouts(ModuleOp module, ArrayRef<triton::DotOp> dots,
                           ModuleAxisInfoAnalysis &axisInfo) {
  auto writerOrders = collectWriterOrders(module, axisInfo);
  // A null encoding marks an allocation whose consumers cannot agree on a
  // supported layout. Keep that allocation unchanged, including all its views.
  llvm::MapVector<Value, Attribute> encodings;
  for (triton::DotOp dot : dots) {
    for (unsigned index : {0u, 1u}) {
      Attribute operandEncoding =
          cast<RankedTensorType>(dot->getOperand(index).getType())
              .getEncoding();
      for (auto load : collectMmaOperandSharedLoads(dot->getOperand(index))) {
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
    if (failed(collectSharedLayoutUpdates(root, encoding, updates)) ||
        failed(checkSharedAccesses(updates, axisInfo)))
      continue;
    accepted.push_back(std::move(updates));
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
    if (dots.empty())
      return;
    ModuleAxisInfoAnalysis axisInfo(module, addTensorTileAxisInfoVisitors);
    optimizeSharedLayouts(module, dots, axisInfo);
  }
};

} // namespace
} // namespace mlir::triton::metax
