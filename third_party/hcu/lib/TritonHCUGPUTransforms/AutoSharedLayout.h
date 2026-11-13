#ifndef TRITON_HCU_AUTO_SHARED_LAYOUT_H
#define TRITON_HCU_AUTO_SHARED_LAYOUT_H

namespace mlir::triton::hcugpu {
namespace {

// Retarget only a closed descriptor/view graph. In particular, a shared pointer
// escaping to arbitrary arithmetic cannot be reinterpreted in a different chart.
// This changes types, not the positions of loads, stores or rendezvous.
class AutoSharedLayout {
  ttg::LocalAllocOp alloc;
  DenseSet<Value> visited;
  SmallVector<Value> descriptors;
  SmallVector<ttg::DotOperandEncodingAttr> consumers;
  std::optional<SmallVector<unsigned>> sourceOrder;
  unsigned budget = 4096;

  bool producer(Value pointer) {
    auto type = dyn_cast<RankedTensorType>(pointer.getType());
    auto ptr = type ? dyn_cast<tt::PointerType>(type.getElementType())
                    : tt::PointerType();
    if (!ptr || ptr.getAddressSpace() != 1 || type.getRank() != 2)
      return false;
    auto order = ttg::getOrderForMemory(type);
    if (sourceOrder && *sourceOrder != order)
      return false;
    sourceOrder = order;
    return true;
  }

  bool stored(Value value) {
    while (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      if (!budget--) return false;
      value = convert.getSrc();
    }
    auto load = value.getDefiningOp<tt::LoadOp>();
    return load && !load.getIsVolatile() && producer(load.getPtr());
  }

  bool read(Value value) {
    if (!budget--) return false;
    bool found = false;
    for (OpOperand &use : value.getUses()) {
      if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(use.getOwner())) {
        if (!read(convert.getResult())) return false;
      } else if (auto dot = dyn_cast<tt::DotOp>(use.getOwner())) {
        auto type = cast<RankedTensorType>(value.getType());
        auto operand = dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding());
        if (!operand || operand.getOpIdx() != use.getOperandNumber() ||
            !isa<ttg::AMDMfmaEncodingAttr>(operand.getParent()))
          return false;
        consumers.push_back(operand);
      } else {
        return false;
      }
      found = true;
    }
    return found;
  }

  bool close(Value value) {
    if (!budget--) return false;
    if (!visited.insert(value).second) return true;
    descriptors.push_back(value);
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        for (Region *region : ws.getPartitionRegions())
          if (!close(region->getArgument(use.getOperandNumber()))) return false;
      } else if (auto index = dyn_cast<ttg::MemDescIndexOp>(op)) {
        // Only the independent leading stage dimension may be removed.
        if (cast<ttg::MemDescType>(value.getType()).getRank() != 3 ||
            !close(index.getResult())) return false;
      } else if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
        if (!close(slice.getResult())) return false;
      } else if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
        if (!read(load.getResult())) return false;
      } else if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        if (use.getOperandNumber() != 1 || !stored(store.getSrc())) return false;
      } else if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        if (use.getOperandNumber() != 1 || copy.getIsVolatile() ||
            !producer(copy.getSrc())) return false;
      } else if (!isa<ttg::LocalDeallocOp>(op)) {
        return false;
      }
    }
    return true;
  }

public:
  explicit AutoSharedLayout(ttg::LocalAllocOp alloc) : alloc(alloc) {}

  void run() {
    auto type = alloc.getType();
    unsigned rank = type.getRank();
    if (alloc.getSrc() || (rank != 2 && rank != 3) ||
        !isa<ttg::SwizzledSharedEncodingAttr>(type.getEncoding()) ||
        !isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()) ||
        !close(alloc.getResult()) || !sourceOrder || consumers.empty())
      return;
    auto shape = type.getShape().take_back(2);
    auto cta = ttg::getCTALayout(type.getEncoding());
    // Cross-CTA shared ownership needs its own proof. This is not a shape or
    // tile whitelist: ordinary single-CTA MFMA layouts share this derivation.
    if (llvm::any_of(cta.getCTAsPerCGA(), [](unsigned n) { return n != 1; }) ||
        llvm::any_of(cta.getCTASplitNum(), [](unsigned n) { return n != 1; }))
      return;
    auto payloadCTA = ttg::CTAEncodingAttr::getDefault(alloc.getContext(), 2);
    ttg::SwizzledSharedEncodingAttr payload;
    for (auto consumer : consumers) {
      auto next = ttg::SwizzledSharedEncodingAttr::get(
          alloc.getContext(), consumer, shape, *sourceOrder, payloadCTA,
          type.getElementTypeBitWidth(), false);
      if (payload && payload != next) return;
      payload = next;
    }
    // Views retain their complete allocation geometry. Selecting from a
    // half-window's shape would silently change the XOR addressing of siblings.
    SmallVector<std::pair<Value, Type>> updates;
    for (Value value : descriptors) {
      auto old = cast<ttg::MemDescType>(value.getType());
      if (old.getRank() != 2 && old.getRank() != 3) return;
      SmallVector<unsigned> order(payload.getOrder());
      if (old.getRank() == 3) {
        for (auto &axis : order) ++axis;
        order.push_back(0);
      }
      auto encoding = ttg::SwizzledSharedEncodingAttr::get(
          alloc.getContext(), payload.getVec(), payload.getPerPhase(),
          payload.getMaxPhase(), order,
          ttg::getCTALayout(old.getEncoding()));
      updates.emplace_back(value, ttg::MemDescType::get(
          old.getShape(), old.getElementType(), encoding, old.getMemorySpace(),
          old.getMutableMemory(), old.getAllocShape()));
    }
    for (auto [value, replacement] : updates) value.setType(replacement);
    alloc->removeAttr("hcu.auto_shared_layout");
  }
};

} // namespace
} // namespace mlir::triton::hcugpu
#endif
