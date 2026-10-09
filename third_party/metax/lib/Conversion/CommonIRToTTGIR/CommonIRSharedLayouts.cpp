// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/Analysis/AxisInfoExt.h"
#include "mlir/IR/Matchers.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
namespace {

// Express writer orders in allocation coordinates. Use neutral swizzle
// parameters only to reuse the existing slot/transpose projection rules.
llvm::DenseMap<Value, Attribute>
collectWriterOrders(ModuleOp module,
                    const tile::TensorLayoutRequirements &requirements,
                    const tile::TTGIRBufferLayoutAdapter &buffers) {
  llvm::DenseMap<Value, Attribute> orders;
  auto collect = [&](OpOperand &source, Value destination) {
    Value root = tile::getBufferLayoutRoot(destination, buffers);
    if (!root || root.getDefiningOp()->hasAttr("tle.gpu_layout"))
      return;
    auto type = dyn_cast<RankedTensorType>(source.get().getType());
    Attribute order;
    if (type) {
      Attribute encoding = type.getEncoding();
      Attribute requested;
      bool conflict = false;
      for (const auto &requirement : requirements.uses) {
        if (requirement.use != &source)
          continue;
        conflict |= requested && requested != requirement.encoding;
        requested = requirement.encoding;
      }
      if (requested)
        encoding = requested;
      if (!conflict && isa<ttg::BlockedEncodingAttr>(encoding)) {
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
               view = buffers.getViewSource(view))
            order = buffers.toSourceEncoding(view, order);
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
    return load.emitError("automatic shared layout requires a rank-2 FP16/BF16 "
                          "ordinary load for MACA MMA v2");

  // The existing hardware constructor asserts exact MMA microtile coverage.
  // Prove it before calling the constructor so a bad candidate is skippable.
  auto elems = mma.getElemsPerThreadOrTrans(dot);
  auto atom = mma.getThreadShape(false);
  unsigned first = dot.getOpIdx() == 0 ? atom[0] : atom[2];
  unsigned second = dot.getOpIdx() == 0 ? atom[2] : atom[1];
  if (type.getShape()[0] % (first * elems[0]) ||
      type.getShape()[1] % (second * elems[1]))
    return load.emitError("shared tile does not cover whole MMA microtiles");
  auto order = ttg::getOrderForDotOperand(dot.getOpIdx(), 2, /*kContig=*/true);
  if (auto preferred =
          dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(writerOrder))
    order.assign(preferred.getOrder().begin(), preferred.getOrder().end());
  return Attribute(ttg::SwizzledSharedEncodingAttr::get(
      load.getContext(), dot, type.getShape(), order, ttg::getCTALayout(dot),
      type.getElementType(), /*needTrans=*/false));
}

bool isLegalCopyMapping(RankedTensorType source, ttg::MemDescType destination,
                        unsigned width) {
  auto shared =
      cast<ttg::SwizzledSharedEncodingAttr>(destination.getEncoding());
  auto plain = ttg::SwizzledSharedEncodingAttr::get(
      source.getContext(), shared.getVec(), 1, 1, shared.getOrder(),
      shared.getCTALayout());
  auto plainType = ttg::MemDescType::get(
      destination.getShape(), destination.getElementType(), plain,
      destination.getMemorySpace(), destination.getMutableMemory(),
      destination.getAllocShape());
  auto src = ttg::toLinearLayout(source);
  src = actionRemoveBroadcastedRegs(src).apply(src);
  auto free = src.getFreeVariableMasks();
  auto name = [&](StringRef str) {
    return StringAttr::get(source.getContext(), str);
  };
  // This first path requires one physical issuer for every element.
  for (StringRef dim : {"lane", "warp", "block"})
    if (free.lookup(name(dim)))
      return false;
  auto conversion = src.invertAndCompose(ttg::toLinearLayout(plainType));
  if (!conversion.isTrivialOver({name("block")}))
    return false;
  conversion = conversion.sublayout(
      {name("register"), name("lane"), name("warp")}, {name("offset")});
  return conversion.isInjective() && conversion.isSurjective() &&
         conversion.getNumConsecutiveInOut() >= width;
}

LogicalResult
collectSharedCopyRequirement(ttg::AsyncCopyGlobalToLocalOp copy,
                             ttg::MemDescType destination,
                             ModuleAxisInfoAnalysis &axisInfo,
                             tile::TensorLayoutRequirements &requirements) {
  auto shared =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(destination.getEncoding());
  auto source = copy.getSrc().getType();
  auto info = axisInfo.getAxisInfo(copy.getSrc());
  if (!shared || source.getRank() != 2 || !info ||
      source.getShape() != destination.getShape())
    return copy.emitError(
        "automatic shared layout requires a rank-2 async copy");
  unsigned dim = shared.getOrder().front();
  unsigned elemBytes = destination.getElementType().getIntOrFloatBitWidth() / 8;
  unsigned contiguous = std::min<int64_t>(
      info->getContiguity(dim), info->getDivisibility(dim) / elemBytes);
  if (shared.getMaxPhase() > 1)
    contiguous = std::min(contiguous, shared.getVec());
  if (Value mask = copy.getMask()) {
    auto maskInfo = axisInfo.getAxisInfo(mask);
    if (!maskInfo)
      return copy.emitError("cannot prove async-copy mask uniformity");
    // LLVM swizzles source addresses along this axis, but does not swizzle
    // predicates. Axis-uniform masks are unchanged by that permutation.
    if (shared.getMaxPhase() > 1 &&
        maskInfo->getConstancy(dim) < source.getShape()[dim])
      return copy.emitError("automatic swizzled shared layout requires an "
                            "async-copy mask uniform along the swizzled axis");
    contiguous = std::min<int64_t>(contiguous, maskInfo->getConstancy(dim));
  }
  if (Value other = copy.getOther()) {
    while (auto convert = other.getDefiningOp<ttg::ConvertLayoutOp>())
      other = convert.getSrc();
    if (!matchPattern(other, m_Zero()) &&
        !matchPattern(other, m_PosZeroFloat()))
      return copy.emitError("MetaX async copy requires zero fill");
  }
  for (unsigned bytes : {16u, 8u, 4u}) {
    if (bytes % elemBytes || bytes / elemBytes > contiguous)
      continue;
    unsigned width = bytes / elemBytes;
    SmallVector<unsigned> sizePerThread(2, 1);
    sizePerThread[dim] = width;
    auto encoding = ttg::BlockedEncodingAttr::get(
        copy.getContext(), source.getShape(), sizePerThread, shared.getOrder(),
        ttg::lookupNumWarps(copy),
        ttg::TritonGPUDialect::getThreadsPerWarp(
            copy->getParentOfType<ModuleOp>()),
        1);
    if (!isLegalCopyMapping(source.cloneWithEncoding(encoding), destination,
                            width))
      continue;
    // Replace the generic coalescing requirement for these exact ports.
    for (OpOperand &operand : copy->getOpOperands()) {
      if (!isa<RankedTensorType>(operand.get().getType()))
        continue;
      llvm::erase_if(requirements.uses, [&](auto requirement) {
        return requirement.use == &operand;
      });
      requirements.uses.push_back({&operand, encoding});
    }
    return success();
  }
  return copy.emitError("no legal 16/8/4-byte async copy for the selected "
                        "shared layout and proven address continuity");
}
} // namespace

FailureOr<tile::BufferLayoutPlan>
planSharedLayouts(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                  tile::TensorLayoutRequirements &requirements) {
  tile::TTGIRBufferLayoutAdapter buffers;
  tile::TTGIRLayoutAdapter tensors;
  auto writerOrders = collectWriterOrders(module, requirements, buffers);
  SmallVector<tile::BufferLayoutRequirement> sharedRequirements;
  for (auto [dot, plan] : choices) {
    for (unsigned index : {0u, 1u}) {
      Attribute encoding = index == 0 ? plan.operandA : plan.operandB;
      for (Value value : tile::collectTensorLayoutComponent(
               dot->getOperand(index), tensors)) {
        auto load = value.getDefiningOp<ttg::LocalLoadOp>();
        if (!load)
          continue;
        Value root = tile::getBufferLayoutRoot(load.getSrc(), buffers);
        if (!root || root.getDefiningOp()->hasAttr("tle.gpu_layout"))
          continue;
        Attribute writerOrder = writerOrders.lookup(root);
        SmallVector<Value> views;
        for (Value view = load.getSrc(); view != root;
             view = buffers.getViewSource(view))
          views.push_back(view);
        for (Value view : llvm::reverse(views)) {
          if (!writerOrder)
            break;
          writerOrder = buffers.toViewEncoding(view, writerOrder);
        }
        auto shared = inferSharedEncoding(load, encoding, writerOrder);
        if (failed(shared))
          return failure();
        sharedRequirements.push_back({load.getSrc(), *shared});
      }
    }
  }
  auto plan = tile::planBufferLayouts(sharedRequirements, buffers);
  if (failed(plan)) {
    module.emitError(
        "incompatible shared layout requirements or unsupported buffer view");
    return failure();
  }
  if (plan->types.empty())
    return plan;
  ModuleAxisInfoAnalysis axisInfo(module, tile::addCommonIRAxisInfoVisitors);
  for (OpOperand *access : plan->accesses) {
    Operation *op = access->getOwner();
    auto shared = cast<ttg::MemDescType>(plan->getType(access->get()));
    if (ttg::getNumCTAs(shared.getEncoding()) != 1) {
      op->emitError("automatic MetaX shared layout currently requires one CTA");
      return failure();
    }
    if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      if (failed(collectSharedCopyRequirement(copy, shared, axisInfo,
                                              requirements)))
        return failure();
    } else if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (load.getMmaMode() != -1 ||
          load.getType().getElementType() != shared.getElementType()) {
        load.emitError(
            "automatic shared layout has an incompatible local load");
        return failure();
      }
    } else if (!isa<ttg::LocalStoreOp, ttg::LocalDeallocOp>(op)) {
      op->emitError(
          "unsupported use of an automatically planned shared buffer");
      return failure();
    }
  }
  return plan;
}
} // namespace mlir::triton::metax
