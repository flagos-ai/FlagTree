#include "triton/Conversion/CommonIRToTTGIR/CommonIRCoalesceAsyncCopy.h"
#include "LayoutConversionUtils.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/IR/Matchers.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRLowerTensorTiles.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorTileAxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIRCOALESCEASYNCCOPY
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"
namespace {
bool isAsyncCopyDestinationSupported(gpu::MemDescType destination) {
  auto shared =
      dyn_cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  // AsyncCopyGlobalToLocalOpConversion drops allocShape for swizzled views.
  return shared && (shared.getMaxPhase() == 1 ||
                    destination.getAllocShape() == destination.getShape());
}

gpu::MemDescType getAsyncCopyLinearDestination(gpu::MemDescType destination) {
  // Match noSwizzleDstTy in LoadStoreOpToLLVM.cpp.
  auto shared =
      cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  auto linear = gpu::SwizzledSharedEncodingAttr::get(
      destination.getContext(), shared.getVec(), 1, 1, shared.getOrder(),
      shared.getCTALayout());
  return gpu::MemDescType::get(
      destination.getShape(), destination.getElementType(), linear,
      destination.getMemorySpace(), destination.getMutableMemory(),
      destination.getAllocShape());
}

unsigned getAsyncCopyVectorSize(RankedTensorType source,
                                gpu::MemDescType destination,
                                const AxisInfo *sourceInfo) {
  auto layout = gpu::toLinearEncoding(source);
  unsigned dim = layout.getOrder()[0];
  unsigned bits = destination.getElementTypeBitWidth();
  unsigned contiguity = 1;
  if (sourceInfo && sourceInfo->getRank() == source.getRank()) {
    unsigned alignment = std::max<int64_t>(
        sourceInfo->getDivisibility(dim) / std::max(bits / 8, 1u), 1);
    contiguity = std::min<unsigned>(alignment, sourceInfo->getContiguity(dim));
  }
  unsigned vec =
      std::min({contiguity, layout.getContigPerThread()[dim], 128 / bits});
  auto shared =
      cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  if (shared.getMaxPhase() > 1)
    vec = std::min(vec, shared.getVec());
  return vec;
}

FailureOr<unsigned>
getAsyncCopyMaxVectorSize(gpu::AsyncCopyGlobalToLocalOp copy,
                          gpu::MemDescType destination,
                          ModuleAxisInfoAnalysis &axisInfo, bool emitErrors) {
  auto reject = [&](StringRef message) -> FailureOr<unsigned> {
    if (emitErrors)
      copy.emitOpError(message);
    return failure();
  };
  auto shared =
      cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  unsigned dim = shared.getOrder().front();
  unsigned maxVec = 128 / destination.getElementTypeBitWidth();
  if (Value mask = copy.getMask()) {
    auto *info = axisInfo.getAxisInfo(mask);
    if (!info || info->getRank() != destination.getRank())
      return reject("cannot prove async-copy mask uniformity");
    // Lowering permutes the addresses and leaves the masks in their original order.
    if (shared.getMaxPhase() > 1 &&
        info->getConstancy(dim) < destination.getShape()[dim])
      return reject(
          "swizzled shared layout requires an async-copy mask uniform along "
          "the swizzled axis");
    maxVec = std::min<int64_t>(maxVec, info->getConstancy(dim));
  }
  if (Value other = copy.getOther()) {
    while (auto convert = other.getDefiningOp<gpu::ConvertLayoutOp>())
      other = convert.getSrc();
    if (!matchPattern(other, m_Zero()) &&
        !matchPattern(other, m_PosZeroFloat()))
      return reject("MetaX async copy requires zero fill");
  }
  return maxVec;
}

bool isAsyncCopyLayoutCompatible(RankedTensorType source,
                                 gpu::MemDescType destination, unsigned vec) {
  if (!isa<gpu::BlockedEncodingAttr>(source.getEncoding()) ||
      !isAsyncCopyDestinationSupported(destination) ||
      source.getShape() != destination.getShape() || !llvm::isPowerOf2_32(vec))
    return false;

  auto *ctx = source.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto block = StringAttr::get(ctx, "block");
  auto offset = StringAttr::get(ctx, "offset");
  auto srcLayout = gpu::toLinearLayout(source);
  auto dstLayout =
      gpu::toLinearLayout(getAsyncCopyLinearDestination(destination));
  auto cvt = srcLayout.invertAndCompose(dstLayout);
  if (!cvt.isTrivialOver({block}))
    return false;

  // The first vec registers must be consecutive in shared, in the same order
  // as the vector global read. Remaining register bits select other copies.
  unsigned regBits = llvm::Log2_32(vec);
  if (cvt.getInDimSizeLog2(reg) < regBits)
    return false;
  for (unsigned bit = 0; bit < regBits; ++bit)
    if (cvt.getBasis(reg, bit, offset) != (1u << bit))
      return false;

  for (unsigned bit = 0; bit < cvt.getInDimSizeLog2(lane); ++bit) {
    // Broadcast lanes are suppressed by the lowering's thread predicate.
    if (llvm::all_of(srcLayout.getBasis(lane, bit),
                     [](int32_t basis) { return basis == 0; }))
      continue;
    if (cvt.getBasis(lane, bit, offset) != (vec << bit))
      return false;
  }
  return true;
}

gpu::BlockedEncodingAttr chooseAsyncCopyLayout(RankedTensorType source,
                                               gpu::MemDescType destination,
                                               const AxisInfo *sourceInfo,
                                               unsigned maxVec) {
  auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(source.getEncoding());
  auto shared =
      dyn_cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  if (!blocked || !shared || !isAsyncCopyDestinationSupported(destination) ||
      source.getShape() != destination.getShape())
    return {};
  auto compatible = [&](RankedTensorType candidate) {
    unsigned vec = getAsyncCopyVectorSize(candidate, destination, sourceInfo);
    return vec <= maxVec &&
           isAsyncCopyLayoutCompatible(candidate, destination, vec);
  };
  if (compatible(source))
    return blocked;

  for (unsigned width = 128 / destination.getElementTypeBitWidth(); width;
       width /= 2) {
    SmallVector<unsigned> sizePerThread(source.getRank(), 1);
    sizePerThread[shared.getOrder()[0]] = width;
    auto encoding = gpu::BlockedEncodingAttr::get(
        source.getContext(), source.getShape(), sizePerThread,
        shared.getOrder(), product(blocked.getWarpsPerCTA()),
        product(blocked.getThreadsPerWarp()), blocked.getCTALayout());
    auto candidate = source.cloneWithEncoding(encoding);
    if (compatible(candidate))
      return encoding;
  }
  return {};
}

} // namespace

ttg::BlockedEncodingAttr
getAsyncCopySourcePreference(ttg::AsyncCopyGlobalToLocalOp copy,
                             ModuleAxisInfoAnalysis &axisInfo) {
  Value source = copy.getSrc();
  while (auto convert = source.getDefiningOp<ttg::ConvertLayoutOp>())
    source = convert.getSrc();
  if (!isStaticMctleTile(source.getDefiningOp()) ||
      tile::getExplicitMemoryEncoding(copy) ||
      llvm::any_of(copy->getOperands(), [](Value v) {
        return bool(tile::getExplicitValueEncoding(v));
      }))
    return {};
  auto module = copy->getParentOfType<ModuleOp>();
  auto type = copy.getSrc().getType();
  return ttg::buildCoalescedEncoding(
      module.getContext(), axisInfo, copy, ttg::lookupNumWarps(copy),
      ttg::TritonGPUDialect::getThreadsPerWarp(module),
      ttg::getCTALayout(type.getEncoding()), ttg::getShapePerCTA(type));
}

namespace {
Attribute selectAsyncCopySourceLayout(ttg::AsyncCopyGlobalToLocalOp copy,
                                      ttg::MemDescType destination,
                                      ModuleAxisInfoAnalysis &axisInfo,
                                      bool emitErrors) {
  auto reject = [&](StringRef message) -> Attribute {
    if (emitErrors)
      copy.emitOpError(message);
    return {};
  };
  auto source = copy.getSrc().getType();
  if (Attribute preferred = getAsyncCopySourcePreference(copy, axisInfo))
    source = source.cloneWithEncoding(preferred);
  if (!isa<ttg::BlockedEncodingAttr>(source.getEncoding()) ||
      !isa<ttg::SwizzledSharedEncodingAttr>(destination.getEncoding()))
    return reject("requires blocked source and swizzled shared destination");
  if (!isAsyncCopyDestinationSupported(destination))
    return reject("native MetaX async copy does not support swizzled "
                  "shared views with a different allocation shape");
  auto maxVec =
      getAsyncCopyMaxVectorSize(copy, destination, axisInfo, emitErrors);
  if (failed(maxVec))
    return {};
  auto encoding = chooseAsyncCopyLayout(
      source, destination, axisInfo.getAxisInfo(copy.getSrc()), *maxVec);
  if (!encoding)
    return reject("cannot legalize lane-to-shared addresses and mask vector "
                  "width for MetaX async copy");
  return encoding;
}

} // namespace

bool canWriteMmaSharedLayout(gpu::AsyncCopyGlobalToLocalOp copy,
                             gpu::MemDescType destination,
                             ModuleAxisInfoAnalysis &axisInfo) {
  Attribute source =
      selectAsyncCopySourceLayout(copy, destination, axisInfo, false);
  if (!source)
    return false;
  // Only change the buffer for MMA if the actual copy can write at least a
  // word.
  auto type = copy.getSrc().getType().cloneWithEncoding(source);
  unsigned vec = getAsyncCopyVectorSize(type, destination,
                                        axisInfo.getAxisInfo(copy.getSrc()));
  return vec * destination.getElementTypeBitWidth() >= 32;
}

namespace {
struct CommonIRCoalesceAsyncCopy
    : impl::CommonIRCoalesceAsyncCopyBase<CommonIRCoalesceAsyncCopy> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
    module.walk(
        [&](ttg::AsyncCopyGlobalToLocalOp copy) { copies.push_back(copy); });
    if (copies.empty())
      return;
    ModuleAxisInfoAnalysis axisInfo(module, addTensorTileAxisInfoVisitors);
    SmallVector<Attribute> encodings;
    for (auto copy : copies) {
      auto destination = copy.getResult().getType();
      Attribute encoding =
          selectAsyncCopySourceLayout(copy, destination, axisInfo, true);
      if (!encoding)
        return signalPassFailure();
      encodings.push_back(encoding);
    }
    // Coalescing examines other uses of the pointer. Finish the analysis before
    // inserting conversions so all queries see the same IR.
    for (auto [copy, encoding] : llvm::zip(copies, encodings)) {
      for (OpOperand &operand : copy->getOpOperands()) {
        auto type = dyn_cast<RankedTensorType>(operand.get().getType());
        if (type)
          operand.set(convertLayout(operand.get(),
                                    type.cloneWithEncoding(encoding), copy));
      }
    }
  }
};
} // namespace
} // namespace mlir::triton::metax
