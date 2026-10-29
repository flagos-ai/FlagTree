#include "triton/Conversion/CommonIRToTTGIR/TensorOps.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/Transforms/LayoutHints.h"
#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

namespace {

LogicalResult validateLayoutHints(ModuleOp module, bool &hasHints) {
  auto product = [](ArrayRef<unsigned> values) {
    return std::accumulate(values.begin(), values.end(), uint64_t{1},
                           std::multiplies<uint64_t>());
  };

  WalkResult result = module.walk([&](tile::SetLayoutOp op) {
    hasHints = true;
    Attribute encoding = op.getLayout();
    Attribute parent = encoding;
    while (auto slice = dyn_cast<ttg::SliceEncodingAttr>(parent))
      parent = slice.getParent();
    if (auto operand = dyn_cast<ttg::DotOperandEncodingAttr>(parent))
      parent = operand.getParent();
    auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(parent);
    auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(parent);
    auto linear = dyn_cast<ttg::LinearEncodingAttr>(parent);
    if (!blocked && !mma && !linear) {
      op.emitOpError("MetaX supports blocked, linear or MACA MMA layouts, their dot "
                     "operands and slices");
      return WalkResult::interrupt();
    }
    if (mma) {
      auto powersOfTwo = [](ArrayRef<unsigned> values) {
        return llvm::all_of(values,
                            [](unsigned v) { return llvm::isPowerOf2_64(v); });
      };
      if (mma.getVersionMajor() != 2 || mma.getWarpsPerCTA().size() != 2 ||
          mma.getElementsMNK().size() != 3 ||
          !powersOfTwo(mma.getWarpsPerCTA()) ||
          !powersOfTwo(mma.getElementsMNK()) || mma.getColMajor() > 1 ||
          mma.getIsATrans() || mma.getIsBTrans() ||
          mma.getElementsStride() != ArrayRef<unsigned>({1, 1})) {
        op.emitOpError("requires a rank-2 MACA MMA v2 layout with power-of-two "
                       "warp/element counts, unit strides and no LDS transpose");
        return WalkResult::interrupt();
      }
    }
    if (ttg::getCTALayout(encoding).getRank() != op.getType().getRank()) {
      op.emitOpError("layout rank must match tensor rank");
      return WalkResult::interrupt();
    }
    auto threadsPerWarp = linear
                              ? linear.getLinearLayout().getInDimSize(
                                    StringAttr::get(op.getContext(), "lane"))
                              : blocked ? product(blocked.getThreadsPerWarp())
                                        : 64;
    auto warpsPerCTA = linear
                           ? linear.getLinearLayout().getInDimSize(
                                 StringAttr::get(op.getContext(), "warp"))
                           : product(blocked ? blocked.getWarpsPerCTA()
                                             : mma.getWarpsPerCTA());
    auto numCTAs = ttg::getCTALayout(parent).getLinearLayout().getInDimSize(
        StringAttr::get(op.getContext(), "block"));
    if (threadsPerWarp != ttg::TritonGPUDialect::getThreadsPerWarp(module) ||
        warpsPerCTA != ttg::lookupNumWarps(op) ||
        numCTAs != ttg::lookupNumCTAs(op)) {
      op.emitOpError("layout must match the module's threads-per-warp, "
                     "num-warps and num-ctas");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

bool isGeneralTensorLayout(Attribute encoding) {
  while (auto slice = dyn_cast<ttg::SliceEncodingAttr>(encoding))
    encoding = slice.getParent();
  return isa<ttg::BlockedEncodingAttr, ttg::LinearEncodingAttr>(encoding);
}

bool canPropagateTo(Value value, Attribute encoding) {
  if (isGeneralTensorLayout(encoding))
    return true;
  auto type = cast<RankedTensorType>(value.getType());
  if (isa<triton::PointerType>(type.getElementType()))
    return false;
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return true;
  // Global memory keeps its coalesced layout; local_load can produce dot A/B.
  if (isa<triton::LoadOp>(producer))
    return false;
  if (isa<triton::DotOp>(producer))
    return isa<ttg::MACAMmaEncodingAttr>(encoding);
  return true;
}

void completeResultEncoding(Operation *op) {
  if (isa<tile::SetLayoutOp, ttg::ConvertLayoutOp, triton::DotOp>(op) ||
      op->getNumResults() != 1)
    return;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getEncoding())
    return;
  bool hasEncodedOperand = llvm::any_of(op->getOperandTypes(), [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.getEncoding();
  });
  if (!hasEncodedOperand)
    return;
  // Propagation can stop at a use (e.g. expand_dims on a different axis).
  // Complete its result before reconciling inputs, keeping the producer layout.
  auto encoding = ttg::getDefaultBlockedEncoding(
      op->getContext(), resultType.getShape(), ttg::lookupNumWarps(op),
      ttg::TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()),
      ttg::lookupNumCTAs(op));
  if (inferSrcEncoding(op, encoding))
    op->getResult(0).setType(resultType.cloneWithEncoding(encoding));
}

Attribute getMemoryOperandEncoding(Operation *op) {
  if (!isa<triton::LoadOp, triton::StoreOp,
           ttg::AsyncCopyGlobalToLocalOp>(op))
    return {};

  // Tensor-of-pointer accesses share one encoding across all tensor operands.
  // Scalar/block pointers and the copy's shared destination are not this group.
  auto ptrType = dyn_cast<RankedTensorType>(getMemAccessPtr(op).getType());
  if (!ptrType)
    return {};
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return cast<RankedTensorType>(load.getType()).getEncoding();

  // Preserve the store value's general layout, otherwise prefer the pointer.
  // An async copy has no tensor result; its mask/other must follow its pointer.
  auto type = ptrType;
  if (auto store = dyn_cast<triton::StoreOp>(op))
    type = cast<RankedTensorType>(store.getValue().getType());
  if (type.getEncoding() && isGeneralTensorLayout(type.getEncoding()))
    return type.getEncoding();
  if (ptrType.getEncoding() && isGeneralTensorLayout(ptrType.getEncoding()))
    return ptrType.getEncoding();
  return ttg::getDefaultBlockedEncoding(
      op->getContext(), type.getShape(), ttg::lookupNumWarps(op),
      ttg::TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()),
      ttg::lookupNumCTAs(op));
}

Attribute getRequiredOperandEncoding(Operation *op, unsigned operandIndex) {
  // These operations intentionally allow different input and output layouts.
  if (isa<tile::SetLayoutOp, ttg::ConvertLayoutOp>(op))
    return {};

  // A/B layouts are supplied explicitly. Only C must match the dot result.
  if (auto dot = dyn_cast<triton::DotOp>(op))
    return operandIndex == 2 ? dot.getType().getEncoding() : Attribute{};

  if (op->getNumResults() != 1)
    return {};
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.getEncoding())
    return {};
  return inferSrcEncoding(op, resultType.getEncoding());
}

void insertRequiredLayoutConversions(ModuleOp module) {
  // Keep the producer's layout and convert at each use that needs another one.
  module.walk([&](Operation *op) {
    completeResultEncoding(op);
    Attribute memoryEncoding = getMemoryOperandEncoding(op);
    OpBuilder builder(op);
    for (OpOperand &operand : op->getOpOperands()) {
      auto type = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!type)
        continue;
      Attribute requiredEncoding = memoryEncoding
                                       ? memoryEncoding
                                       : getRequiredOperandEncoding(
                                             op, operand.getOperandNumber());
      if (!requiredEncoding || type.getEncoding() == requiredEncoding)
        continue;
      auto targetType = type.cloneWithEncoding(requiredEncoding);
      auto convert = builder.create<ttg::ConvertLayoutOp>(
          op->getLoc(), targetType, operand.get());
      operand.set(convert.getResult());
    }
  });
}

LogicalResult validateDotLayouts(ModuleOp module) {
  WalkResult result = module.walk([&](triton::DotOp dot) {
    Attribute mma = dot.getType().getEncoding();
    Attribute a = dot.getA().getType().getEncoding();
    Attribute b = dot.getB().getType().getEncoding();
    auto isMmaOperand = [](Attribute encoding) {
      auto operand = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(encoding);
      return operand && isa<ttg::MACAMmaEncodingAttr>(operand.getParent());
    };
    if (!isa_and_nonnull<ttg::MACAMmaEncodingAttr>(mma) &&
        !isMmaOperand(a) && !isMmaOperand(b))
      return WalkResult::advance();
    if (!isa_and_nonnull<ttg::MACAMmaEncodingAttr>(mma)) {
      dot.emitError("MACA dot layout hints require an MMA layout for the "
                    "result or accumulator");
      return WalkResult::interrupt();
    }
    if (!isMmaOperand(a) || !isMmaOperand(b)) {
      dot.emitError("MMA layout hints require explicit dot-operand layouts "
                    "for A and B");
      return WalkResult::interrupt();
    }
    auto target = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    int capability;
    if (!target ||
        target.getValue().split(':').second.getAsInteger(10, capability)) {
      dot.emitError("requires a target compute capability for MMA validation");
      return WalkResult::interrupt();
    }
    DotLayoutPlan plan{mma, a, b};
    return failed(validateDotLayoutPlan(dot, plan, capability,
                                        ttg::lookupNumWarps(dot)))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult markExplicitMemoryLayouts(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (!getMemAccessPtr(op))
      return WalkResult::advance();
    Attribute encoding;
    if (failed(tile::inferExplicitMemoryEncoding(op, encoding)))
      return WalkResult::interrupt();
    if (encoding)
      tile::setExplicitMemoryEncoding(op, encoding);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace

LogicalResult applyCommonIRLayoutHints(ModuleOp module) {
  bool hasHints = false;
  if (failed(validateLayoutHints(module, hasHints)))
    return failure();
  if (!hasHints)
    return success();

  auto convertLayout = [](OpBuilder &builder, Location loc, Value source,
                          RankedTensorType type) -> Value {
    return builder.create<ttg::ConvertLayoutOp>(loc, type, source).getResult();
  };
  const tile::LayoutHintCallbacks callbacks{
      inferSrcEncoding, inferDstEncoding, getTiedArgs, canPropagateTo,
      convertLayout, canLowerTensorTileInEncoding};
  if (failed(tile::propagateLayoutHints(module, callbacks)))
    return failure();
  if (failed(validateDotLayouts(module)))
    return failure();

  insertRequiredLayoutConversions(module);
  return markExplicitMemoryLayouts(module);
}

} // namespace mlir::triton::metax
