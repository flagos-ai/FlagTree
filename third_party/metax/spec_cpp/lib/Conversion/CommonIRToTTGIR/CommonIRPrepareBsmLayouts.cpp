// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Conversion/CommonIRToTTGIR/BsmLayout.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIRPREPAREBSMLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {
FailureOr<DotLayoutPlan> inferBsmDotLayout(triton::DotOp dot,
                                         int computeCapability, int numWarps) {
  auto a = dot.getA().getType();
  auto b = dot.getB().getType();
  auto d = dot.getType();
  if (a.getRank() != 2 || b.getRank() != 2 || d.getRank() != 2 ||
      computeCapability < 80 || computeCapability >= 90 || numWarps <= 0 ||
      !llvm::isPowerOf2_64(numWarps)) {
    dot.emitError("BSM layout preparation requires rank-2 MACA v2 dots and a "
                  "power-of-two warp count");
    return failure();
  }
  for (auto type : {a, b, d}) {
    if (llvm::any_of(type.getShape(), [](int64_t dim) {
          return dim <= 0 || dim > std::numeric_limits<int>::max() ||
                 !llvm::isPowerOf2_64(dim);
        })) {
      dot.emitError("BSM layout preparation requires static power-of-two shapes");
      return failure();
    }
  }

  auto mma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(d.getEncoding());
  if (!mma) {
    // Gluon's C500 BSM rule: one K fragment, 32 carrier slots, then allocate
    // enough N warps to avoid repeating the B fragment. The remaining warps
    // and elements cover M. The physical checker below validates the mapping.
    constexpr unsigned atomM = 16, atomN = 16, atomK = 4, slots = 32;
    int64_t k = a.getShape()[1];
    if (k % atomK || k / atomK > slots || slots % (k / atomK)) {
      dot.emitError("BSM K must fit one 32-slot MMA fragment");
      return failure();
    }
    unsigned elementsK = k / atomK;
    unsigned elementsN = slots / elementsK;
    unsigned atomsN = llvm::divideCeil(b.getShape()[1], int64_t(atomN));
    unsigned warpsN = llvm::PowerOf2Ceil(llvm::divideCeil(atomsN, elementsN));
    if (warpsN > unsigned(numWarps) || numWarps % warpsN) {
      dot.emitError("BSM N fragment needs more warp partitions than num-warps");
      return failure();
    }
    unsigned warpsM = numWarps / warpsN;
    unsigned atomsM = llvm::divideCeil(d.getShape()[0], int64_t(atomM));
    unsigned elementsM = llvm::PowerOf2Ceil(llvm::divideCeil(atomsM, warpsM));
    if (warpsM * elementsM > atomsM) {
      dot.emitError("BSM warp grid cannot exactly cover the M fragment");
      return failure();
    }
    mma = ttg::MACAMmaEncodingAttr::get(
        dot.getContext(), 2, computeCapability % 10, {warpsM, warpsN},
        {elementsM, elementsN, elementsK}, /*colMajor=*/0,
        /*isATrans=*/false, /*isBTrans=*/false, {1, 1});
  }
  DotLayoutPlan plan{
      mma, ttg::DotOperandEncodingAttr::get(dot.getContext(), 0, mma,
                                          a.getElementType()),
      ttg::DotOperandEncodingAttr::get(dot.getContext(), 1, mma,
                                      b.getElementType())};
  if (failed(validateDotLayoutPlan(dot, plan, computeCapability, numWarps)))
    return failure();
  return plan;
}

FailureOr<tile::BufferLayoutPlan>
planBsmBuffers(ModuleOp module, ArrayRef<DotLayoutChoice> choices) {
  TTGIRBufferLayoutAdapter adapter;
  SmallVector<tile::BufferLayoutRequirement> requirements;
  for (auto [dot, plan] : choices) {
    auto found = getBsmLayoutChain(dot.getB());
    if (failed(found) || !*found)
      return failure();
    auto chain = **found;
    auto shared = chain.load.getSrc().getType();
    Value root = tile::getBufferLayoutRoot(chain.load.getSrc(), adapter);
    if (!root) {
      chain.load.emitError(
          "BSM requires a shared allocation or supported slot view");
      return failure();
    }
    Attribute encoding = shared.getEncoding();
    if (!root.getDefiningOp()->hasAttr("tle.gpu_layout")) {
      auto operand = cast<ttg::DotOperandEncodingAttr>(plan.operandB);
      auto mma = cast<ttg::MACAMmaEncodingAttr>(plan.mma);
      auto elements = mma.getElementsMNK();
      auto atom = mma.getThreadShape(false);
      // Guard the shared-encoding constructor's exact microtile assertions.
      if (shared.getRank() != 2 ||
          (!shared.getElementType().isF16() &&
           !shared.getElementType().isBF16()) ||
          shared.getShape()[0] % (atom[2] * elements[2]) ||
          shared.getShape()[1] % (atom[1] * elements[1])) {
        chain.load.emitError(
            "BSM shared shape must cover whole FP16/BF16 MMA microtiles");
        return failure();
      }
      encoding = ttg::SwizzledSharedEncodingAttr::get(
          module.getContext(), operand, shared.getShape(), {1, 0},
          ttg::getCTALayout(operand), shared.getElementType(), /*needTrans=*/false);
    }
    requirements.push_back({chain.load.getSrc(), encoding});
  }
  auto buffers = tile::planBufferLayouts(requirements, adapter);
  if (failed(buffers)) {
    module.emitError("BSM shared views require incompatible layouts");
    return failure();
  }
  for (OpOperand *access : buffers->accesses) {
    if (!isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalDeallocOp,
             ttg::AsyncCopyGlobalToLocalOp>(access->getOwner())) {
      access->getOwner()->emitError("unsupported use of a BSM shared buffer");
      return failure();
    }
  }
  return buffers;
}

struct CommonIRPrepareBsmLayoutsPass
    : impl::CommonIRPrepareBsmLayoutsBase<CommonIRPrepareBsmLayoutsPass> {
  using CommonIRPrepareBsmLayoutsBase::CommonIRPrepareBsmLayoutsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasBsm = false;
    module.walk([&](Operation *op) {
      auto load = dyn_cast<ttg::LocalLoadOp>(op);
      hasBsm |= isa<ttg::BsmPermOp>(op) || (load && load.getMmaMode() == 2);
    });
    if (!hasBsm)
      return;

    SmallVector<DotLayoutChoice> choices;
    WalkResult result = module.walk([&](triton::DotOp dot) {
      auto a = getBsmLayoutChain(dot.getA());
      auto b = getBsmLayoutChain(dot.getB());
      if (failed(a) || failed(b))
        return WalkResult::interrupt();
      if (*a) {
        dot.emitError("BSM is supported only for dot operand B");
        return WalkResult::interrupt();
      }
      if (!*b)
        return WalkResult::advance();
      auto plan = inferBsmDotLayout(dot, computeCapability, numWarps);
      if (failed(plan))
        return WalkResult::interrupt();
      choices.push_back({dot, *plan});
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    auto buffers = planBsmBuffers(module, choices);
    if (failed(buffers))
      return signalPassFailure();
    TensorLayoutRequirements requirements;
    if (failed(collectBsmLayoutRequirements(module, choices, requirements,
                                            &*buffers)))
      return signalPassFailure();
    for (auto [dot, plan] : choices) {
      requirements.assignments[dot.getResult()] = plan.mma;
      requirements.uses.push_back({&dot->getOpOperand(0), plan.operandA, false});
      requirements.uses.push_back({&dot->getOpOperand(2), plan.mma});
    }
    for (auto [value, encoding] : requirements.assignments) {
      if (auto explicitEncoding = tile::getExplicitValueEncoding(value)) {
        if (explicitEncoding != encoding) {
          value.getDefiningOp()->emitError("BSM layout conflicts with set_layout");
          return signalPassFailure();
        }
      }
    }
    // Plan both storage and registers before changing any types. Async copies
    // into the new shared layout are legalized by the normal pre-LLVM pass.
    TTGIRLayoutAdapter adapter;
    buffers->apply();
    applyLayoutRequirements(module, requirements, adapter);
  }
};
} // namespace
} // namespace mlir::triton::metax
