// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "triton/Conversion/CommonIRToTTGIR/MmaLayout.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/SharedLayoutUtils.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIRRESOLVEBSMLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {
struct BsmDotLayout {
  triton::DotOp dot;
  DotLayoutPlan plan;
  BsmLayoutChain chain;
};

// Use the existing MMA layout or select a BSM-compatible default.
FailureOr<DotLayoutPlan>
inferBsmDotLayout(triton::DotOp dot, int computeCapability, int numWarps) {
  auto a = dot.getA().getType();
  auto b = dot.getB().getType();
  auto d = dot.getType();
  if (auto error =
          checkMacaDotPreconditions(dot, computeCapability, numWarps)) {
    dot.emitError("BSM layout preparation failed: ") << *error;
    return failure();
  }

  auto mma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(d.getEncoding());
  if (!mma) {
    // Gluon's C500 BSM rule: one K fragment, 32 carrier slots, then allocate
    // enough N warps to avoid repeating the B fragment. The remaining warps
    // and elements cover M. checkBsmLayout validates the physical mapping.
    auto atom = ttg::getMmaThreadShape();
    unsigned atomM = atom[0], atomN = atom[1], atomK = atom[2];
    int64_t k = a.getShape()[1];
    if (k % atomK || k / atomK > kBsmSlots || kBsmSlots % (k / atomK)) {
      dot.emitError("BSM K must fit one 32-slot MMA fragment");
      return failure();
    }
    unsigned elementsK = k / atomK;
    unsigned elementsN = kBsmSlots / elementsK;
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
    mma =
        buildMmaEncoding(dot.getContext(), computeCapability, {warpsM, warpsN},
                         {elementsM, elementsN, elementsK}, /*colMajor=*/0);
  }
  DotLayoutPlan plan = makeDotLayoutPlan(dot, mma);
  if (auto error =
          checkMacaDotLayout(dot, plan.mma, plan.operandA, plan.operandB,
                             computeCapability, numWarps)) {
    dot.emitError("MMA layout is not legal for this dot and target: ")
        << *error;
    return failure();
  }
  return plan;
}

LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<BsmDotLayout> choices,
                             TensorLayoutRequirements &requirements,
                             const SharedLayoutUpdates &buffers) {
  llvm::MapVector<Value, Attribute> assignments;
  llvm::SetVector<OpOperand *> coveredUses;
  for (auto [dot, plan, chain] : choices) {
    Attribute encoding = plan.operandB;
    // Both selected candidates and existing MMA layouts are validated here.
    auto shared = cast<ttg::MemDescType>(buffers.getType(chain.load.getSrc()));
    if (auto error = checkBsmLayout(chain, encoding, shared))
      return chain.perm.emitError(*error);
    auto requireLayout = [&](Value value) -> LogicalResult {
      auto [it, inserted] = assignments.insert({value, encoding});
      if (inserted || it->second == encoding)
        return success();
      return dot.emitError(
                 "shared split-load/BSM value requires conflicting layouts: ")
             << it->second << " versus " << encoding;
    };
    for (Value value : {chain.load.getResult(), chain.perm.getResult()})
      if (failed(requireLayout(value)))
        return failure();
    for (auto cvt : chain.conversions) {
      if (failed(requireLayout(cvt.getResult())))
        return failure();
      coveredUses.insert(&cvt->getOpOperand(0));
    }
    coveredUses.insert(&chain.perm->getOpOperand(0));
    coveredUses.insert(&dot->getOpOperand(1));
  }
  auto result = module.walk([&](Operation *op) {
    auto load = dyn_cast<ttg::LocalLoadOp>(op);
    if ((isa<ttg::BsmPermOp>(op) || (load && load.getMmaMode() == 2)) &&
        !assignments.contains(op->getResult(0))) {
      op->emitError(
          "split load/BSM must feed dot B through layout conversions only");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  // Packed raw values cannot be handed to arbitrary users via an ordinary
  // convert_layout. Validate all uses before publishing any requirements.
  for (auto [value, encoding] : assignments) {
    for (OpOperand &use : value.getUses()) {
      if (!coveredUses.contains(&use))
        return use.getOwner()->emitError(
            "BSM layout assignment does not support this use outside "
            "split load/BSM/dot B chains");
    }
  }
  for (auto [value, encoding] : assignments)
    requirements.assignments[value] = encoding;
  for (OpOperand *use : coveredUses)
    requirements.uses.push_back({use, assignments.lookup(use->get()), false});
  return success();
}

FailureOr<SharedLayoutUpdates>
collectBsmBufferUpdates(ModuleOp module, ArrayRef<BsmDotLayout> choices) {
  llvm::MapVector<Value, Attribute> encodings;
  for (auto [dot, plan, chain] : choices) {
    auto shared = chain.load.getSrc().getType();
    Value root = getSharedAllocation(chain.load.getSrc());
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
          ttg::getCTALayout(operand), shared.getElementType(),
          /*needTrans=*/false);
    }
    for (Value view = chain.load.getSrc(); view != root && encoding;
         view = getSharedViewSource(view))
      encoding = getSharedSourceEncoding(view, encoding);
    auto [it, inserted] = encodings.try_emplace(root, encoding);
    if (!encoding || (!inserted && it->second != encoding)) {
      module.emitError("BSM shared views require incompatible layouts");
      return failure();
    }
  }
  SharedLayoutUpdates buffers;
  for (auto [root, encoding] : encodings) {
    if (failed(collectSharedLayoutUpdates(root, encoding, buffers))) {
      module.emitError("BSM shared views require incompatible layouts");
      return failure();
    }
  }
  for (OpOperand *access : buffers.accesses) {
    if (!isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalDeallocOp,
             ttg::AsyncCopyGlobalToLocalOp>(access->getOwner())) {
      access->getOwner()->emitError("unsupported use of a BSM shared buffer");
      return failure();
    }
  }
  return buffers;
}

struct CommonIRResolveBsmLayoutsPass
    : impl::CommonIRResolveBsmLayoutsBase<CommonIRResolveBsmLayoutsPass> {
  using CommonIRResolveBsmLayoutsBase::CommonIRResolveBsmLayoutsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasBsm = false;
    module.walk([&](Operation *op) {
      auto load = dyn_cast<ttg::LocalLoadOp>(op);
      hasBsm |= isa<ttg::BsmPermOp>(op) || (load && load.getMmaMode() == 2);
    });
    if (!hasBsm)
      return;

    SmallVector<BsmDotLayout> choices;
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
      choices.push_back({dot, *plan, std::move(**b)});
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    auto buffers = collectBsmBufferUpdates(module, choices);
    if (failed(buffers))
      return signalPassFailure();
    TensorLayoutRequirements requirements;
    if (failed(collectBsmLayoutRequirements(module, choices, requirements,
                                            *buffers)))
      return signalPassFailure();
    for (auto [dot, plan, chain] : choices) {
      requirements.assignments[dot.getResult()] = plan.mma;
      requirements.uses.push_back(
          {&dot->getOpOperand(0), plan.operandA, false});
      requirements.uses.push_back({&dot->getOpOperand(2), plan.mma});
    }
    for (auto [value, encoding] : requirements.assignments) {
      if (auto explicitEncoding = tile::getExplicitValueEncoding(value)) {
        if (explicitEncoding != encoding) {
          value.getDefiningOp()->emitError(
              "BSM layout conflicts with set_layout");
          return signalPassFailure();
        }
      }
    }
    // Apply buffer and tensor updates after validation. Copies are legalized at
    // TTGIR exit.
    TTGIRLayoutAdapter adapter;
    buffers->apply();
    applyLayoutRequirements(module, requirements, adapter);
  }
};
} // namespace
} // namespace mlir::triton::metax
