// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/MmaLayout.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;
#define GEN_PASS_DEF_COMMONIRRESOLVEBSMLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {
LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                             TensorLayoutRequirements &requirements,
                             const tile::BufferLayoutPlan *buffers) {
  llvm::MapVector<Value, Attribute> assignments;
  llvm::SetVector<OpOperand *> coveredUses;
  for (auto [dot, plan] : choices) {
    auto found = getBsmLayoutChain(dot.getB());
    if (failed(found))
      return failure();
    if (!*found)
      continue;
    auto chain = **found;
    Attribute encoding = plan.operandB;
    // Both selected candidates and existing MMA layouts are validated here.
    auto shared =
        buffers ? cast<ttg::MemDescType>(buffers->getType(chain.load.getSrc()))
                : ttg::MemDescType{};
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
    // Apply both plans after validation. Async copies are legalized at TTGIR exit.
    TTGIRLayoutAdapter adapter;
    buffers->apply();
    applyLayoutRequirements(module, requirements, adapter);
  }
};
} // namespace
} // namespace mlir::triton::metax
