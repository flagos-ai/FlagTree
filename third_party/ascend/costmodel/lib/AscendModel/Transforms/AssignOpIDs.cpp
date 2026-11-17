//===- AssignOpIDs.cpp - Assign unique operation IDs ---------------------===//
//
// This file assigns unique IDs to AscendModel operations for tracking
// in the pipeline analysis.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_ASSIGNOPIDSPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

struct AssignOpIDsPass : public impl::AssignOpIDsPassBase<AssignOpIDsPass> {
  using AssignOpIDsPassBase::AssignOpIDsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int64_t nextId = 0;

    module.walk([&](Operation *op) {
      // Check if this is an AscendModel operation
      if (op->getDialect() && op->getDialect()->getNamespace() == "ascend") {

        // Set op_id attribute
        op->setAttr(
            "op_id",
            IntegerAttr::get(IntegerType::get(op->getContext(), 64), nextId++));
      }
    });

    // Store total operation count as module attribute
    module->setAttr(
        "ascend.total_ops",
        IntegerAttr::get(IntegerType::get(module.getContext(), 64), nextId));
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
