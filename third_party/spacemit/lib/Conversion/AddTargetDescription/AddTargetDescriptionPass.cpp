//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/AddTargetDescription/AddTargetDescription.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "add-target-description"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_ADDTARGETDESCRIPTION
#include "triton-shared/Conversion/AddTargetDescription/Passes.h.inc"
} // namespace mlir::triton

namespace {

class AddTargetDescriptionPass
    : public triton::impl::AddTargetDescriptionBase<AddTargetDescriptionPass> {

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    auto numThreadsAttr =
        moduleOp->getAttrOfType<mlir::IntegerAttr>("tt.num_threads");
    auto archAttr = moduleOp->getAttrOfType<mlir::StringAttr>("tt.arch_id");

    if (!numThreadsAttr || !archAttr) {
      llvm::outs()
          << "Error: Missing required attributes (tt.num_threads or tt.arch)\n";
      return signalPassFailure();
    }

    int32_t num_threads = numThreadsAttr.getInt();
    StringRef arch_id = archAttr.getValue();

    moduleOp->removeAttr("tt.num_threads");
    moduleOp->removeAttr("tt.arch_id");

    SmallVector<mlir::DataLayoutEntryInterface> dlEntries;
    dlEntries.push_back(
        mlir::DataLayoutEntryAttr::get(mlir::StringAttr::get(ctx, "arch_id"),
                                       mlir::StringAttr::get(ctx, arch_id)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "num_threads"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), num_threads)));

    auto deviceSpec = mlir::TargetDeviceSpecAttr::get(ctx, dlEntries);

    SmallVector<mlir::DataLayoutEntryInterface> systemEntries;
    systemEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "CPU"), deviceSpec));

    moduleOp->setAttr(mlir::DLTIDialect::kTargetSystemDescAttrName,
                      mlir::TargetSystemSpecAttr::get(ctx, systemEntries));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createAddTargetDescriptionPass() {
  return std::make_unique<AddTargetDescriptionPass>();
}
