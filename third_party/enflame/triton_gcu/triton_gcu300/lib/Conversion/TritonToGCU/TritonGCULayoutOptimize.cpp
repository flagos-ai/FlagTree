/*
 * Copyright 2020 - 2022 Enflame.All Rights Reserved.
 *
 */

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
namespace mlir {
#define GEN_PASS_DEF_TRITONGCULAYOUTOPTIMIZEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "triton-gcu-data-layout-optimize"
namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct TritonGCULayoutOptimizePass
    : public mlir::impl::TritonGCULayoutOptimizePassBase<
          TritonGCULayoutOptimizePass> {
  using Base::Base;

  void runOnOperation() override;
  void RefineGcuLoadStoreLayout();
  void reWriteGcuStoreLayout(triton::gcu::StoreOp store);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect,
                triton::TritonDialect, mlir::triton::gcu::TritonGCUDialect>();
  }
};

void TritonGCULayoutOptimizePass::reWriteGcuStoreLayout(
    triton::gcu::StoreOp store) {
  auto src = store.getValue().getDefiningOp();
  if (auto convetLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(src)) {
    auto users = src->getUsers();
    auto userNumber = std::distance(users.begin(), users.end());
    // only one user delete and fusion convert layout to store
    if (userNumber == 1) {
      auto trueSrc = convetLayout.getSrc();
      convetLayout.getResult().replaceAllUsesWith(trueSrc);
      convetLayout.erase();
    }
  }
}

void TritonGCULayoutOptimizePass::RefineGcuLoadStoreLayout() {
  auto trionModule = getOperation();
  llvm::SmallVector<triton::gcu::StoreOp> storeList;
  trionModule.walk(
      [&](triton::gcu::StoreOp store) { storeList.push_back(store); });
  for (auto &store : storeList) {
    reWriteGcuStoreLayout(store);
  }
}
} // namespace
using namespace mlir;
void TritonGCULayoutOptimizePass::runOnOperation() {
  RefineGcuLoadStoreLayout();
}
