/*
 * Copyright 2020 - 2022 Enflame.All Rights Reserved.
 *
 */

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

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

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
namespace mlir {
#define GEN_PASS_DEF_TRITONGPUTOTRITONGCUPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir
#define DEBUG_TYPE "triton-gpu-to-triton-gcu"
namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct TritonGPUToTritonGCUPass
    : public mlir::impl::TritonGPUToTritonGCUPassBase<
          TritonGPUToTritonGCUPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TritonDialect, mlir::triton::gcu::TritonGCUDialect>();
  }
  void rewriteLoalLoad();
  void runOnOperation() override;
};

void TritonGPUToTritonGCUPass::rewriteLoalLoad() {
  auto trionModule = getOperation();
  llvm::SmallVector<triton::gpu::LocalAllocOp> localAllocList;
  trionModule.walk([&](triton::gpu::LocalAllocOp alloc) {
    localAllocList.push_back(alloc);
  });
  for (auto &alloc : localAllocList) {
    for (auto user : alloc->getUsers()) {
      if (llvm::isa_and_nonnull<triton::gpu::LocalLoadOp>(user)) {
        OpBuilder rewriter(user);
        auto localLoad = cast<triton::gpu::LocalLoadOp>(user);
        auto convert = rewriter.create<ConvertLayoutOp>(
            user->getLoc(), localLoad.getType(), alloc.getSrc());
        localLoad.getResult().replaceAllUsesWith(convert.getResult());
        localLoad.erase();
      } else if (llvm::isa_and_nonnull<triton::gpu::LocalDeallocOp>(user)) {
        user->erase();
      } else {
        user->dump();
        trionModule.dump();
        llvm::report_fatal_error("please check IR can't rewrite");
      }
      alloc.erase();
    }
  }
}

} // namespace
using namespace mlir;
void TritonGPUToTritonGCUPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "TritonGPUToTritonGCUPass\n");
  rewriteLoalLoad();
}
