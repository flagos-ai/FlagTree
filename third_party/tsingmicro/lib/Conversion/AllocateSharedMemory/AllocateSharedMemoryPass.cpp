//===------------------- AllocateSharedMemoryPass.cpp ---------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "Analysis/Allocation.h"
#include "Analysis/Utility.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Analysis/Allocation.h"
#include "tsingmicro-tx81/Conversion/AllocateSharedMemory/Passes.h"

#define DEBUG_TYPE "allocate-shared-memory"

using namespace mlir;

namespace mlir::triton::alloc {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORY
#include "tsingmicro-tx81/Conversion/AllocateSharedMemory/Passes.h.inc"

} // namespace mlir::triton::alloc

namespace {
struct AllocateSharedMemory
    : public mlir::triton::alloc::impl::AllocateSharedMemoryBase<
          AllocateSharedMemory> {

  Operation *findAlignmentRestrictOpOperandBuffer(Value v) {
    auto op = v.getDefiningOp();
    assert(op && "Value has no defining op");
    if (isa<memref::AllocOp>(op))
      return op;
    assert(isa<memref::ExpandShapeOp>(op) || isa<memref::ReshapeOp>(op));
    return findAlignmentRestrictOpOperandBuffer(op->getOperand(0));
  }

  void handleTargetDependentAlignmentRequirements(ModuleOp &mod) {
    // NOTE: TX8 gemm instructions require 256-byte alignment for shared
    // memory operands.
    mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      funcOp.walk([&](Operation *op) {
        // FIXME: Abstract for other ops that may require alignment, e.g.
        if (!isa<mk::DotOp, mk::ReduceMaxOp, mk::ReduceMinOp, mk::ReduceSumOp>(
                op))
          return;
        for (auto user : op->getOperands()) {
          auto allocOp = findAlignmentRestrictOpOperandBuffer(user);
          assert(isa<memref::AllocOp>(allocOp));
          cast<memref::AllocOp>(allocOp).setAlignment(256);
        }
      });
      return WalkResult::skip();
    });
  }

  // Move the buffer allocations to just before their first user. (buffer and
  // user need to be in the same block)
  void relocateAllocationsToFirstUser(ModuleOp &mod) {
    DenseMap<Operation *, size_t> operationId;
    mod->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    OpBuilder builder(mod->getContext());
    DenseMap<Operation *, Operation *> bufferCanMove;
    mod->walk([&](Operation *op) {
      if (!isa<memref::AllocOp>(op))
        return;
      auto allocOp = cast<memref::AllocOp>(op);
      auto users = allocOp->getUsers();
      assert(!users.empty() && "tensor.empty should have no uses here");

      Operation *minIDUser = nullptr;
      size_t minID = std::numeric_limits<size_t>::max();
      for (auto user : users) {
        if (operationId[user] > minID)
          continue;
        minID = operationId[user];
        minIDUser = user;
      }
      assert(minIDUser && "There should be at least one user");
      if (!minIDUser->getParentOp()->isAncestor(allocOp))
        return;

      bufferCanMove.insert({allocOp, minIDUser});
    });

    for (auto [bufferOp, userOp] : bufferCanMove) {
      bufferOp->moveBefore(userOp);
    }
  }

  void setAllocationOffsetAttrs(ModuleOp &mod, MLIRContext *ctx,
                                triton::alloc::ModuleAllocation &allocation) {
    mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      auto *funcAllocation = allocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
        // Only handle memref::AllocOp
        if (!isa<memref::AllocOp>(op))
          return;
        int offset = -1;
        Value value = op->getResult(0);
        auto vBufferId = funcAllocation->getBufferId(value);
        if (vBufferId != triton::alloc::Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(vBufferId);

        if (offset == -1)
          return;
        if (op->hasAttr("allocation.offset"))
          return;
        op->setAttr("allocation.offset",
                    IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      });
      return WalkResult::skip();
    });
    mod->setAttr("triton_tsm.spm_use",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    handleTargetDependentAlignmentRequirements(mod);
    relocateAllocationsToFirstUser(mod);
    triton::alloc::ModuleAllocation allocation(mod);
    setAllocationOffsetAttrs(mod, ctx, allocation);
  }
};

} // namespace
