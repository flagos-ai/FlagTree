#include "epu/memory.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DEF_INSERTDEALLOCOP
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;
struct InsertDeallocOpPass
    : public ::impl::InsertDeallocOpBase<InsertDeallocOpPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration() || func.getSymName() != "kernel")
        continue;
      Liveness liveness(func);
      SmallVector<OpResult> allocValues;
      func.walk([&](Operation *op) {
        if (mlir::ev::isMemoryAllocOp(op)) {
          allocValues.push_back(op->getResult(0));
        } else if (mlir::ev::isSubkernelBufferOp(op)) {
          for (auto result : op->getResults()) {
            allocValues.push_back(result);
          }
        }
      });
      OpBuilder builder(func.getBody());
      for (auto allocValue : allocValues) {
        if (Operation *lastUser =
                findLastUserInOneBlock(allocValue, liveness)) {
          builder.setInsertionPointAfter(lastUser);
          if (auto memrefAlloc =
                  dyn_cast<memref::AllocOp>(allocValue.getOwner())) {
            builder.create<memref::DeallocOp>(lastUser->getLoc(), memrefAlloc);
          } else {
            builder.create<bufferization::DeallocTensorOp>(lastUser->getLoc(),
                                                           allocValue);
          }
        }
      }
    }
  }

private:
  bool obtain(Value src, Operation *dst, Block *block) {
    for (Operation *useOp : src.getUsers()) {
      useOp = block->findAncestorOpInBlock(*useOp);
      if (!useOp)
        continue;
      if (useOp == dst) {
        return true;
      }
      if ((isa<bufferization::ToTensorOp, bufferization::ToBufferOp,
               memref::LoadOp, ViewLikeOpInterface>(useOp) ||
           useOp->getDialect()->getNamespace() ==
               tensor::TensorDialect::getDialectNamespace()) &&
          obtain(useOp->getResult(0), dst, block)) {
        return true;
      }
    }
    return false;
  }

  // liveness analysis within multiple blocks is not supported for now
  Operation *findLastUserInOneBlock(OpResult allocValue, Liveness liveness) {
    Block *block = allocValue.getOwner()->getBlock();
    auto liveInfo = liveness.getLiveness(block);
    assert(!liveInfo->isLiveOut(allocValue) &&
           "liveness within multiple blocks is not supported");
    for (Operation &curOp : llvm::reverse(llvm::make_range(
             allocValue.getOwner()->getIterator(), block->without_terminator().end()))) {
      if (obtain(allocValue, &curOp, block)) {
        return &curOp;
      }
    }
    return nullptr;
  }
};
} // namespace
std::unique_ptr<Pass> createInsertDeallocOpPass() {
  return std::make_unique<InsertDeallocOpPass>();
}
} // namespace mlir::triton::ev