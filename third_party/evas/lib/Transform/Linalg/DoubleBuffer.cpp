#include "epu/memory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Transform/common_utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include <cstddef>
#include <iterator>
#define GEN_PASS_DEF_DOUBLEBUFFER
#include "evas/Transform/Linalg/Passes.h.inc"

using namespace mlir;
namespace mlir::triton::ev {

namespace {

class DoubleBufferPass : public ::impl::DoubleBufferBase<DoubleBufferPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    // Walk through all scf.for operations in the function
    moduleOp.walk([&](scf::ForOp forOp) {
      // Skip if this loop is not suitable for optimization
      if (auto target = FindPrefetchTarget(forOp))
        applyPrefetching(forOp, target);
    });
  }

private:
  // Check if the loop is suitable for double buffering optimization
  Operation *FindPrefetchTarget(scf::ForOp forOp) {
    // Check if the loop has constant bounds and step
    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    auto step = forOp.getStep();

    // Check if lower bound is constant 0
    auto lowerBoundOp = lowerBound.getDefiningOp<arith::ConstantOp>();
    if (!lowerBoundOp)
      return nullptr;

    for (auto &op : llvm::reverse(forOp.getBody()->without_terminator())) {
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (callOp->hasAttr(mlir::ev::prefetchName) &&
            callOp->getAttrOfType<BoolAttr>(mlir::ev::prefetchName)
                .getValue()) {
          return &op;
        }
      }
    }

    return nullptr;
  }
  // Clone operations for prefetching before the loop
  void clonePrefetchOperations(mlir::scf::ForOp forOp, Operation *target,
                               mlir::OpBuilder &builder,
                               IRMapping &prefetchMap) {

    // Get the loop induction variable and its initial value
    Value inductionVar = forOp.getInductionVar();
    Value lowerBound = forOp.getLowerBound();
    builder.setInsertionPoint(forOp);
    // Map the induction variable to the lower bound for the prefetch operations
    prefetchMap.map(inductionVar, lowerBound);

    // Clone operations from the loop body for the first iteration
    Block &loopBody = forOp.getRegion().front();

    for (Operation &op :
         llvm::make_range(loopBody.begin(), std::next(target->getIterator()))) {
      // Skip the terminator
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      // Clone the operation with mapped operands
      builder.clone(op, prefetchMap);
    }
  }
  Value findToTensorUser(Operation *op) {
    // Iterate through all users of the operation's result
    for (Operation *user : op->getResult(0).getUsers()) {
      // Check if the user is a ToTensorOp
      if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
        return toTensorOp->getResult(0);
      }
    }
    return nullptr;
  }
  mlir::scf::ForOp createNewForOp(mlir::scf::ForOp forOp, Operation *target,
                                  mlir::OpBuilder &builder,
                                  mlir::IRMapping &prefetchMap,
                                  mlir::IRMapping &newForOpMap) {
    auto loc = forOp.getLoc();
    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    auto step = forOp.getStep();
    builder.setInsertionPoint(forOp);
    // TODO(wyann): suporrt multiple outputs
    // auto targetValue = target->getResult(0);
    // loopArgs.push_back(prefetchMap.lookup(targetValue));
    auto newForOp = builder.create<scf::ForOp>(loc, lowerBound, upperBound,
                                               step, forOp.getInitArgs());
    Block *newloopBody = newForOp.getBody();
    builder.setInsertionPointToStart(newloopBody);
    auto prefetchInductor = builder.create<mlir::arith::AddIOp>(
        loc, newForOp.getInductionVar().getType(), newForOp.getInductionVar(),
        newForOp.getStep());
    newForOpMap.map(forOp.getInductionVar(), prefetchInductor);
    auto condPreftchOutOfBound = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, prefetchInductor,
        newForOp.getUpperBound());
    auto ifOp =
        builder.create<mlir::scf::IfOp>(loc, condPreftchOutOfBound, false);
    auto ifOpThenBlock = ifOp.thenBlock();
    builder.setInsertionPointToStart(ifOpThenBlock);
    auto prefetchInductorMod = builder.create<mlir::arith::RemUIOp>(
        loc, prefetchInductor.getType(), prefetchInductor,
        builder.create<mlir::arith::ConstantIntOp>(
            loc, prefetchInductor.getType(), 2));

    auto doubleBufferSelectCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, prefetchInductorMod,
        builder.create<mlir::arith::ConstantIntOp>(
            loc, prefetchInductorMod.getType(), 0));
    // clone before target
    llvm::SmallDenseMap<Operation *, Value> pangAllocsMap;
    for (Operation &op : llvm::make_range(forOp.getBody()->begin(),
                                          std::next(target->getIterator()))) {
      builder.clone(op, newForOpMap);
      if (isa<memref::AllocOp>(op)) {
        auto pang_alloc = newForOpMap.lookup(op.getResult(0));
        auto ping_alloc = prefetchMap.lookup(op.getResult(0));
        auto dbSelect = builder.create<mlir::arith::SelectOp>(
            loc, doubleBufferSelectCond, ping_alloc, pang_alloc);
        // hoist the double buffer alloc out of loop
        newForOpMap.map(op.getResult(0), dbSelect);
        pangAllocsMap[&op] = pang_alloc;
        pang_alloc.getDefiningOp()->moveBefore(newForOp);
      }
    }
    builder.setInsertionPointAfter(ifOp);

    // adjust coveredOp inductor value
    newForOpMap.map(forOp.getInductionVar(), newForOp.getInductionVar());

    auto coveredInductorMod = builder.create<mlir::arith::RemUIOp>(
        loc, newForOp.getInductionVar().getType(), newForOp.getInductionVar(),
        builder.create<mlir::arith::ConstantIntOp>(
            loc, newForOp.getInductionVar().getType(), 2));

    auto selectCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, coveredInductorMod,
        builder.create<mlir::arith::ConstantIntOp>(
            loc, coveredInductorMod.getType(), 0));
    // adjust buffer selected op

    for (auto [op, pangAlloc] : pangAllocsMap) {
      auto toTensorUser = findToTensorUser(op);
      auto selectOp = builder.create<mlir::arith::SelectOp>(
        loc, selectCond, prefetchMap.lookup(op->getResult(0)),
        pangAlloc);
      newForOpMap.map(
          toTensorUser == nullptr ? op->getResult(0) : toTensorUser,
          builder.create<bufferization::ToTensorOp>(
              loc, memref::getTensorTypeFromMemRefType(selectOp.getType()), selectOp));
    }

    // clone after target
    for (Operation &op :
         llvm::make_range(std::next(target->getIterator()),
                          forOp.getBody()->without_terminator().end())) {

      // Clone the operation with mapped operands
      builder.clone(op, newForOpMap);
    }
    return newForOp;
  }

  void addFinalIterationCode(mlir::scf::ForOp forOp, mlir::scf::ForOp newForOp,
                             Operation *target, mlir::OpBuilder &builder,
                             mlir::IRMapping &newForOpMap) {
    builder.setInsertionPointAfter(newForOp);
    IRMapping finalIterMap;

    finalIterMap.map(forOp.getInductionVar(), newForOp.getUpperBound());
    // TODO(wyann): suporrt multiple outputs
    finalIterMap.map(target->getResult(0), newForOp.getResult(0));
    Block *loopBody = forOp.getBody();
    for (Operation &op :
         llvm::make_range(std::next(target->getIterator()),
                          loopBody->without_terminator().end())) {
      builder.clone(op, finalIterMap);
    }
  }
  // Apply the double buffering optimization to the loop
  void applyPrefetching(scf::ForOp forOp, Operation *target) {
    OpBuilder builder(forOp);
    MLIRContext *context = builder.getContext();
    IRMapping prefetchMap;
    IRMapping newForOpMap;
    // 1. Clone the first iteration before the loop
    clonePrefetchOperations(forOp, target, builder, prefetchMap);
    // 2. Create a new loop with adjusted bounds
    auto newForOp =
        createNewForOp(forOp, target, builder, prefetchMap, newForOpMap);
    // // 3. Add code for the final iteration after the loop
    // addFinalIterationCode(forOp, newForOp, target, builder, newForOpMap);
    // 4. Remove the original loop
    forOp.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createDoubleBufferPass() {
  return std::make_unique<DoubleBufferPass>();
}
} // namespace mlir::triton::ev
