#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#define GEN_PASS_DEF_REMOVELOOPITERARGSWITHMEMREFTYPE
#include "evas/Transform/Linalg/Passes.h.inc"
namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;

scf::ForOp isInForLoop(Operation *op) {
  // Walk up the parent operations to find a scf::ForOp
  Operation *currentOp = op;
  while (currentOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
      return forOp;
    }
    currentOp = currentOp->getParentOp();
  }
  return nullptr;
}

Value findInitArgFromIterArg(Value iterArg, scf::ForOp forOp) {
  // Get the position of iterArg in the loop's region iter args
  int position = -1;
  for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (arg == iterArg) {
      position = idx;
      break;
    }
  }

  // If iterArg not found in region iter args, return nullptr
  if (position == -1)
    return nullptr;

  // Return the corresponding init arg at the same position
  return forOp.getInitArgs()[position];
}

LogicalResult replaceMemRefIterArgs(scf::ForOp forOp, BufferOriginAnalysis &bufferOriginAnalysis) {
  // Track which init args are memref type and have different yield values
  llvm::SmallVector<bool> shouldReplace(forOp.getNumRegionIterArgs(), false);
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Check each init arg and corresponding yield value
  for (auto [i, initAndYield] :
       llvm::enumerate(llvm::zip(forOp.getInitArgs(), yieldOp.getOperands()))) {
    auto [initArg, yieldVal] = initAndYield;

    // Check if init arg is memref type
    if (isa<MemRefType>(initArg.getType())) {
      // Check if yield value is different from init arg
      shouldReplace[i] = !bufferOriginAnalysis.isSameAllocation(initArg, yieldVal).value();
    }
  }

  // If no replacement needed, return failure
  if (llvm::none_of(shouldReplace, [](bool replace) { return replace; })) {
    return failure();
  }

  // Replace uses of yield values with iter args where needed
  for (auto [i, vals] : llvm::enumerate(llvm::zip(yieldOp.getOperands(),
                                                  forOp.getRegionIterArgs(),
                                                  forOp.getInitArgs()))) {
    if (shouldReplace[i]) {
      auto [yieldVal, iterArg, initArg] = vals;
      // Replace all uses of yield value with iter arg within the loop body
      yieldVal.replaceAllUsesWith(initArg);
      iterArg.replaceAllUsesWith(initArg);
    }
  }
  return success();
}

LogicalResult removeUnusedIterArgs(scf::ForOp forOp) {
  // Track which iter args are used
  llvm::SmallVector<bool> isIterArgUsed(forOp.getNumRegionIterArgs(), false);

  // Check usage of each iter arg in the loop body
  forOp.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (operand == iterArg) {
          isIterArgUsed[idx] = true;
        }
      }
    }
  });

  // If all iter args are used, nothing to do
  if (llvm::all_of(isIterArgUsed, [](bool used) { return used; })) {
    return failure();
  }

  // Collect used iter args, init args and yield operands
  SmallVector<Value> newInitArgs;
  SmallVector<Value> newIterArgs;
  SmallVector<Value> newYieldOperands;
  SmallVector<Value> newOutputs;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (auto i : llvm::seq<unsigned>(0, forOp.getNumRegionIterArgs())) {
    if (isIterArgUsed[i]) {
      newInitArgs.push_back(forOp.getInitArgs()[i]);
      newIterArgs.push_back(forOp.getRegionIterArgs()[i]);
      newYieldOperands.push_back(yieldOp.getOperands()[i]);
      newOutputs.push_back(forOp->getResult(i));
    } else {
      forOp->getResult(i).replaceAllUsesWith(forOp.getInitArgs()[i]);
    }
  }

  // Create new ForOp with only used iter args
  OpBuilder builder(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  // Clone the body of old ForOp into new ForOp using IRMapping
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip(newIterArgs, newForOp.getRegionIterArgs())) {
    mapping.map(oldArg, newArg);
  }

  // Remove the automatically created terminator if it exists
  Block *newBody = newForOp.getBody();
  if (!newBody->empty() && isa<scf::YieldOp>(newBody->getTerminator())) {
    newBody->back().erase();
  }
  
  builder.setInsertionPointToStart(newBody);

  // Clone all operations except the terminator
  for (auto &op : forOp.getBody()->without_terminator()) {
    builder.clone(op, mapping);
  }

  // Update yield operands by mapping them through IRMapping
  SmallVector<Value> mappedYieldOperands;
  for (Value yieldOp : newYieldOperands) {
    mappedYieldOperands.push_back(mapping.lookupOrDefault(yieldOp));
  }
  builder.create<scf::YieldOp>(forOp.getLoc(), mappedYieldOperands);

  // Replace old ForOp with new one
  for (auto [oldResult, newOutput] :
       llvm::zip(newOutputs, newForOp->getResults())) {
    oldResult.replaceAllUsesWith(newOutput);
  }
  forOp->erase();

  return success();
}

bool CanonicalizeMemRefIterArgs(scf::ForOp forOp, BufferOriginAnalysis &bufferOriginAnalysis) {
  return succeeded(replaceMemRefIterArgs(forOp, bufferOriginAnalysis)) &&
         succeeded(removeUnusedIterArgs(forOp));
}

// class LinalgAddInplacePattern : public OpRewritePattern<linalg::AddOp> {
// public:
//   using OpRewritePattern<linalg::AddOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(linalg::AddOp op,
//                                 PatternRewriter &rewriter) const override {

//     Value lhs = op.getOperand(0);
//     Value rhs = op.getOperand(1);
//     Value dst_output = op.getOperand(2);
//     Value output = op.getResult(0);

//     if (lhs == dst_output || rhs == dst_output)
//       return failure();

//     // Check if value is in the iter_args of a scf.for loop
//     auto isInIterArgs = [](Value value, scf::ForOp forOp) {
//       for (Value iterArg : forOp.getRegionIterArgs()) {
//         if (iterArg == value)
//           return true;
//       }
//       return false;
//     };

//     auto forOp = isInForLoop(op);
//     if (!forOp)
//       return failure();

//     // Create a new linalg op with the iter_args input replacing dst_output
//     SmallVector<Value> outOperands;
//     SmallVector<Value> inputOperands;
//     if (isInIterArgs(rhs, forOp)) {
//       outOperands.push_back(findInitArgFromIterArg(
//           rhs, forOp)); // Use rhs as both input and output
//       inputOperands = {lhs, outOperands[0]};
//     } else if (isInIterArgs(lhs, forOp)) {
//       outOperands.push_back(findInitArgFromIterArg(
//           lhs, forOp)); // Use lhs as both input and output
//       inputOperands = {outOperands[0], rhs};
//     } else {
//       return failure();
//     }

//     auto newOp = rewriter.create<linalg::AddOp>(
//         op.getLoc(), op.getResultTypes(), inputOperands, outOperands,
//         linalg::getPrunedAttributeList(op));

//     rewriter.replaceOp(op, newOp.getResults());
//     return success();
//   }
// };
struct RemoveLoopIterArgsWithMemrefTypePass
    : public ::impl::RemoveLoopIterArgsWithMemrefTypeBase<RemoveLoopIterArgsWithMemrefTypePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    BufferOriginAnalysis bufferOriginAnalysis(moduleOp);
    // RewritePatternSet patterns(&getContext());
    // patterns.add<LinalgAddInplacePattern>(&getContext());
    // (void)mlir::applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
    moduleOp.walk([&](scf::ForOp forOp) {
      if (CanonicalizeMemRefIterArgs(forOp, bufferOriginAnalysis)) {
        return WalkResult::advance();
      }
      return WalkResult::skip();
    });
  }
};

} // namespace
std::unique_ptr<Pass> createRemoveLoopIterArgsWithMemrefTypePass() {
  return std::make_unique<RemoveLoopIterArgsWithMemrefTypePass>();
}
} // namespace mlir::triton::ev
