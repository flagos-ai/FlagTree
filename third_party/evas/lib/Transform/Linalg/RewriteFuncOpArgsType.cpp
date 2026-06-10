#include "epu/memory.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton-shared/Transform/common_utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

#define GEN_PASS_DEF_REWRITEFUNCOPARGSTYPE
#include "evas/Transform/Linalg/Passes.h.inc"
namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;

struct RewriteFuncOpArgsTypePass
    : public ::impl::RewriteFuncOpArgsTypeBase<RewriteFuncOpArgsTypePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    SymbolTable symTable(moduleOp);

    // Walk through all AnnotateOp operations in the module
    moduleOp.walk([&](func::CallOp op) {
      auto funcOp = utils::getCalledFunction(op);
      // Check if all operands come from to_tensor ops
      SmallVector<Value> newOperands;
      SmallVector<Type> newTypes;

      for (auto operand : op.getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(defOp)) {
            newOperands.push_back(toTensorOp.getOperand());
            newTypes.push_back(toTensorOp.getOperand().getType());
          } else {
            op->emitError("operands must all come from to_tensor ops");
            return;
          }
        }
      }

      // Create new function type with memref types and no results
      auto newFuncType = builder.getFunctionType(newTypes, funcOp.getResultTypes());
      builder.setInsertionPoint(op->getParentOfType<func::FuncOp>());
      // Construct new function name by appending "_memref" to original name
      auto newFuncName = funcOp.getName().str() + "_memref";
      // Create new function with updated type
      auto newFuncOp = builder.create<func::FuncOp>(
          funcOp.getLoc(), newFuncName, newFuncType,
          funcOp.getSymVisibilityAttr(), funcOp.getArgAttrsAttr(),
          funcOp.getResAttrsAttr());
      newFuncOp->setAttrs(funcOp->getDiscardableAttrDictionary());
      symTable.insert(newFuncOp);
      // Copy function body if exists
      if (!funcOp.empty()) {
        Block &oldEntryBlock = funcOp.getBody().front();
        Block *newEntryBlock = newFuncOp.addEntryBlock();

        // Insert to_tensor ops at the beginning of the new function
        builder.setInsertionPointToStart(newEntryBlock);
        SmallVector<Value> tensorArgs;
        for (auto arg : newFuncOp.getArguments()) {
          auto toTensor =
              builder.create<mlir::bufferization::ToTensorOp>(arg.getLoc(), 
                            memref::getTensorTypeFromMemRefType(arg.getType()), 
                            arg)->getResult(0);
          tensorArgs.push_back(toTensor); 
        }

        // Clone the rest of function body
        IRMapping mapper;
        for (auto [oldArg, newArg] :
             llvm::zip(oldEntryBlock.getArguments(), tensorArgs)) {
          mapper.map(oldArg, newArg);
        }
        for (auto &op : oldEntryBlock) {
          builder.clone(op, mapper);
        }
      }
      // Replace old function with new one
      funcOp.erase();
      // Create new call op with memref operands and no results
      builder.setInsertionPoint(op);
      auto newCall =
          builder.create<func::CallOp>(op.getLoc(), newFuncOp, newOperands);
      newCall->setAttrs(op->getDiscardableAttrDictionary());
      // Remove old call op since it's no longer needed
      op.replaceAllUsesWith(newCall);
      op.erase();
    });
  }
};

} // namespace
std::unique_ptr<Pass> createRewriteFuncOpArgsTypePass() {
  return std::make_unique<RewriteFuncOpArgsTypePass>();
}
} // namespace mlir::triton::ev
