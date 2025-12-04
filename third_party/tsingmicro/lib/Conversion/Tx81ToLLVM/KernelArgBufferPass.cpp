//===- KernelArgBufferPass.cpp - Convert kernel args to single buffer -----===//
//
//
//===----------------------------------------------------------------------===//
//
// This pass transforms kernel function signatures by converting multiple
// arguments into a single void* buffer containing all the arguments.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class KernelArgBufferPass
    : public mlir::triton::KernelArgBufferPassBase<KernelArgBufferPass> {
  using KernelArgBufferPassBase<KernelArgBufferPass>::KernelArgBufferPassBase;

private:
  // Check if the function is a kernel function
  bool isKernelFunction(LLVM::LLVMFuncOp func);

public:
  StringRef getArgument() const final { return "kernel-arg-buffer"; }
  StringRef getDescription() const final {
    return "Convert kernel arguments to a single buffer argument";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override;

private:
  // Insert load op to get real kernel args from new buffered argument
  // Side effect: calculate offset and create ops
  Value insertKernelArgLoad(OpBuilder &builder, Location loc, Value argsBuffer,
                            Type argType, int64_t &currentOffset);
};

bool KernelArgBufferPass::isKernelFunction(LLVM::LLVMFuncOp func) {
  return !(func.getSymName().contains("__Print") ||
           func.getSymName().contains("get_spm_memory_mapping_wrapper"));
}

Value KernelArgBufferPass::insertKernelArgLoad(OpBuilder &builder, Location loc,
                                               Value argsBuffer, Type argType,
                                               int64_t &currentOffset) {
  // Get pointer to the current position in args buffer
  auto offsetValue = builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(currentOffset));

  // NOTE: GEPOp need distinguish the scalar and ptr type. So here ptr + offset
  Value elementPtr =
      builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), argsBuffer);
  elementPtr = builder.create<LLVM::AddOp>(loc, builder.getI64Type(),
                                           elementPtr, offsetValue);
  elementPtr = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), elementPtr);

  // Increment offset. Assume all args are 8 bytes
  currentOffset += sizeof(int64_t);

  // Load the real kernel arg value
  return builder.create<LLVM::LoadOp>(loc, argType, elementPtr);
}

void KernelArgBufferPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Collect functions to process
  SmallVector<LLVM::LLVMFuncOp, 4> kernelFuncs;
  for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (!isKernelFunction(func))
      continue;
    kernelFuncs.push_back(func);
  }
  // NOTE: We move this pass before tx81-to-llvm pass.
  // So we assume the func op must be only one and must be the triton kernel
  assert(kernelFuncs.size() == 1 && "Only one kernel function expected");

  // Process each kernel function
  // TODO: Delete the for loop if the assert is always true for all examples
  for (auto func : kernelFuncs) {
    // Create new function with bufferized signature
    builder.setInsertionPointAfter(func);
    // Save the old block arguments
    SmallVector<BlockArgument> blockArguments =
        llvm::to_vector<8>(func.getArguments());
    auto numArguments = blockArguments.size();

    // New bufferized arg type
    auto voidPtrType = LLVM::LLVMPointerType::get(builder.getContext());

    // New bufferized function type
    auto newFuncType = LLVM::LLVMFunctionType::get(
        func.getFunctionType().getReturnType(), voidPtrType);
    func.setFunctionType(newFuncType);
    SmallVector<DictionaryAttr> newArgAttrs({DictionaryAttr()});
    func.setAllArgAttrs(newArgAttrs);

    // Add the new bufferized argument
    Location loc = func.getLoc();
    Block &entryBlock = func.getBlocks().front();
    entryBlock.insertArgument((unsigned)0, voidPtrType, func.getLoc());

    OpBuilder builder(&entryBlock, entryBlock.begin());
    // Get the bufferized argument
    Value argsBuffer = entryBlock.getArgument(0);

    // Offset tracking for buffer access
    int64_t currentOffset = 0;

    // Process each original argument
    for (auto argIndex : llvm::seq<unsigned>(0, numArguments)) {
      auto oldArg = blockArguments[argIndex];
      Type argType = oldArg.getType();
      Value loadedArg = insertKernelArgLoad(builder, func.getLoc(), argsBuffer,
                                            argType, currentOffset);

      if (blockArguments[argIndex].use_empty())
        continue;
      oldArg.replaceAllUsesWith(loadedArg);
    }
    // Remove the old arguments when replace the use-chain
    entryBlock.eraseArguments(1, numArguments);
  }
}

} // namespace

std::unique_ptr<Pass> triton::createKernelArgBufferPass() {
  return std::make_unique<KernelArgBufferPass>();
}
