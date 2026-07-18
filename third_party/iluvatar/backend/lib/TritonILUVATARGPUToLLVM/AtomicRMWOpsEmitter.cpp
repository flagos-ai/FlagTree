#include "Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "AtomicRMWOpsEmitter.h"

using namespace triton::ILUVATAR;

namespace mlir::LLVM::ILUVATAR {

Value AtomicRMWEmitter::emitAtomicRMW(
    RewriterBase &rewriter, Value rmwPtr, Value valElem, Value rmwMask,
    std::optional<Value> sharedMemBase) const {
  auto loc = rmwPtr.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type retType = valElem.getType();
  Value undefVal = b.undef(retType);
  // Build blocks to bypass the atomic instruction for ~rmwMask.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *atomicBlock = rewriter.createBlock(
      curBlock->getParent(), std::next(Region::iterator(curBlock)));
  endBlock->addArgument({retType}, {loc});

  rewriter.setInsertionPointToEnd(curBlock);

  LLVM::CondBrOp::create(rewriter, loc, rmwMask, atomicBlock, endBlock,
                         undefVal);

  rewriter.setInsertionPointToEnd(atomicBlock);
  Value atom = LLVM::AtomicRMWOp::create(rewriter, loc, binOp, rmwPtr, valElem,
                                         memOrder, scopeStr.c_str())
                   .getResult();

  if (sharedMemBase.has_value()) {
    Value atomPtr = *sharedMemBase;
    b.store(atom, atomPtr);
  }
  LLVM::BrOp::create(rewriter, loc, atom, endBlock);
  rewriter.setInsertionPointToStart(endBlock);

  return endBlock->getArgument(0);
}

} // namespace mlir::LLVM::ILUVATAR
