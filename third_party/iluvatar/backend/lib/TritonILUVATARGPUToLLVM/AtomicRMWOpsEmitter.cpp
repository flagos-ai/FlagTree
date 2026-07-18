// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
