/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/AttrInterfaces.cpp.inc"
#include "triton/Dialect/Triton/IR/Dialect.cpp.inc"
#include "triton/Dialect/Triton/IR/OpInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// TritonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

bool TritonInlinerInterface::isLegalToInline(Operation *call,
                                             Operation *callable,
                                             bool wouldBeCloned) const {
  auto funcOp = dyn_cast<triton::FuncOp>(callable);
  if (!funcOp)
    return true;
  if (funcOp->hasAttr("noinline"))
    return !funcOp->getAttrOfType<BoolAttr>("noinline").getValue();
  return true;
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void TritonInlinerInterface::handleTerminator(Operation *op,
                                              Block *newDest) const {
  // Only return needs to be handled here.
  auto returnOp = dyn_cast<triton::ReturnOp>(op);
  if (!returnOp)
    return;

  // Replace the return with a branch to the dest.
  OpBuilder builder(op);
  mlir::cf::BranchOp::create(builder, op->getLoc(), newDest,
                             returnOp.getOperands());
  op->erase();
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void TritonInlinerInterface::handleTerminator(Operation *op,
                                              ValueRange valuesToRepl) const {
  // Only return needs to be handled here.
  auto returnOp = cast<triton::ReturnOp>(op);

  // Replace the values directly with the return operands.
  assert(returnOp.getNumOperands() == valuesToRepl.size());
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
      >();

  // We can also add interface here.
  addInterfaces<TritonInlinerInterface>();
}

Operation *TritonDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
