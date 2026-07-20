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

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUREMOVETMEMTOKENSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

void eraseResult(Operation *op, unsigned resultIdx, Value replacement) {
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       op->getResultTypes(), op->getAttrs());
  state.types.erase(std::next(state.types.begin(), resultIdx));
  OpBuilder b(op);
  Operation *newOp = b.create(state);
  SmallVector<Value> replacements = newOp->getResults();
  replacements.insert(std::next(replacements.begin(), resultIdx), replacement);
  op->replaceAllUsesWith(replacements);
  op->erase();
}

void removeTMEMToken(Operation *op, Value dummy) {
  if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
    mmaOp.getAccDepMutable().clear();
    if (mmaOp.getToken())
      eraseResult(mmaOp, 0, dummy);
  } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
    store.getDepMutable().clear();
    if (store.getToken())
      eraseResult(store, 0, dummy);
  } else if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    if (alloc.getToken())
      eraseResult(alloc, 1, dummy);
  } else if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    load.getDepMutable().clear();
    if (load.getToken())
      eraseResult(load, 1, dummy);
  }
}

} // anonymous namespace

class TritonNvidiaGPURemoveTMEMTokensPass
    : public impl::TritonNvidiaGPURemoveTMEMTokensPassBase<
          TritonNvidiaGPURemoveTMEMTokensPass> {
public:
  using TritonNvidiaGPURemoveTMEMTokensPassBase::
      TritonNvidiaGPURemoveTMEMTokensPassBase;

  void runOnOperation() override {
    for (auto func : getOperation().getOps<FuncOp>()) {
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      // Placeholder value that will get DCE'd by the canonicalizer.
      Value dummy = ub::PoisonOp::create(
          b, func.getLoc(), b.getType<triton::gpu::AsyncTokenType>());
      func.walk([&](Operation *op) { removeTMEMToken(op, dummy); });
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
