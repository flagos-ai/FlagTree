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

#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_SCHEDULEBUFFERSTOREPASS
#include "Dialect/ProtonGPU/Transforms/Passes.h.inc"

struct ScheduleBufferStorePass
    : public impl::ScheduleBufferStorePassBase<ScheduleBufferStorePass> {

  using impl::ScheduleBufferStorePassBase<
      ScheduleBufferStorePass>::ScheduleBufferStorePassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();
    OpBuilder builder(context);

    // TODO(srir): Add support for non-inline kernels
    FuncOp func = *m.getOps<triton::FuncOp>().begin();
    auto startStoreList = llvm::SmallVector<CircularStoreOp, 8>();
    auto endStoreMap = llvm::SmallDenseMap<int, CircularStoreOp, 8>();

    func.walk([&](CircularStoreOp store) {
      if (store.getIsStart())
        startStoreList.push_back(store);
      else
        endStoreMap[store.getScopeId()] = store;
    });

    for (auto store : startStoreList) {
      int scopeId = store.getScopeId();
      auto endStore = endStoreMap[scopeId];
      if (!endStore) {
        mlir::emitError(func.getLoc(), "proton end store not found");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(endStore);
      builder.clone(*store);
      store->erase();
    }
  }
};

} // namespace mlir::triton::proton::gpu
