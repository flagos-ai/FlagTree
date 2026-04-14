// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
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

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEOPTIMIZELOCALPOINTERSTORES
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

class OptimizeLocalPointerStoresPass
    : public impl::TritonTleOptimizeLocalPointerStoresBase<
          OptimizeLocalPointerStoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<triton::StoreOp> stores;
    module.walk([&](triton::StoreOp store) { stores.push_back(store); });

    for (triton::StoreOp store : stores) {
      if (!store)
        continue;

      Value ptr = stripConvertLayouts(store.getPtr());
      auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>();
      if (!localPointers)
        continue;

      auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
      auto memDescTy =
          dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
      if (!valueTy || !memDescTy)
        continue;

      if (!store.getBoundaryCheck().empty())
        continue;
      if (valueTy.getShape() != memDescTy.getShape())
        continue;
      if (valueTy.getElementType() != memDescTy.getElementType())
        continue;

      OpBuilder builder(store);
      Value valueToStore = store.getValue();

      if (Value mask = store.getMask()) {
        auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
        if (!maskTy || maskTy.getShape() != valueTy.getShape())
          continue;
        if (maskTy.getEncoding() != valueTy.getEncoding()) {
          auto targetMaskTy =
              RankedTensorType::get(maskTy.getShape(), maskTy.getElementType(),
                                    valueTy.getEncoding());
          mask = builder
                     .create<ttg::ConvertLayoutOp>(store.getLoc(), targetMaskTy,
                                                   mask)
                     .getResult();
        }
        Value oldValue = builder.create<ttg::LocalLoadOp>(
            store.getLoc(), valueTy, localPointers.getSrc());
        valueToStore = builder
                           .create<arith::SelectOp>(store.getLoc(), mask,
                                                    valueToStore, oldValue)
                           .getResult();
      }

      builder.create<ttg::LocalStoreOp>(store.getLoc(), valueToStore,
                                        localPointers.getSrc());
      store.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
