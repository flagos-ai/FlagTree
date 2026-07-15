/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace mlir {

#define GEN_PASS_DEF_TLEPROMOTEASYNCLOADTOAIUPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {

class TlePromoteAsyncLoadToAIUPass
    : public impl::TlePromoteAsyncLoadToAIUPassBase<
          TlePromoteAsyncLoadToAIUPass> {
public:
  using impl::TlePromoteAsyncLoadToAIUPassBase<
      TlePromoteAsyncLoadToAIUPass>::TlePromoteAsyncLoadToAIUPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());

    SmallVector<triton::LoadOp> toPromote;
    m.walk([&](triton::LoadOp loadOp) {
      auto asyncAttr = llvm::cast_if_present<BoolAttr>(
          loadOp->getAttr("tt.load.async"));
      if (!asyncAttr || !asyncAttr.getValue())
        return;
      if (!triton::isTensorPointerType(loadOp.getPtr().getType()))
        return;
      toPromote.push_back(loadOp);
    });

    for (auto loadOp : toPromote) {
      builder.setInsertionPoint(loadOp);
      auto aiuLoad = triton::AIULoadOp::create(
          builder, loadOp.getLoc(), loadOp.getType(), loadOp.getPtr(),
          loadOp.getCache(), loadOp.getEvict());
      loadOp.replaceAllUsesWith(aiuLoad.getResult());
      loadOp.erase();
    }
  }
};

} // namespace
} // namespace mlir
