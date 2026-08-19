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

#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

namespace mlir::triton::gpu {

void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      auto oBufferId = funcAllocation->getBufferId(op);
      int offset = -1;
      if (oBufferId != Allocation::InvalidBufferId)
        offset = funcAllocation->getOffset(oBufferId);
      else if (op->getNumResults() == 1) {
        Value value = op->getResult(0);
        auto vBufferId = funcAllocation->getBufferId(value);
        if (vBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(vBufferId);
      }
      if (offset == -1)
        return;
      op->setAttr("allocation.offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
    });
    return WalkResult::skip();
  });
  mod->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                      allocation.getSharedMemorySize()));
}

} // namespace mlir::triton::gpu
