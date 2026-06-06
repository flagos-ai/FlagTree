
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/SplitDataflow/AddBlockIdForControlOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "add-block-id-for-control-ops";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

// Pass Entry Point
void AddBlockIdForControlOpsPass::runOnOperation()
{
  LOG_DEBUG("\n--- enter AddBlockIdForControlOpsPass --->\n");
  ModuleOp module = getOperation();

  // Step 1: find the max block_id
  int maxBlockId = -1;
  module.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      int currentId = attr.getInt();
      if (currentId > maxBlockId) {
        maxBlockId = currentId;
      }
    }
  });

  LOG_DEBUG("maxBlockId: " << maxBlockId << "\n");

  // Step 2: add block_id for control flow ops
  module.walk([&](Operation *op) {
    // skip op with block_id
    if (op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      return;
    }

    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
      maxBlockId++;
      static constexpr int kIntegerBitWidth = 32;
      op->setAttr("ssbuffer.block_id",
                  IntegerAttr::get(IntegerType::get(module.getContext(), kIntegerBitWidth), maxBlockId));
      LOG_DEBUG("Added block_id " << maxBlockId << " to " << *op << "\n");
    }
  });

  LOG_DEBUG("\n--- exit AddBlockIdForControlOpsPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddBlockIdForControlOpsPass()
{
    return std::make_unique<AddBlockIdForControlOpsPass>();
}

} // namespace triton
} // namespace mlir