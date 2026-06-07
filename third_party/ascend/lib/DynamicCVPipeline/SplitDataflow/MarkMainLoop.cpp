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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "mark-main-loop";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

// Pass Entry Point
void MarkMainLoopPass::runOnOperation()
{
    LOG_DEBUG("\n--- enter MarkMainLoopPass --->\n");
    ModuleOp module = getOperation();

    int mainLoopIdCounter = 0;
    SmallVector<scf::ForOp> mainLoops;

    // Find all candidate main loops
    module.walk([&](Operation *op) {
        if (isa<hivm::FixpipeOp, hivm::CopyOp>(op)) {
            if (auto forOp = op->getParentOfType<scf::ForOp>()) {
                mainLoops.push_back(forOp);
            }
        }
    });

    for (scf::ForOp forOp : mainLoops) {
        if (!forOp->hasAttr("ssbuffer.main_loop")) {
            // Add attribute with integer value (current counter ID)
            forOp->setAttr("ssbuffer.main_loop", Builder(module.getContext()).getI32IntegerAttr(mainLoopIdCounter));
            mainLoopIdCounter++;
        }
    }

    // Remove main_loop attribute from outer loops if nested loops both have it
    // Keep only the innermost main_loop
    SmallVector<scf::ForOp> allMainLoops;
    module.walk([&](scf::ForOp forOp) {
        if (forOp->hasAttr("ssbuffer.main_loop")) {
            allMainLoops.push_back(forOp);
        }
    });

    for (scf::ForOp forOp : allMainLoops) {
        // Check if there's any nested for loop with main_loop attribute
        bool hasNestedMainLoop = false;
        forOp.walk([&](scf::ForOp nestedForOp) {
            if (nestedForOp != forOp && nestedForOp->hasAttr("ssbuffer.main_loop")) {
                hasNestedMainLoop = true;
            }
        });
        // Remove attribute from outer loop if inner loop also has it
        if (hasNestedMainLoop) {
            forOp->removeAttr("ssbuffer.main_loop");
        }
    }

    LOG_DEBUG("--- exit MarkMainLoopPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createMarkMainLoopPass()
{
    return std::make_unique<MarkMainLoopPass>();
}
} // namespace triton
} // namespace mlir
