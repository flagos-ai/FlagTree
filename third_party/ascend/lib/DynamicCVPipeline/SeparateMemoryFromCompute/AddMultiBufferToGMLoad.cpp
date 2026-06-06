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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadInternal.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"

using namespace mlir;
using namespace triton;
using namespace gmload;

// Default multi-buffer depth for GM load operations.
static constexpr int kDefaultBufferDepth = 2;

// ============================================================================
// Step functions
// ============================================================================

void AddMultiBufferToGMLoadPass::collectAndGroupMarkedOps()
{
    auto module = getOperation();

    // Scan all IR for marked ops (generic, no loop dependency).
    markedOps_ = collectMarkedOps(module);
    if (markedOps_.empty()) {
        LOG_DEBUG("No marked loads found, nothing to transform\n");
        return;
    }
    LOG_DEBUG("Marked loads collected, start transformation\n");

    // Group by enclosing scf::ForOp.
    groupByEnclosingForOp(markedOps_, contexts_);

    // Apply depth policy: skip loops whose compile-time trip count is too small
    // to benefit, then record the slot count on each group.
    int depth = kDefaultBufferDepth;
    llvm::erase_if(contexts_, [depth](const ForBufferCtx &context) {
        if (auto tripCount = getConstantTripCount(context.forOp))
            return *tripCount <= depth;
        return false;
    });
    for (auto &context : contexts_)
        for (auto &group : context.groups)
            group.depth = depth;

    if (contexts_.empty())
        LOG_DEBUG("No bufferable loops found\n");
}

void AddMultiBufferToGMLoadPass::sortContextsInnerFirst()
{
    // Sort contexts inner-first so that inner loops are transformed
    // before the outer loops that contain them.
    auto getNestingDepth = [](Operation *loopOp) {
        unsigned nestingDepth = 0;
        for (Operation *parentOp = loopOp->getParentOp(); parentOp; parentOp = parentOp->getParentOp())
            ++nestingDepth;
        return nestingDepth;
    };

    llvm::sort(contexts_, [&getNestingDepth](const ForBufferCtx &leftContext, const ForBufferCtx &rightContext) {
        Operation *leftLoopOp = const_cast<scf::ForOp &>(leftContext.forOp).getOperation();
        Operation *rightLoopOp = const_cast<scf::ForOp &>(rightContext.forOp).getOperation();
        unsigned leftDepth = getNestingDepth(leftLoopOp);
        unsigned rightDepth = getNestingDepth(rightLoopOp);
        if (leftDepth != rightDepth)
            return leftDepth > rightDepth;
        if (leftLoopOp->getBlock() == rightLoopOp->getBlock())
            return leftLoopOp->isBeforeInBlock(rightLoopOp);
        return false;
    });
}

LogicalResult AddMultiBufferToGMLoadPass::applyMultiBufferToGMLoadLoops()
{
    for (auto &context : contexts_)
        allCtxForOps_.insert(context.forOp.getOperation());

    // Process inner loops before outer loops.
    for (auto &context : contexts_)
        if (failed(applyMultiBufferToForLoop(context, allCtxForOps_)))
            return failure();

    return success();
}

void AddMultiBufferToGMLoadPass::cleanupTransformedIR()
{
    auto module = getOperation();

    // Deduplicate untagged constants that inner-loop processing created
    // before outer-loop processing could provide the dominating equivalents.
    deduplicateConstants(module);

    // Erase replaced original for ops.
    llvm::DenseSet<Operation *> nestedForOps;
    for (auto &context : contexts_) {
        Operation *parentOp = context.forOp->getParentOp();
        while (parentOp) {
            if (allCtxForOps_.contains(parentOp)) {
                nestedForOps.insert(context.forOp.getOperation());
                break;
            }
            parentOp = parentOp->getParentOp();
        }
    }

    for (auto &context : llvm::reverse(contexts_)) {
        if (nestedForOps.contains(context.forOp.getOperation()))
            continue;
        context.forOp.erase();
    }
}

// ============================================================================
// Pass entry point
// ============================================================================

void AddMultiBufferToGMLoadPass::runOnOperation()
{
    auto module = getOperation();
    LOG_DEBUG("Enter add-multi-buffer-to-gm-load pass\n");
    LOG_DEBUG("Before add-multi-buffer-to-gm-load:\n" << module << "\n");

    // Step 1: Collect marked ops and group by enclosing forOp
    markedOps_.clear();
    contexts_.clear();
    collectAndGroupMarkedOps();
    if (contexts_.empty())
        return;

    // Step 2: Sort contexts inner-first
    sortContextsInnerFirst();

    // Step 3: Transform each for loop with multi-buffer logic
    allCtxForOps_.clear();
    if (failed(applyMultiBufferToGMLoadLoops())) {
        module.emitError() << "[" << DEBUG_TYPE << "] Step 3 applyMultiBufferToGMLoadLoops failed";
        signalPassFailure();
        return;
    }

    // Step 4: Cleanup transformed IR
    cleanupTransformedIR();

    LOG_DEBUG("After add-multi-buffer-to-gm-load:\n" << module << "\n");
    LOG_DEBUG("Process successfully\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass()
{
    return std::make_unique<AddMultiBufferToGMLoadPass>();
}

} // namespace triton
} // namespace mlir
