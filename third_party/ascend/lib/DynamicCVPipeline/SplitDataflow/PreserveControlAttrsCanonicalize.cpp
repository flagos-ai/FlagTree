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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "PreserveControlAttrsCanonicalize";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

template <typename... Args>
static void logDebug(const Args &...args)
{
    LLVM_DEBUG({
        auto &debugStream = llvm::dbgs();
        debugStream << '[' << DEBUG_TYPE << "] ";
        (debugStream << ... << args);
        debugStream << "\n";
    });
}

namespace {

static bool isTrackedControlFlowOp(Operation *op)
{
    return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static bool canTransferAttrs(Operation *from, Operation *to)
{
    return from && to && from != to && isTrackedControlFlowOp(from) &&
           isTrackedControlFlowOp(to) && from->getName() == to->getName();
}

class PreserveControlAttrsListener : public RewriterBase::Listener {
public:
    void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint) override
    {
        recentInserts.insert(op);
    }

    void notifyOperationErased(Operation *op) override
    {
        recentInserts.remove(op);
    }

    void notifyOperationReplaced(Operation *op, Operation *newOp) override
    {
        transferAttrs(op, newOp);
    }

    void notifyOperationReplaced(Operation *op, ValueRange values) override
    {
        if (Operation *newOp = findReplacementOp(op, values)) {
            transferAttrs(op, newOp);
        }
    }

private:
    Operation *findReplacementOp(Operation *oldOp, ValueRange replacements) const
    {
        if (!isTrackedControlFlowOp(oldOp)) {
            return nullptr;
        }

        for (Value value : replacements) {
            Operation *defOp = value.getDefiningOp();
            if (defOp && recentInserts.contains(defOp) && canTransferAttrs(oldOp, defOp)) {
                return defOp;
            }
        }

        for (Operation *candidate : llvm::reverse(recentInserts.getArrayRef())) {
            if (canTransferAttrs(oldOp, candidate)) {
                return candidate;
            }
        }
        return nullptr;
    }

    static void transferAttrs(Operation *from, Operation *to)
    {
        if (!canTransferAttrs(from, to)) {
            return;
        }

        for (NamedAttribute attr : from->getAttrs()) {
            if (to->hasAttr(attr.getName())) {
                continue;
            }
            to->setAttr(attr.getName(), attr.getValue());
            logDebug("carry attr '", attr.getName(), "' from ",
                     from->getName().getStringRef(), " to ",
                     to->getName().getStringRef());
        }
    }

    llvm::SetVector<Operation *> recentInserts;
};

static void populateCanonicalizationPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns)
{
    for (Dialect *dialect : ctx->getLoadedDialects()) {
        dialect->getCanonicalizationPatterns(patterns);
    }

    for (RegisteredOperationName opName : ctx->getRegisteredOperations()) {
        opName.getCanonicalizationPatterns(patterns, ctx);
    }
}

} // namespace

void mlir::triton::PreserveControlAttrsCanonicalizePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    populateCanonicalizationPatterns(&getContext(), patterns);

    PreserveControlAttrsListener listener;
    GreedyRewriteConfig config;
    config.setListener(&listener);

    if (failed(applyPatternsGreedily(getOperation(),
                                     FrozenRewritePatternSet(std::move(patterns)),
                                     config))) {
        getOperation()->emitError("PreserveControlAttrsCanonicalizePass failed");
        signalPassFailure();
    }
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreserveControlAttrsCanonicalizePass()
{
    return std::make_unique<PreserveControlAttrsCanonicalizePass>();
}

void registerPreserveControlAttrsCanonicalizePasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createPreserveControlAttrsCanonicalizePass();
    });
}

} // namespace triton
} // namespace mlir