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

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "FoldExpandExtCollapse";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace mlir::triton::CVSplit {

LogicalResult
FoldExpandExtCollapse::matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                       PatternRewriter &rewriter) const {
  auto collapseReassoc = collapseOp.getReassociationIndices();

  if (collapseReassoc.size() != 1 || collapseReassoc[0].size() != 2 ||
      collapseReassoc[0][0] != 0 || collapseReassoc[0][1] != 1) {
    return failure();
  }

  auto extOp = collapseOp.getSrc().getDefiningOp();
  if (!extOp) {
    return failure();
  }

  auto extUIOp = dyn_cast<arith::ExtUIOp>(extOp);
  auto extSIOp = dyn_cast<arith::ExtSIOp>(extOp);
  if (!extUIOp && !extSIOp) {
    return failure();
  }

  if (!extOp->hasOneUse()) {
    return failure();
  }

  auto expandOp = extOp->getOperand(0).getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandOp) {
    return failure();
  }

  if (!expandOp->hasOneUse()) {
    return failure();
  }

  auto expandReassoc = expandOp.getReassociationIndices();
  if (expandReassoc.size() != 1 || expandReassoc[0].size() != 2 ||
      expandReassoc[0][0] != 0 || expandReassoc[0][1] != 1) {
    return failure();
  }

  Value extInput = expandOp.getSrc();
  Type resultType = collapseOp.getResult().getType();

  if (extUIOp) {
    auto newExtUI = rewriter.create<arith::ExtUIOp>(collapseOp.getLoc(),
                                                    resultType, extInput);
    rewriter.replaceOp(collapseOp, newExtUI.getResult());
  } else {
    auto newExtSI = rewriter.create<arith::ExtSIOp>(collapseOp.getLoc(),
                                                    resultType, extInput);
    rewriter.replaceOp(collapseOp, newExtSI.getResult());
  }

  return success();
}

} // namespace mlir::triton::CVSplit
