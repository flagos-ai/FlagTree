// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

struct RewriteArithSelectOp : mlir::OpConversionPattern<mlir::arith::SelectOp> {
  using mlir::OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Note we're replacing the select op with an if op because we are
    // converting one value into many values.
    auto newIf = mlir::scf::IfOp::create(
        rewriter, op.getLoc(), mlir::TypeRange(adaptor.getTrueValue()),
        op.getCondition(), true);
    // We set the attributes from the op in case the op has any additional
    // attributes
    newIf->setAttrs(op->getAttrs());

    {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(newIf.thenBlock());
      mlir::scf::YieldOp::create(rewriter, op->getLoc(),
                                 adaptor.getTrueValue());
      rewriter.setInsertionPointToStart(newIf.elseBlock());
      mlir::scf::YieldOp::create(rewriter, op->getLoc(),
                                 adaptor.getFalseValue());
    }

    // Replace the old operation results
    rewriter.replaceOpWithMultiple(op, {newIf->getResults()});

    return mlir::success();
  }
};

} // namespace
namespace mlir::triton {

void populateArithTypeConversions(const TypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<RewriteArithSelectOp>(converter, patterns.getContext());
}

} // namespace mlir::triton
