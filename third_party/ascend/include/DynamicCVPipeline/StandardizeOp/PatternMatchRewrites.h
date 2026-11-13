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

#ifndef TRITON_ADAPTER_DYNAMIC_CVPIPELINE_PATTERN_MATCH_REWRITES_H
#define TRITON_ADAPTER_DYNAMIC_CVPIPELINE_PATTERN_MATCH_REWRITES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::CVSplit {

class SplitMatmulPattern : public mlir::OpRewritePattern<linalg::MatmulOp> {
public:
  explicit SplitMatmulPattern(mlir::MLIRContext *context, bool needSplitAll)
      : OpRewritePattern<linalg::MatmulOp>(context),
        needSplitAll(needSplitAll) {}
  llvm::LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                      PatternRewriter &rewriter) const override;

private:
  bool needSplitAll;
};

class FoldExpandExtCollapse
    : public mlir::OpRewritePattern<mlir::tensor::CollapseShapeOp> {
public:
  using OpRewritePattern<mlir::tensor::CollapseShapeOp>::OpRewritePattern;
  llvm::LogicalResult matchAndRewrite(mlir::tensor::CollapseShapeOp collapseOp,
                                      PatternRewriter &rewriter) const override;
};

class PatternMatchRewritePass
    : public PassWrapper<PatternMatchRewritePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PatternMatchRewritePass);

  PatternMatchRewritePass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pattern-match";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPatternMatchRewritePass();

} // namespace mlir::triton::CVSplit

#endif // TRITON_ADAPTER_DYNAMIC_CVPIPELINE_PATTERN_MATCH_REWRITES_H
