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

#include "TritonControlFlowOpt/TritonControlFlowOptPass.h"

#include "TritonControlFlowOpt/BlockPtrDecompose.h"
#include "TritonControlFlowOpt/CFGStructuring.h"
#include "TritonControlFlowOpt/TensorPtrDecompose.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

void TritonControlFlowOptPass::getDependentDialects(
    DialectRegistry &registry) const {
  // CFG structuring creates SCF operations, while pointer decomposition
  // materializes arith and Triton pointer operations.
  registry
      .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              scf::SCFDialect, scope::ScopeDialect, triton::TritonDialect>();
}

void TritonControlFlowOptPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Apply the control-flow preprocessing pipeline in dependency order:
  //   1. normalize supported cf graphs to scf;
  //   2. decompose block-pointer descriptors across supported boundaries;
  //   3. replace common-base tensor pointers at those boundaries with offsets.
  // TODO: Extend stage 3 to carry both the base and complete offsets, then add
  // StructuredOffsetsDecompose to further split structured offsets into a
  // base offset and per-dimension strides.
  // Keep these calls explicit so each transformation has one owner, can be
  // tested independently and can be implemented without changing pass
  // registration.
  if (failed(controlflow::structureCFG(module)) ||
      failed(controlflow::runBlockPtrDecompose(module)) ||
      failed(controlflow::runTensorPtrDecompose(module)) ||
      failed(verify(module)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createTritonControlFlowOptPass() {
  // Keep construction in this translation unit; registration is generated
  // from Passes.td and exposes only this public factory.
  return std::make_unique<TritonControlFlowOptPass>();
}

} // namespace mlir::triton
