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

#ifndef TRITON_ADAPTER_CONVERSION_TRITON_CONTROL_FLOW_OPT_H
#define TRITON_ADAPTER_CONVERSION_TRITON_CONTROL_FLOW_OPT_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_CLASSES
#include "ascend/include/TritonControlFlowOpt/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTritonControlFlowOptPass();

class TritonControlFlowOptPass
    : public TritonControlFlowOptBase<TritonControlFlowOptPass> {
public:
  TritonControlFlowOptPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_CONVERSION_TRITON_CONTROL_FLOW_OPT_H
