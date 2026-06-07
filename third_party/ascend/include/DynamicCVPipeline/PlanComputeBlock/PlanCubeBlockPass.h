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

#ifndef TRITION_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_PLAN_CUBE_BLOCK_PASS_H
#define TRITION_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_PLAN_CUBE_BLOCK_PASS_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

class PlanCubeBlockPass : public PassWrapper<PlanCubeBlockPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanCubeBlockPass);

    PlanCubeBlockPass() = default;
    void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createPlanCubeBlockPass();
} // namespace triton
} // namespace mlir

#endif // TRITION_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_PLAN_CUBE_BLOCK_PASS_H
