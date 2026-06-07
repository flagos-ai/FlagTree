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

#ifndef TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H
#define TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DECL_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"

#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"

extern bool compileOn91095Flag;

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options = {});
} // namespace triton
} // namespace mlir

namespace {
using namespace mlir;
using namespace triton;

class AddDynamicCVPipelinePass
    : public ::impl::AddDynamicCVPipelineBase<AddDynamicCVPipelinePass> {
public:
    explicit AddDynamicCVPipelinePass(const AddDynamicCVPipelineOptions &options);
    void runOnOperation() override;
};

} // namespace

#endif // TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H