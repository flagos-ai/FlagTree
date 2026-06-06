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

#ifndef TRITON_ASCEND_SSBUF_UPDATE_FOR_OPS_FOR_CONTROL_FLOW_H
#define TRITON_ASCEND_SSBUF_UPDATE_FOR_OPS_FOR_CONTROL_FLOW_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"

namespace mlir {
namespace triton {

class UpdateForOpsPass
    : public PassWrapper<UpdateForOpsPass, OperationPass<ModuleOp>> {
public:
  UpdateForOpsPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info)
  {
    this->info = info;
  }

  llvm::StringRef getArgument() const override
  {
    return "update-for-ops";
  }

private:
  LogicalResult addBlockCountersAndInnerDepConds(ModuleOp module, ControlFlowConditionInfo *info);

  LogicalResult deriveBlockCountersFromIfOps(ModuleOp module, ControlFlowConditionInfo *info);

  LogicalResult insertInterCorePipeS(ModuleOp module);

  ControlFlowConditionInfo *info = nullptr;
};

std::unique_ptr<OperationPass<ModuleOp>> createUpdateForOpsPass();

} // namespace triton
} // namespace mlir
#endif // TRITON_ASCEND_SSBUF_UPDATE_FOR_OPS_FOR_CONTROL_FLOW_H
