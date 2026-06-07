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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace triton {

struct ControlFlowConditionInfo {
  llvm::DenseMap<scf::ForOp, SmallVector<int>> blockCounters;
  llvm::DenseMap<scf::ForOp, int> blockCounterNums;
  llvm::DenseMap<scf::ForOp, SmallVector<int>> innerDepConds;

  llvm::DenseMap<Value, SmallVector<Value>> crossCoreDependentMap;
  llvm::DenseMap<scf::ForOp, llvm::DenseMap<Value, SmallVector<Value>>> intraCoreDependentMap;

  // unique counter value for each ifblock
  llvm::DenseMap<scf::IfOp, Value> cntArgs;
};

class AddControlFlowConditionPass
    : public PassWrapper<AddControlFlowConditionPass, OperationPass<ModuleOp>> {
public:
  AddControlFlowConditionPass() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override
  {
    registry.insert<LLVM::LLVMDialect>();
  }

  llvm::StringRef getArgument() const override { return "add-control-flow-condition"; }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass();

void registerAddControlFlowConditionPasses();
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H