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

#ifndef TRITON_ASCEND_ANALYZE_DATAFLOW_H
#define TRITON_ASCEND_ANALYZE_DATAFLOW_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// Pass for analyzing tensor args in main_loop forOps
class AnalyzeArgsPass
    : public PassWrapper<AnalyzeArgsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeArgsPass)

  AnalyzeArgsPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-args"; }
  llvm::StringRef getDescription() const override {
    return "Analyze tensor args in main_loop forOps";
  }
};

// Pass for analyzing function names
class AnalyzeNamePass
    : public PassWrapper<AnalyzeNamePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeNamePass)

  AnalyzeNamePass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-name"; }
  llvm::StringRef getDescription() const override {
    return "Analyze function names for dynamic CV pipeline";
  }
};

// Wrapper pass for AnalyzeDataFlow
class AnalyzeDataFlowPass
    : public PassWrapper<AnalyzeDataFlowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeDataFlowPass)

  AnalyzeDataFlowPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-data-flow"; }
  llvm::StringRef getDescription() const override {
    return "Analyze data flow and detect tensor args in different block_ids";
  }
};

// Pass for analyzing flag operations
class AnalyzeFlagPass
    : public PassWrapper<AnalyzeFlagPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeFlagPass)

  AnalyzeFlagPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-flag"; }
  llvm::StringRef getDescription() const override {
    return "Analyze flag operations for control flow optimization";
  }
};

// Pass for analyzing scope-level issues
class AnalyzeScopePass
    : public PassWrapper<AnalyzeScopePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeScopePass)

  AnalyzeScopePass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-scope"; }
  llvm::StringRef getDescription() const override {
    return "Analyze scope-level issues for control flow";
  }
};

// Pass for analyzing whether an scf.while condition depends on non-scalar
// (tensor) data through its loop-carried arg update chain
class AnalyzeWhileConditionArgsPass
    : public PassWrapper<AnalyzeWhileConditionArgsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeWhileConditionArgsPass)

  AnalyzeWhileConditionArgsPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override {
    return "analyze-while-condition-args";
  }
  llvm::StringRef getDescription() const override {
    return "Analyze whether scf.while condition args are updated with "
           "non-scalar data";
  }
};

// Pass for analyzing cube control flow input chain
class AnalyzeCubeControlFlowInputChainPass
    : public PassWrapper<AnalyzeCubeControlFlowInputChainPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AnalyzeCubeControlFlowInputChainPass)

  AnalyzeCubeControlFlowInputChainPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override {
    return "analyze-cube-control-flow-input-chain";
  }
  llvm::StringRef getDescription() const override {
    return "Analyze cube control flow input chain";
  }
};

// Pass for analyzing split-if scenarios. Rolls back the dynamic CV pipeline
// when a split-if is the only meaningful compute in both the VECTOR and CUBE
// main loops (i.e. the split-if enables no cube/vector overlap), and the same
// ssbuffer.splitted_if tag set appears on both cores.
class AnalyzeSplitIfPass
    : public PassWrapper<AnalyzeSplitIfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeSplitIfPass)

  AnalyzeSplitIfPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-split-if"; }
  llvm::StringRef getDescription() const override {
    return "Analyze split-if scenarios and roll back when split-if is the "
           "only compute in both VECTOR and CUBE main loops";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeArgsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeFlagPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeNamePass();
std::unique_ptr<OperationPass<ModuleOp>>
createAnalyzeCubeContolFLowInputChainPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeDataFlowPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeScopePass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeWhileConditionArgsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeSplitIfPass();

void registerAnalyzeDataFlowPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_ANALYZE_DATAFLOW_H
