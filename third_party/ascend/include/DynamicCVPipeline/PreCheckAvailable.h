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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PRE_CHECK_AVAILABLE_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PRE_CHECK_AVAILABLE_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

class PreCheckBlacklistPass
    : public PassWrapper<PreCheckBlacklistPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreCheckBlacklistPass)

  PreCheckBlacklistPass() = default;

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  ::llvm::StringRef getArgument() const override {
    return "pre-check-blacklist";
  }
  ::llvm::StringRef getDescription() const override {
    return "Check whether the input module contains any blacklist operations";
  }
};

class PreCheckMatmul
    : public PassWrapper<PreCheckMatmul, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreCheckMatmul)

  PreCheckMatmul() = default;

  void runOnOperation() override;

  ::llvm::StringRef getArgument() const override { return "pre-check-matmul"; }
  ::llvm::StringRef getDescription() const override {
    return "Check whether the input module contains matmul operations.";
  }
};

class PreCheckAvailablePass
    : public PassWrapper<PreCheckAvailablePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreCheckAvailablePass)

  PreCheckAvailablePass() = default;

  void runOnOperation() override;

  static constexpr ::llvm::StringRef getArgumentName() {
    return "pre-check-dynamic-cv-pipeline-available";
  }
  ::llvm::StringRef getArgument() const override { return getArgumentName(); }
  ::llvm::StringRef getDescription() const override {
    return "Check whether dynamic CV pipeline can process the input module";
  }
  ::llvm::StringRef getName() const override { return "PreCheckAvailablePass"; }
};

class PreCheckDisablePreload
    : public PassWrapper<PreCheckDisablePreload, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreCheckDisablePreload)

  PreCheckDisablePreload() = default;

  void runOnOperation() override;

  ::llvm::StringRef getArgument() const override {
    return "pre-disable-tuple-preload";
  }
  ::llvm::StringRef getDescription() const override {
    return "Check whether the input module should disable the 3-preload "
           "feature";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckBlacklistPass();
std::unique_ptr<OperationPass<ModuleOp>> createPreCheckMatmulPass();
std::unique_ptr<OperationPass<ModuleOp>> createPreCheckAvailablePass();
std::unique_ptr<OperationPass<ModuleOp>> createPreCheckDisablePreloadPass();
void registerPreCheckAvailablePasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PRE_CHECK_AVAILABLE_H
