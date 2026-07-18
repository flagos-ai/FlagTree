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

#ifndef LLVMXPU_DIALECT_LLVMIR_TRANSLATION_INTERFACE_H
#define LLVMXPU_DIALECT_LLVMIR_TRANSLATION_INTERFACE_H

// clang-format off
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsXPU.h" //llvm::Intrinsic
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "XPUToLLVMTranslationForSDNN.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"
// clang-format on

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMXPU dialect to LLVM IR.
class LLVMXPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
    if (SDNNConvertOperation(opInst, builder, moduleTranslation).succeeded())
      return success();

    // Manually handle ops that fail with createIntrinsicCall on trust LLVM
    if (auto coreIdOp = dyn_cast<::mlir::LLVM::XPU::CoreIdOp>(opInst)) {
      auto *module = builder.GetInsertBlock()->getModule();
      auto *i32Ty = builder.getInt32Ty();
      auto *funcTy = llvm::FunctionType::get(i32Ty, {}, false);
      auto callee = module->getOrInsertFunction("llvm.xpu.core_id", funcTy);
      moduleTranslation.mapValue(coreIdOp.getRes()) =
          builder.CreateCall(funcTy, callee.getCallee(), {});
      return success();
    }
    if (auto clusterIdOp = dyn_cast<::mlir::LLVM::XPU::ClusterIdOp>(opInst)) {
      auto *module = builder.GetInsertBlock()->getModule();
      auto *i32Ty = builder.getInt32Ty();
      auto *funcTy = llvm::FunctionType::get(i32Ty, {}, false);
      auto callee = module->getOrInsertFunction("llvm.xpu.cluster_id", funcTy);
      moduleTranslation.mapValue(clusterIdOp.getRes()) =
          builder.CreateCall(funcTy, callee.getCallee(), {});
      return success();
    }

#include "triton/Dialect/LLVMXPU/IR/LLVMXPUConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    auto generateMetadata = [&](int dim, StringRef name) {
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmContext, name),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmContext), dim))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmContext, llvmMetadata);
      moduleTranslation.getOrInsertNamedModuleMetadata("xpu.annotations")
          ->addOperand(llvmMetadataNode);
    };

    return success();
  }
};

#endif
