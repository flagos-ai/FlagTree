//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===- LLVMXPUToLLVMIRTranslation.cpp - Translate LLVMXPU to LLVM IR
//------------===//
//
// This file implements a translation between the MLIR LLVMXPU dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#define SDNN_ANONYMIZATION

// clang-format off
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"
#include "XPUToLLVMTranslationForSDNN.h"
#include "LLVMXPUDialectLLVMIRTranslationInterface.h"
// clang-format on

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

void mlir::registerLLVMXPUDialectTranslation(DialectRegistry &registry) {
  registerLLVMXPUSDNNDialectTranslation(registry);
  registry.insert<XPU::LLVMXPUDialect>();
  registry.addExtension(+[](MLIRContext *ctx, XPU::LLVMXPUDialect *dialect) {
    dialect->addInterfaces<LLVMXPUDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerLLVMXPUDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMXPUDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
