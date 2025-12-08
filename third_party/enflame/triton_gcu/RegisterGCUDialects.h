/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCU_REGISTER_GCU_DIALECT_H
#define GCU_REGISTER_GCU_DIALECT_H

#include "Conversion/Passes.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"

namespace mlir {
namespace gcu {

inline void registerGCUDialects(mlir::DialectRegistry &registry) {
  // dialects
  mlir::registerAllDialects(registry);
  registry.insert<mlir::gcu::GCUDialect, mlir::triton::gcu::TritonGCUDialect,
                  mlir::memref_ext::MemrefExtDialect,
                  mlir::math_ext::MathExtDialect>();

  // extensions
  mlir::registerAllExtensions(registry);

  // passes
  mlir::registerAllPasses();

  mlir::registerTritonGCUConversionPasses();
  mlir::registerTritonGCUTransformsPasses();

  // translation
  mlir::registerAllTranslations();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerX86VectorDialectTranslation(registry);

  // Extension required for translating GPU offloading Ops.
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}

} // namespace gcu
} // namespace mlir

#endif // GCU_REGISTER_GCU_DIALECT_H
