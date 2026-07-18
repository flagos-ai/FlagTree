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

#pragma once
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

#include "Transform/Passes.h"
#include "TritonToTileIR/Passes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// clang-format off
#include "mlir/InitAllPasses.h"
// clang-format on

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonCudaTileDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();

  mlir::triton::registerConvertTritonToCudaTilePass();
  mlir::triton::registerRewriteAssumeWithCudaTilePass();
  mlir::triton::registerAutoGenMemoryTokenPass();
  registry.insert<mlir::cuda_tile::CudaTileDialect>();
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                  mlir::ub::UBDialect>();
}
