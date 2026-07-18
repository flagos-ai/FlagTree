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
