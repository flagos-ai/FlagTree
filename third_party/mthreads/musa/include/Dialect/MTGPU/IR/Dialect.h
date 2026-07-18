#ifndef TRITON_DIALECT_MTGPU_IR_DIALECT_H_
#define TRITON_DIALECT_MTGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/ADT/StringRef.h"

// clang-format off
#include "Dialect/MTGPU/IR/Dialect.h.inc"
#include "Dialect/MTGPU/IR/OpsEnums.h.inc"
// clang-format on

#define GET_TYPEDEF_CLASSES
#include "Dialect/MTGPU/IR/MTGPUTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/MTGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace mtgpu {} // namespace mtgpu
} // namespace triton
} // namespace mlir

#endif
