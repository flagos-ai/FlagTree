//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TX81_TO_LLVM_CONVERSION_PASSES_H
#define TX81_TO_LLVM_CONVERSION_PASSES_H

#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Tx81ToLLVM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TX81_TO_LLVM_CONVERSION_PASSES_H
