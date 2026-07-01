#ifndef TRITON_DIALECT_LLVMXPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_LLVMXPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/NewAnalysis/Utility.h" // helper
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"   // dependentDialects
#include "llvm/ADT/TypeSwitch.h"                 // TypeSwitch
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h" // llvm_unreachable

namespace mlir {
namespace LLVM {
namespace XPU {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h.inc"

} // namespace XPU
} // namespace LLVM
} // namespace mlir
#endif
