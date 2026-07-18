#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/hcu/include/Dialect/TritonHCUGPU/IR/Dialect.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonHCUGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir::triton::hcugpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL_TRITONHCUGPUOPTIMIZEDOTOPERANDS
#include "TritonHCUGPUTransforms/Passes.h.inc"

void registerTritonHCUGPUOptimizeDotOperands();
} // namespace mlir::triton::hcugpu

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonHCUGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_PASSES_H_
