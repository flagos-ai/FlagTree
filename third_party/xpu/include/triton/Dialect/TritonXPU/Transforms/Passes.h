#ifndef TRITON_DIALECT_TRITONXPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONXPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/NewAnalysis/Utility.h" // helper
#include "triton/Dialect/TritonXPU/IR/Dialect.h" // dependentDialects
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h" // TypeSwitch
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h" // llvm_unreachable

namespace mlir {
namespace triton {
namespace xpu {

constexpr llvm::StringLiteral kBF16ToFP32VecOptOffAttrName =
    "triton_xpu.bf16_to_fp32_vec_opt_off";

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

/// Factory that hands the compiled payloads (raw source id -> LLVM IR text) to
/// the materialization pass. The map is intentionally kept off the PassOptions
/// surface; see the Passes.td comment on TritonXPUMaterializeDeferredRaw.
std::unique_ptr<mlir::Pass> createTritonXPUMaterializeDeferredRawWithSources(
    const llvm::StringMap<std::string> &sources);

} // namespace xpu
} // namespace triton
} // namespace mlir
#endif
