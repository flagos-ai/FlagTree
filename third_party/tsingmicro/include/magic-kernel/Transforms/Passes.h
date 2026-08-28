#ifndef MK_TRANSFORMS_PASSES_H
#define MK_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeStridedLinalgInputsPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "magic-kernel/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // MK_TRANSFORMS_PASSES_H