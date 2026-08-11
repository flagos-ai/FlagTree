#ifndef ADD_TARGET_DESCRIPTION_PASSES_H
#define ADD_TARGET_DESCRIPTION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/AddTargetDescription/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAddTargetDescriptionPass();

} // namespace triton
} // namespace mlir

#endif
