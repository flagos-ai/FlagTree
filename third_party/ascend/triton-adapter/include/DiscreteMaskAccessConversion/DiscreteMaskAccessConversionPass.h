#ifndef TRITON_ADAPTER_DISCRETEMASKACCESSCONVERSION_H
#define TRITON_ADAPTER_DISCRETEMASKACCESSCONVERSION_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_CLASSES
#include "../../include/DiscreteMaskAccessConversion/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createDiscreteMaskAccessConversionPass();

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

class DiscreteMaskAccessConversionPass
    : public DiscreteMaskAccessConversionBase<DiscreteMaskAccessConversionPass> {
public:

  void runOnOperation() override;
};

} // namespace

#endif // DISCRETE_MASK_ACCESS_CONVERSION_H