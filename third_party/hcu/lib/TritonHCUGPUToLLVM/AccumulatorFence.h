#ifndef TRITON_HCU_ACCUMULATOR_FENCE_H
#define TRITON_HCU_ACCUMULATOR_FENCE_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::HCU {

// Tie every actual per-thread accumulator register across a scheduling
// boundary. The empty form is useful when a pure matrix instruction is moved
// across the priority update; it retains the original memory boundary.
inline SmallVector<Value> emitAccumulatorFence(Location loc,
                                               ArrayRef<Value> values,
                                               StringRef assembly,
                                               RewriterBase &rewriter) {
  assert(!values.empty());
  SmallVector<Type> types;
  std::string constraints;
  for (Value value : values) {
    types.push_back(value.getType());
    if (!constraints.empty())
      constraints += ',';
    constraints += "=v";
  }
  for (unsigned i = 0; i < values.size(); ++i)
    constraints += ',' + std::to_string(i);
  constraints += ",~{memory}";
  Type resultType = types.size() == 1
                        ? types.front()
                        : LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                          types);
  Value result = LLVM::InlineAsmOp::create(
      rewriter, loc, resultType, values, assembly, constraints,
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT),
      ArrayAttr())->getResult(0);
  return unpackLLElements(loc, result, rewriter);
}

} // namespace mlir::triton::HCU

#endif
