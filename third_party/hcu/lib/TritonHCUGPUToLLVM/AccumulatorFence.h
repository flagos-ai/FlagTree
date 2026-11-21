#ifndef TRITON_HCU_ACCUMULATOR_FENCE_H
#define TRITON_HCU_ACCUMULATOR_FENCE_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::HCU {

// Tuple grouping is a register-allocation choice for legacy MMAC, never a
// change to lane ownership or synchronization. Other layouts keep scalar ties.
inline bool usesMMACAccumulatorTuples(Operation *op, RankedTensorType type) {
  auto module = op->getParentOfType<ModuleOp>();
  auto target =
      module ? module->getAttrOfType<StringAttr>("ttg.target") : StringAttr{};
  auto mma =
      type ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(type.getEncoding())
           : triton::gpu::AMDMfmaEncodingAttr{};
  return target && target.getValue() == "hip:gfx936" && mma && mma.isHCUMmac();
}

// Tie every actual per-thread accumulator register across a scheduling
// boundary. The empty form is useful when a pure matrix instruction is moved
// across the priority update; it retains the original memory boundary.
inline SmallVector<Value>
emitAccumulatorFence(Location loc, ArrayRef<Value> values, StringRef assembly,
                     RewriterBase &rewriter, bool mmacTuples = false) {
  assert(!values.empty());
  TritonLLVMOpBuilder b(loc, rewriter);
  bool tuple =
      mmacTuples && values.size() % 4 == 0 &&
      (assembly.empty() || assembly == "s_setprio 0" ||
       assembly == "s_setprio 1") &&
      llvm::all_of(values, [](Value value) { return value.getType().isF32(); });
  SmallVector<Value> operands(values);
  if (tuple) {
    operands.clear();
    auto vector = VectorType::get({4}, rewriter.getF32Type());
    for (unsigned group = 0; group < values.size(); group += 4) {
      Value cohort = LLVM::PoisonOp::create(rewriter, loc, vector);
      for (unsigned lane = 0; lane < 4; ++lane)
        cohort = b.insert_element(vector, cohort, values[group + lane],
                                  b.i32_val(lane));
      operands.push_back(cohort);
    }
  }
  SmallVector<Type> types;
  std::string constraints;
  for (Value value : operands) {
    types.push_back(value.getType());
    if (!constraints.empty())
      constraints += ',';
    constraints += "=v";
  }
  for (unsigned i = 0; i < operands.size(); ++i)
    constraints += ',' + std::to_string(i);
  constraints += ",~{memory}";
  Type resultType =
      types.size() == 1
          ? types.front()
          : LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value result = LLVM::InlineAsmOp::create(
                     rewriter, loc, resultType, operands, assembly, constraints,
                     /*has_side_effects=*/true, /*is_align_stack=*/false,
                     LLVM::TailCallKind::None,
                     LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                               LLVM::AsmDialect::AD_ATT),
                     ArrayAttr())
                     ->getResult(0);
  auto outputs = types.size() == 1 ? SmallVector<Value>{result}
                                   : unpackLLElements(loc, result, rewriter);
  if (!tuple)
    return outputs;
  SmallVector<Value> scalars;
  for (Value cohort : outputs)
    for (unsigned lane = 0; lane < 4; ++lane)
      scalars.push_back(
          b.extract_element(rewriter.getF32Type(), cohort, b.i32_val(lane)));
  return scalars;
}

} // namespace mlir::triton::HCU

#endif
