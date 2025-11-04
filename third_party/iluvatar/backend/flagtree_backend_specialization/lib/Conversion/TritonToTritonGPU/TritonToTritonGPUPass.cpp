#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

constexpr static char AttrNumStagesForDot[] = "triton_gpu.dot.num-stages";

void ConvertTritonToTritonGPU_setAttrNumStagesForDot(ModuleOp &mod,
                                                     IntegerType i32_ty,
                                                     int numStages) {
  mod->setAttr(AttrNumStagesForDot,
               IntegerAttr::get(i32_ty, llvm::APInt(32, numStages)));
}

} // namespace triton
} // namespace mlir
