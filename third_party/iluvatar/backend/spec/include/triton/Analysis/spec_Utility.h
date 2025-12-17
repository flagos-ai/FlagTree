#ifndef ILUVATAR_TRITON_ANALYSIS_SPEC_UTILITY_H
#define ILUVATAR_TRITON_ANALYSIS_SPEC_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

bool isMmaToDotShortcutForB(RankedTensorType srcTy, RankedTensorType dstTy);
bool areLayoutsEquivalent(Attribute srcLayout, Attribute dstLayout,
                          ArrayRef<int64_t> srcShape,
                          ArrayRef<int64_t> dstShape);

} // namespace mlir

#endif // ILUVATAR_TRITON_ANALYSIS_SPEC_UTILITY_H
