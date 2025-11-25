#ifndef ILUVATAR_TRITON_ANALYSIS_UTILITY_H
#define ILUVATAR_TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"

#define FLAGTREE_SPEC_Utility_isMmaToDotSlowShortcut
#define FLAGTREE_SPEC_Utility_getBackwardSliceCorex
#define FLAGTREE_SPEC_Utility_getBackwardSliceImplCorex
#define FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG bool
#define FLAGTREE_SPEC_Analysis_Utility_maybeSharedAllocationOp
#define FLAGTREE_SPEC_Analysis_Utility_supportMMA
#define FLAGTREE_SPEC_Analysis_Utility_isMmaToMmaShortcut
#define FLAGTREE_SPEC_Analysis_Utility_isMmaToDotShortcut

namespace mlir {

SetVector<Operation *> multiRootGetSlice(
    Operation *op, TransitiveFilter backwardFilter = nullptr,
    TransitiveFilter forwardFilter = nullptr,
    FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG omitBlockArguments = true);

} // namespace mlir

#endif // ILUVATAR_TRITON_ANALYSIS_UTILITY_H
