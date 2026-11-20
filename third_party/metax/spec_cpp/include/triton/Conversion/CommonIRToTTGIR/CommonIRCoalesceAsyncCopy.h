#ifndef TRITON_METAX_COMMON_IR_COALESCE_ASYNC_COPY_H
#define TRITON_METAX_COMMON_IR_COALESCE_ASYNC_COPY_H

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {

gpu::BlockedEncodingAttr
getAsyncCopySourcePreference(gpu::AsyncCopyGlobalToLocalOp copy,
                             ModuleAxisInfoAnalysis &axisInfo);

bool canWriteMmaSharedLayout(gpu::AsyncCopyGlobalToLocalOp copy,
                             gpu::MemDescType destination,
                             ModuleAxisInfoAnalysis &axisInfo);

} // namespace mlir::triton::metax

#endif
