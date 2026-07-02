/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_FOLD_STAGING_COPY_CONVERSION_PASSES_H
#define TRITON_ADAPTER_FOLD_STAGING_COPY_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace triton {

/// Fold redundant staging memref.alloc + memref.copy pairs produced by
/// TritonToLinalgIncubated's tt.load lowering.  When a staging alloc feeds
/// a bufferization.to_tensor and then a memref.copy to an on-chip buffer
/// (one with an explicit memory space), merge the two copies into one
/// direct GBM -> on-chip transfer.
std::unique_ptr<OperationPass<ModuleOp>> createFoldStagingCopyPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/FoldStagingCopy/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_FOLD_STAGING_COPY_CONVERSION_PASSES_H
