/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_ERASE_LINALG_CASTS_CONVERSION_PASSES_H
#define TRITON_ADAPTER_ERASE_LINALG_CASTS_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace triton {

/// Lower the unrealized_conversion_cast ops left by TileIRToHIVM into
/// legitimate MLIR bridges (bufferization.to_memref / bufferization.to_tensor /
/// tt.load) so the subsequent triton-to-linalg-incubated pass produces
/// cast-free linalg-dialect IR.
std::unique_ptr<OperationPass<ModuleOp>> createEraseLinalgCastsPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/EraseLinalgCasts/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ERASE_LINALG_CASTS_CONVERSION_PASSES_H
