#ifndef TRITON_CONVERSION_TRITONTOTILEIR_PASS_H
#define TRITON_CONVERSION_TRITONTOTILEIR_PASS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

#include "Utils.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToCudaTilePass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToCudaTilePass(bool approx, bool ftz, int capability,
                                  int num_ctas, int simt_num_warps,
                                  int occupancy, std::optional<int> num_stages);

} // namespace triton

namespace cuda_tile {
void legalize_agent_captures(Operation *rop);
} // namespace cuda_tile

} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOTILEIR_PASS_H
