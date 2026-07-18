// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
