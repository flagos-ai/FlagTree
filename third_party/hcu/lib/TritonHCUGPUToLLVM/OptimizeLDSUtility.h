// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUTOLLVM_OPTIMIZELDSUTILITY_H_
#define TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUTOLLVM_OPTIMIZELDSUTILITY_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::HCU {

std::vector<SmallVector<unsigned>> factorizePowerOf2(int n, int rank);

/// Copy given layout with different warpsPerCTA parameter
///
/// \param layout original layout
/// \param warpsPerCTA new warpsPerCTA
/// \returns create layout
triton::gpu::DistributedEncodingTrait
createTmpLayout(triton::gpu::DistributedEncodingTrait layout,
                ArrayRef<unsigned> warpsPerCTA);

/// Creates two chained convert layout operations
///
/// %1 = cvtOp %0 (srcLayout -> dstLayout) // original operation
/// ->
/// %2 = cvtOp %0 (srcLayout -> tmpLayout) // <returned pair>.first
/// %3 = cvtOp %2 (tmpLayout -> dstLayout) // <returned pair>.second
///
/// \param builder
/// \param cvtOp original operation
/// \param tmpLayout
/// \returns pair of created operations
std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(OpBuilder &builder, triton::gpu::ConvertLayoutOp &cvtOp,
                    Attribute tmpLayout);

struct Resources {
  int LDSPad;
  int LDSSwizzle;
};

Resources
estimateResourcesForReplacement(OpBuilder builder,
                                mlir::triton::gpu::ConvertLayoutOp cvtOp,
                                Attribute tmpLayout);

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUTOLLVM_OPTIMIZELDSUTILITY_H_
