/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ascend/include/CommonIRToHIVM/TensorViewStrategy.h"

namespace mlir {
namespace triton {

/// Select scalar transfer unless per-index payload DMA is supported.
FailureOr<TensorViewPlan>
chooseTensorViewPlan(const TensorViewAccessInfo &view,
                     const GatherScatterAccessInfo &access) {
  TensorViewPlan plan;

  // The payload is the dense suffix after the last sparse dim. Leading sparse
  // dims read index tensors; leading dense dims use index * traversal + coord.
  // Either ordering uses the same payload DMA; sparse dims need not be a
  // prefix.
  if (access.sparseDims.empty())
    return plan;

  // An innermost sparse dim leaves no dense payload; use scalar transfer.
  if (access.payload.isScalar())
    return plan;

  // A strideable payload can use one DMA even with gaps in the base layout.
  // emitPayloadGmView preserves base strides. Fall back to scalar transfer when
  // strideability cannot be proved, including transposed or dynamic strides.
  if (!access.payload.affine || !access.payload.staticallyStrideable)
    return plan;

  if (access.sparseIndices.size() != access.sparseDims.size())
    return failure();

  // Keep Segmented and General on scalar transfer. Check every sparse dim,
  // since each has its own index pattern.
  for (const IndexPatternInfo &index : access.sparseIndices)
    if (index.kind == IndexPatternKind::Segmented ||
        index.kind == IndexPatternKind::General)
      return plan;

  // Per-index DMA handles the remaining patterns by resolving each coordinate.
  // Do not select SingleAffineDMA until its gather/scatter emitters are
  // implemented; their failure() stubs would reject otherwise supported views.
  plan.strategy = TensorViewStrategy::PerIndexPayloadDMA;
  return plan;
}

} // namespace triton
} // namespace mlir
