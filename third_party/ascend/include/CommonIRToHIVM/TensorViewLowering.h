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

#ifndef COMMONIR_TO_HIVM_TENSORVIEW_LOWERING_H
#define COMMONIR_TO_HIVM_TENSORVIEW_LOWERING_H

#include "ascend/include/CommonIRToHIVM/TensorViewStrategy.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {

/// Element-wise padded gather into a caller-provided buffer, valid for any
/// tile rank and for both a dense boundary tile and a gather_scatter view. One
/// scf.for is nested per tile dimension; for each element an scf.if loads from
/// GM when the element is in bounds or stores `pad` otherwise. A
/// gather_scatter sparse dimension reads its logical coordinate from the
/// per-lane index tensor; every other dimension follows the dense rule
/// index * traversal + coord.
///
/// Writing element-by-element into `buffer` (rather than assembling a tensor)
/// lets a dense caller share the same destination buffer with its whole-tile
/// DMA fast path.
void emitElementwiseFill(OpBuilder &b, Location loc,
                         const TensorViewAccessInfo &info, ValueRange indices,
                         Value buffer, Value pad);

/// Unsupported views return failure(). Runtime bounds are handled in the
/// selected strategy: ScalarTransfer checks each element; PerIndexPayloadDMA
/// clips dense leading loop bounds and payload sizes, and masks sparse tuples.
/// Out-of-bounds elements retain the padding value. SingleAffineDMA is not
/// implemented or selected.
FailureOr<Value> emitGatherLoad(OpBuilder &b, Location loc,
                                const TensorViewAccessInfo &info,
                                ValueRange indices, RankedTensorType resultType,
                                hivm::AddressSpace space, Value pad);

/// Unsupported views return failure(). ScalarTransfer checks each element;
/// PerIndexPayloadDMA clips dense leading loop bounds and payload sizes, and
/// masks sparse tuples. Out-of-bounds elements are not written.
/// SingleAffineDMA is not implemented or selected.
///
/// Sparse-index conflicts are intentionally not analyzed.
LogicalResult emitScatterStore(OpBuilder &b, Location loc,
                               const TensorViewAccessInfo &info,
                               ValueRange indices, Value source,
                               hivm::AddressSpace space);

} // namespace triton
} // namespace mlir
#endif // COMMONIR_TO_HIVM_TENSORVIEW_LOWERING_H
