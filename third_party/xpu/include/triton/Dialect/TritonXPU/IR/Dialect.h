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

#ifndef TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_

// TritonXPUDialect
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h" // cf
#include "triton/Dialect/Triton/IR/Dialect.h"           // arith/scf/math/triton
#include "triton/Dialect/TritonGPU/IR/Dialect.h"        // SliceEncodingAttr

#include "triton/Dialect/TritonXPU/IR/Dialect.h.inc" // TritonXPUDialect
#include "triton/Dialect/TritonXPU/IR/TritonXPUEnums.h.inc"

// TritonXPUAttr
#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrInterfaces.h.inc"
namespace mlir {
namespace triton {
namespace xpu {
// Bring CTAEncodingAttr into the xpu namespace so the tablegen-generated
// declaration `CTAEncodingAttr getCTALayout() const;` (inserted by the
// LayoutEncodingTrait interface) resolves correctly.
using ::mlir::triton::gpu::CTAEncodingAttr;
} // namespace xpu
} // namespace triton
} // namespace mlir
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.h.inc"

// TritonXPUOps
#define GET_OP_CLASSES
#include "triton/Dialect/TritonXPU/IR/Ops.h.inc"

// TritonXPUTypes
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/Types.h.inc"

namespace mlir {
namespace triton {
namespace xpu {

unsigned getTotalElemsPerThread(Type eltTy);

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy);

unsigned getGroupSize(Attribute layout);

// Return a blocked encoding where the shape is distributed contiguously amongst
// the threads, warps, CTAs with 1 element per threads.
triton::xpu::ClusterLayoutAttr
getDefaultClusterEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          uint32_t buffer_size, uint32_t core_num);

SmallVector<unsigned>
getCoresPerClusterWithUniqueData(Attribute layout,
                                 ArrayRef<int64_t> tensorShape);

SmallVector<unsigned>
getCoresPerGroupWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape);

SmallVector<unsigned> getUniqueContigPerCore(Attribute layout,
                                             ArrayRef<int64_t> shape);

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
