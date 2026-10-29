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

#ifndef COMMONIR_TO_HIVM_UTILS_H
#define COMMONIR_TO_HIVM_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {
namespace triton {
/// Cast an integer value to the MLIR index type if needed.
inline Value asIndexValue(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
}

/// Build a memref type with the requested HIVM address space.
inline MemRefType makeSpaceMemref(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  Type elementType, hivm::AddressSpace space) {
  auto spaceAttr = hivm::AddressSpaceAttr::get(ctx, space);
  return MemRefType::get(shape, elementType, MemRefLayoutAttrInterface{},
                         spaceAttr);
}

/// Create a single-element GM view at a dynamic flat element offset.
inline Value emitGmElementView(OpBuilder &b, Location loc, Value base,
                               Value offset, Type elementType) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto gmSpace =
      hivm::AddressSpaceAttr::get(b.getContext(), hivm::AddressSpace::GM);
  auto ty = MemRefType::get({1}, elementType, layout, gmSpace);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(offset),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

} // namespace triton
} // namespace mlir
#endif // COMMONIR_TO_HIVM_UTILS_H
