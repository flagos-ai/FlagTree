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

#ifndef TLE_TILE_TO_LLVM_UTILS_H
#define TLE_TILE_TO_LLVM_UTILS_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace mlir::triton::tle {

template <typename T1, typename T2, typename BinaryOp>
llvm::SmallVector<T2> multiDimElementwise(llvm::ArrayRef<T1> lhs,
                                          llvm::ArrayRef<T2> rhs, BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Dimensions must match");
  llvm::SmallVector<T2> result;
  result.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(static_cast<T2>(op(lhs[i], rhs[i])));
  return result;
}

llvm::SmallVector<unsigned> getCTATileOrder(::mlir::RankedTensorType type);

llvm::SmallVector<unsigned> delinearize(unsigned linearIndex,
                                        llvm::ArrayRef<unsigned> shape,
                                        llvm::ArrayRef<unsigned> order);

unsigned linearize(llvm::ArrayRef<unsigned> coords,
                   llvm::ArrayRef<unsigned> shape,
                   llvm::ArrayRef<unsigned> order);

llvm::SmallVector<unsigned> getShapePerCTATile(::mlir::RankedTensorType type);

llvm::SmallVector<::mlir::Value>
computeThreadOffsets(::mlir::Location loc,
                     ::mlir::ConversionPatternRewriter &rewriter,
                     ::mlir::RankedTensorType tensorType,
                     const ::mlir::triton::TargetInfoBase &targetInfo);

} // namespace mlir::triton::tle

#endif
