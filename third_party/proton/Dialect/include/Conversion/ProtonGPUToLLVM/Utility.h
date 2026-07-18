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

#ifndef PROTONGPU_TO_LLVM_UTILITY_H
#define PROTONGPU_TO_LLVM_UTILITY_H

#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

Value getRawThreadId(OpBuilder &rewriter, Location loc);

namespace LLVM {

struct SegmentObject {
  Value base;
  Value segmentBase;
  Value indexPtr;

  SegmentObject(Value base, Value segmentBase, Value indexPtr)
      : base(base), segmentBase(segmentBase), indexPtr(indexPtr) {}

  Value getStruct(Location loc, ConversionPatternRewriter &rewriter);

  static LLVMStructType getStructType(MLIRContext *ctx, int memorySpace,
                                      int indexPtrAddrSpace);

  static SegmentObject fromStruct(Location loc, Value segmentStruct,
                                  ConversionPatternRewriter &rewriter);
};

} // namespace LLVM

namespace triton {
namespace proton::gpu {

struct CircularStoreDataPack {
  Value isWriter;
  Value record;
  Value ptr;
  uint32_t addrSpace;
};

CircularStoreDataPack
lowerCircularStoreOpHelper(CircularStoreOp op, Value segmentStruct,
                           ConversionPatternRewriter &rewriter);

SmallVector<FunctionOpInterface> getTritonFunctions(ModuleOp mod);

} // namespace proton::gpu
} // namespace triton

} // namespace mlir

#endif // PROTONGPU_TO_LLVM_UTILITY_H
