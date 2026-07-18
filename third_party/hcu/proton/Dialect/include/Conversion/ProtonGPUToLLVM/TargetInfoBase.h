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

#ifndef PROTONGPU_TO_LLVM_TARGETINFO_BASE_H
#define PROTONGPU_TO_LLVM_TARGETINFO_BASE_H

#include "mlir/IR/Attributes.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::proton::gpu {

class TargetInfoBase {
public:
  explicit TargetInfoBase(const mlir::triton::TargetInfoBase &helper)
      : helper(helper) {}

  virtual const mlir::triton::TargetInfoBase &getTritonTargetInfo() const {
    return helper;
  }

  // Return the local cycle counter value.
  virtual Value clock(ConversionPatternRewriter &rewriter, Location loc,
                      bool isClock64) const = 0;

  // Return the global cycle counter value (i.e., synchronized across SMs) in
  // nanoseconds, regardless of the clock frequency.
  virtual Value globalTime(ConversionPatternRewriter &rewriter,
                           Location loc) const = 0;

  virtual Value processorId(ConversionPatternRewriter &rewriter,
                            Location loc) const = 0;

  virtual int getAddressSpace(Attribute addressSpace) const = 0;

  virtual int getIndexPtrAddrSpace() const = 0;

  virtual ~TargetInfoBase() = default;

protected:
  const mlir::triton::TargetInfoBase &helper;
};
} // namespace mlir::triton::proton::gpu

#endif // PROTONGPU_TO_LLVM_TARGETINFO_BASE_H
