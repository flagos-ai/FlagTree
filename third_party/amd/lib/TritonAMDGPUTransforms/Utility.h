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

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_

#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

// DFS the def chain of 'defValue' starting from 'consumer' and will return the
// minimum found when accumulating countFunc(op) for all non control flow ops
// between value and the consumer. This function will traverse through for loop
// iterations and to the outside of the loop to find all its producers.
//    CountOp(Operation*) should return the value to accumulate for the
//    operation
// Returns 0 if there is an error traversing the def chain
int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc);

// Returns a padded shared encoding minimizing bank conflicts for the given
// tensor and dot encoding.
triton::gpu::PaddedSharedEncodingAttr
composePaddedLayout(const triton::AMD::TargetInfo &targetInfo,
                    triton::gpu::DotOperandEncodingAttr dotOpEnc,
                    triton::gpu::TensorOrMemDesc srcTy,
                    ArrayRef<unsigned> sharedOrder, bool useAsyncCopy);

#endif
