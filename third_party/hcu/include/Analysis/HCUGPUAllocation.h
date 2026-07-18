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

#ifndef TRITONHCU_ANALYSIS_HCUGPU_ALLOCATION_H
#define TRITONHCU_ANALYSIS_HCUGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::HCU {

constexpr char AttrSharedMemPadded[] = "hcug.use_padded_scratch_shmem";

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        bool usePadding);

unsigned HCUAllocationAnalysisScratchSizeFn(Operation *op);

// To convert a tensor from one layout to another, we need to allocate a
// temporary buffer (i.e., scratch buffer) in shared memory. The conversion may
// require multiple iterations, with each iteration involving multiple
// vectorized loads/stores. The scratch buffer has a shape (`repShape`) that
// represents the maximum size accessed in each dimension during each iteration.
// It is padded (`paddedRepShape`) to avoid bank conflicts and is accessed in a
// specific `order`.
struct ScratchConfig {
  SmallVector<unsigned> repShape;
  SmallVector<unsigned> paddedRepShape;
  SmallVector<unsigned> order;
  unsigned inVec;
  unsigned outVec;

  ScratchConfig(SmallVector<unsigned> repShape,
                SmallVector<unsigned> paddedRepShape, unsigned inVec = 1,
                unsigned outVec = 1)
      : repShape(repShape), paddedRepShape(paddedRepShape), inVec(inVec),
        outVec(outVec) {}

  void print(llvm::raw_ostream &os) const {
    os << "repShape: [";
    llvm::interleaveComma(repShape, os);
    os << "]";
    os << ", paddedRepShape: [";
    llvm::interleaveComma(paddedRepShape, os);
    os << "]";
    os << ", order: [";
    llvm::interleaveComma(order, os);
    os << "]";
    os << ", inVec: " << inVec << ", outVec: " << outVec << "\n";
  }
};

// For a layout conversion between `srcTy` and `dstTy`, return the vector length
// that can be used for the stores to and loads from shared memory,
// respectively.
std::pair</*inVec*/ unsigned, /*outVec*/ unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy);

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy);

} // namespace mlir::triton::HCU

#endif // TRITONHCU_ANALYSIS_HCUGPU_ALLOCATION_H
