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

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::NVIDIA {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability, int ptxVersion)
      : computeCapability(computeCapability), ptxVersion(ptxVersion) {}

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override;

  void warpSync(Location loc, RewriterBase &rewriter) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  bool supportLdMatrix() const override { return computeCapability >= 75; }
  bool supportStMatrix() const override { return computeCapability >= 90; }
  bool supportLdStMatrixB8() const override { return computeCapability >= 100; }

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,

              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override;

  int getPtxVersion() const { return ptxVersion; }
  int getComputeCapability() const { return computeCapability; }

  bool isCuda() const override { return true; }

private:
  int computeCapability;
  int ptxVersion;
};

} // namespace mlir::triton::NVIDIA

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
