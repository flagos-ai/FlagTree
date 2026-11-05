#ifndef TRITON_GPNPUTOLLVM_TARGETINFO_H
#define TRITON_GPNPUTOLLVM_TARGETINFO_H

#include "intellifusion/lib/TargetCapabilities.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::Intellifusion {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int threadsPerWarp, int numWarps)
      : threadsPerWarp(threadsPerWarp), numWarps(numWarps) {}

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override;
  void clusterBarrier(Location loc, RewriterBase &rewriter,
                      Operation *sourceOp) const override;

  void warpSync(Location loc, RewriterBase &rewriter) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Value val, Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  bool supportLdMatrix() const override { return target::kSupportsLdMatrix; }
  bool supportStMatrix() const override { return target::kSupportsStMatrix; }
  bool supportLdStMatrixB8() const override {
    return target::kSupportsLdStMatrixB8;
  }
  bool supportBitwidth16Elementwise() const override {
    return target::kSupportsBitwidth16Elementwise;
  }
  bool supportBitwidth32Elementwise() const override {
    return target::kSupportsBitwidth32Elementwise;
  }

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
                  triton::ReduceOp op,
                  unsigned reduceLaneIdMask) const override;

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

  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumWarps() const { return numWarps; }
  // Fixed compatibility level for the shared lowering conventions.
  int getComputeCapability() const { return 100; }
  // Fixed PTX compatibility level used by GPNPU lowering.
  int getPtxVersion() const { return 86; }

private:
  int threadsPerWarp;
  int numWarps;
};

} // namespace mlir::triton::Intellifusion

#endif // TRITON_GPNPUTOLLVM_TARGETINFO_H
