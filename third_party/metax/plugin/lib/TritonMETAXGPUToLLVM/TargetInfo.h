#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOMETAX_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOMETAX_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::METAX {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability, int ptxVersion)
      : computeCapability(computeCapability), ptxVersion(ptxVersion) {}

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  Value ballotIntrinsic(RewriterBase &rewriter, Location loc, Type type,
                        Value cmp, Operation *op) const;

  void barrier(Location loc, RewriterBase &rewriter,
               bool isWarpSync = false) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;
  Value loadDSharedIntrinsic(RewriterBase &rewriter, Location loc, Value ptr,
                             std::optional<Value> ctaId, Type elemTy,
                             Value pred,
                             Operation *localLoadOp = nullptr) const override;

  bool supportLdMatrix() const override { return false; }
  bool supportStMatrix() const override { return false; }

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleXorIntrinsic(RewriterBase &rewriter, Location loc, Value val,
                            int i, Operation *op) const;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleUpIntrinsic(RewriterBase &rewriter, Location loc, Value val,
                           int i, Operation *op) const;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;
  Value shuffleIdxIntrinsic(RewriterBase &rewriter, Location loc, Value val,
                            int i, Operation *op) const;
  Value shuffleIdxIntrinsic(RewriterBase &rewriter, Location loc, Value val,
                            Value i, Operation *op) const;

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

} // namespace mlir::triton::METAX

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOMETAX_H
