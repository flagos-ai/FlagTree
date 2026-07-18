#ifndef PROTONGPU_TO_LLVM_TARGETINFO_HCU_H
#define PROTONGPU_TO_LLVM_TARGETINFO_HCU_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/hcu/lib/TritonHCUGPUToLLVM/TargetInfo.h" // TODO(fywkevin): move hcu TargetInfo.h to include/
#include <string>

namespace mlir::triton::proton::gpu::HCU {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::HCU::TargetInfo &helper,
                      std::string arch)
      : mlir::triton::proton::gpu::TargetInfoBase(helper),
        arch(std::move(arch)) {}

  const mlir::triton::HCU::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::HCU::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  Value globalTime(ConversionPatternRewriter &rewriter,
                   Location loc) const override;

  Value processorId(ConversionPatternRewriter &rewriter,
                    Location loc) const override;

  int getAddressSpace(Attribute addressSpace) const override;

  int getIndexPtrAddrSpace() const override;

  ~TargetInfo() = default;

private:
  std::string arch;
};
} // namespace mlir::triton::proton::gpu::HCU

#endif // PROTONGPU_TO_LLVM_TARGETINFO_HCU_H
