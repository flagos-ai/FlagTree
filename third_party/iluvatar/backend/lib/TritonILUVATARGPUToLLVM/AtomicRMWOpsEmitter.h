#ifndef TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_ATOMICRMWOPSEMITTER_H_
#define TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_ATOMICRMWOPSEMITTER_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/Utility.h"

namespace mlir::LLVM::ILUVATAR {

class AtomicRMWEmitter {
public:
  AtomicRMWEmitter(const mlir::triton::ILUVATAR::TargetInfo &targetInfo,
                   LLVM::AtomicBinOp binOp, LLVM::AtomicOrdering memOrder,
                   StringRef scopeStr)
      : targetInfo(targetInfo), binOp(binOp), memOrder(memOrder),
        scopeStr(scopeStr) {}

  Value emitAtomicRMW(RewriterBase &rewriter, Value rmwPtr, Value valElem,
                      Value rmwMask, std::optional<Value> sharedMemBase) const;

  void setAtomicOrdering(LLVM::AtomicOrdering memOrder) {
    this->memOrder = memOrder;
  }

private:
  const mlir::triton::ILUVATAR::TargetInfo &targetInfo;

  mlir::LLVM::AtomicBinOp binOp;
  mlir::LLVM::AtomicOrdering memOrder;
  std::string scopeStr;
};

} // namespace mlir::LLVM::ILUVATAR

#endif // TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_ATOMICRMWEMITTER_H_
