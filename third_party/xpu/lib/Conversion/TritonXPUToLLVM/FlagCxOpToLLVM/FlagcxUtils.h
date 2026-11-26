#ifndef TRITON_XPU_FLAGCX_UTILS_H
#define TRITON_XPU_FLAGCX_UTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::xpu {

enum class FlagcxDeviceFunction {
  Init,
  GetIntraRank,
  GetIntraSize,
  GetIntraPointer,
  IntraBarrierArrive,
  IntraBarrierWait,
  IntraBarrierSync,
  SignalInc,
  SignalAdd,
  WaitSignal,
  WaitCounter,
  WaitSignalMeetShadow,
};

LLVM::LLVMPointerType getFlagcxGlobalPtrType(MLIRContext *context);

// The first argument of every entry point is a device communicator or memory
// handle, represented as an integer in the XPU dialect.
FailureOr<LLVM::CallOp>
createFlagcxDeviceCall(Operation *op, ConversionPatternRewriter &rewriter,
                       FlagcxDeviceFunction function, Value handle,
                       ValueRange args = {});

} // namespace mlir::triton::xpu

#endif // TRITON_XPU_FLAGCX_UTILS_H
