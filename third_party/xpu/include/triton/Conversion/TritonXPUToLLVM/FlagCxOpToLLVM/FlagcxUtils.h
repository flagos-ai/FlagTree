#ifndef TRITON_XPU_FLAGCX_UTILS_H
#define TRITON_XPU_FLAGCX_UTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>

namespace mlir::triton::xpu {

LLVM::LLVMPointerType getFlagcxGlobalPtrType(MLIRContext *context);

FailureOr<LLVM::CallOp> getDeviceInitFuncCall(Operation *op,
                                             ConversionPatternRewriter &rewriter,
                                             Value comm, Value scratch);

FailureOr<LLVM::CallOp> getLocalPeFuncCall(Operation *op,
                                          ConversionPatternRewriter &rewriter,
                                          Value comm);

FailureOr<LLVM::CallOp> getNumPesFuncCall(Operation *op,
                                         ConversionPatternRewriter &rewriter,
                                         Value comm);

FailureOr<LLVM::CallOp>
getIntraPointerFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                        Value memory, Value byteOffset, Value peer);

FailureOr<LLVM::CallOp>
getIntraBarrierFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                        Value comm, int32_t coopKind, int32_t barrierIndex,
                        int32_t order, StringRef barrierKind);

FailureOr<LLVM::CallOp>
getSignalFuncCall(Operation *op, ConversionPatternRewriter &rewriter, Value comm,
                  int32_t teamKind, Value peer, Value slot, Value addend,
                  int32_t contextId, int32_t coopKind, int32_t scope);

FailureOr<LLVM::CallOp>
getDevNetWaitFuncCallByKind(Operation *op, ConversionPatternRewriter &rewriter,
                            Value comm, Value slot, Value least,
                            int32_t contextId, int32_t coopKind, int32_t order,
                            StringRef waitKind);

} // namespace mlir::triton::xpu

#endif // TRITON_XPU_FLAGCX_UTILS_H
