#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_TOOLS_FLAGCXUTILS_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_TOOLS_FLAGCXUTILS_H_

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::iluvatar_tle {

LLVM::CallOp getLocalPeFuncCall(mlir::Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value memPtrInt);

LLVM::CallOp getNumPesFunCall(mlir::Location loc,
                              ConversionPatternRewriter &rewriter,
                              Value memPtrInt);

LLVM::CallOp getBarrierFuncCall(mlir::Location loc,
                                ConversionPatternRewriter &rewriter, Value comm,
                                size_t barrier_index, FlagCXCoopKind coopKind,
                                MemoryOrder order, llvm::StringRef barrierType);

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_TOOLS_FLAGCXUTILS_H_
