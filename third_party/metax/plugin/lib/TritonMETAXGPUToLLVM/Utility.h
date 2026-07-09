#ifndef TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators

namespace mlir::triton::gpu {
class MemDescType;

SmallVector<Value> emitSwizzleColOffsets(RewriterBase &rewriter, Operation *op,
                                         RankedTensorType srcTy,
                                         ArrayRef<int64_t> shape,
                                         Attribute swizzledSharedEnc,
                                         Attribute noSwizzledSharedEnc,
                                         unsigned vec);
} // namespace mlir::triton::gpu

namespace mlir {
namespace LLVM {

namespace METAX {

class TargetInfo;

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleXorIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          int i, Operation *op = nullptr);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUpIntrinsic(Location loc, RewriterBase &rewriter, Value val, int i,
                         Operation *op);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);
Value shuffleIdxIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          int i, Operation *op);
Value shuffleIdxIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          Value i, Operation *op);
Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

// Create bar.warp.sync
void createSyncWarp(Location loc, OpBuilder &builder);

// Lower ldmatrix and stmatrix
LogicalResult lowerLdStMatrix(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const mlir::triton::METAX::TargetInfo &targetInfo);

Type getFunctionType(Type resultType, const ValueRange &operands);

LLVM::LLVMFuncOp appendOrGetFuncOp(OpBuilder &rewriter, Operation *op,
                                   StringRef funcName, Type funcType);

__attribute__((optimize("O0"))) __attribute__((noinline)) Value
createBuiltinFunc(OpBuilder &rewriter, Location loc, Operation *op,
                  StringRef funcName, Type resultType,
                  const ValueRange &argValues);

void appendIntrinsicModifer(std::string &str, int vec, Type elemType);
// Convert bfloat16 to float32
Value convertBf16ToFp32(Location loc, RewriterBase &rewriter, const Value &v);

// Convert float32 to bfloat16
Value convertFp32ToBf16(Location loc, Operation *op, RewriterBase &rewriter,
                        const Value &v,
                        const mlir::triton::RoundingMode rounding =
                            mlir::triton::RoundingMode::RTNE);

} // namespace METAX

#ifdef USE_MACA

Type getFunctionType(Type resultType, const ValueRange &operands);

template <typename T>
LLVM::LLVMFuncOp appendOrGetFuncOp(OpBuilder &rewriter, T op,
                                   StringRef funcName, Type funcType) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->template getParentOfType<LLVMFuncOp>());
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  return ret;
}

template <typename T>
__attribute__((optimize("O0"))) __attribute__((noinline)) Value
createBuiltinFunc(OpBuilder &rewriter, Location loc, T op, StringRef funcName,
                  Type resultType, const ValueRange &argValues) {
  Type funcType = getFunctionType(resultType, argValues);
  auto funcOp = appendOrGetFuncOp<T>(rewriter, op, funcName, funcType);
  return rewriter.create<LLVM::CallOp>(loc, funcOp, argValues).getResult();
}

#endif
} // namespace LLVM

} // namespace mlir

#endif
