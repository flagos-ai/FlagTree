#ifndef TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_UTILITY_H
#define TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_UTILITY_H

#include "Dialect/MTGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace LLVM {
namespace MUSA {

inline constexpr char Predicated_Load[] = "__predicated_load";
inline constexpr char Predicated_InplaceLoad[] = "__predicated_inplace_load";
inline constexpr char Predicated_Store[] = "__predicated_store";

struct SqmmaAccumulatorCarrierInfo {
  RankedTensorType tensorType;
  unsigned fragmentCount;
  unsigned fragmentElems;
  Type fragmentType;
  Type carrierType;
};

FailureOr<SqmmaAccumulatorCarrierInfo>
getSqmmaAccumulatorCarrierInfo(Type type);

SmallVector<Value> unpackSqmmaAccumulatorCarrier(Location loc, Value carrier,
                                                 Type type,
                                                 RewriterBase &rewriter);
Value packSqmmaAccumulatorCarrier(Location loc, ValueRange fragments, Type type,
                                  RewriterBase &rewriter);
Value carrierFragmentToMathVec(Location loc, Value fragment, Type type,
                               RewriterBase &rewriter);
Value mathVecToCarrierFragment(Location loc, Value mathVec, Type type,
                               RewriterBase &rewriter);
Value packSqmmaAccumulatorCarrierFromTensor(Location loc, Value tensorValue,
                                            RankedTensorType tensorType,
                                            const LLVMTypeConverter *converter,
                                            RewriterBase &rewriter);
Value unpackSqmmaAccumulatorCarrierToTensor(Location loc, Value carrier,
                                            RankedTensorType tensorType,
                                            const LLVMTypeConverter *converter,
                                            RewriterBase &rewriter);

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                unsigned width);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 unsigned width);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               triton::ProgramIDDim axis);

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal);

Value llInplaceLoad(RewriterBase &rewriter, Location loc, Value ptr,
                    Type elemTy, Value pred, Value falseVal);

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred);

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask);

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, PatternRewriter &rewriter);

LLVM::LLVMFuncOp getLibdeviceFuncCall(RewriterBase &rewriter, Operation *op,
                                      StringRef funcName, Type retType,
                                      ValueRange ins = {});

} // namespace MUSA
} // namespace LLVM
} // namespace mlir

#endif
