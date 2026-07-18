// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
