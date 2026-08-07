/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_UTILITY_H

#include "third_party/ppu/include/TritonPPUGPUToLLVM/TIXAsmFormat.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "third_party/ppu/include/Dialect/PPUGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators

namespace mlir::triton::gpu {
class MemDescType;
}

namespace mlir {
namespace LLVM {

namespace PPU {
class TargetInfo;

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);
Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, RewriterBase &rewriter);
Value createElectPredicateWarp0(Location loc, RewriterBase &rewriter);

// Create bar.warp.sync
void createSyncWarp(Location loc, OpBuilder &builder);

// Lower ldmatrix and stmatrix
LogicalResult lowerLdStMatrix(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const mlir::triton::ppu::TargetInfo &targetInfo);

// Return true if the src and dst layout match.
bool matchMmaV1AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);

Value toUniformB32(Location loc, PatternRewriter &rewriter, Value val);

Value toUniformB64(Location loc, PatternRewriter &rewriter, Value val);

} // namespace PPU
} // namespace LLVM

} // namespace mlir

#endif
