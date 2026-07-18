// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Utility.h"
// LLVM22 compatibility: re-introduce dragon-style free macros (i32_val, etc.).
// Must be included AFTER the upstream Utility.h pulled in above.
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h"

namespace mlir::LLVM::XPU {

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};

  // TODO[dyq]: add Dimension:y & Dimension:z mapping
  Value blockId;
  switch (axis) {
  case 0: {
    blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
    break;
  }
  case 1:
  case 2: {
    blockId = i32_val(0);
    break;
  }
  default: {
    llvm_unreachable("ProgramIdOp Get Invalid Axis");
  }
  }

  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Type getFunctionType(mlir::OpBuilder &builder, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  mlir::MLIRContext *ctx = builder.getContext();
  auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  return LLVM::LLVMFunctionType::get(voidTy, operandTypes);
}

Value createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                       Operation *op, Type &elemTy, ValueRange &operands,
                       Location &loc) {
  Type funcType = mlir::triton::gpu::getFunctionType(elemTy, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult();
}

void createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                      Operation *op, ValueRange &operands, Location &loc) {
  OpBuilder builder(op);
  Type funcType = getFunctionType(builder, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  rewriter.create<LLVM::CallOp>(loc, funcOp, operands);
  return;
}

} // namespace mlir::LLVM::XPU
