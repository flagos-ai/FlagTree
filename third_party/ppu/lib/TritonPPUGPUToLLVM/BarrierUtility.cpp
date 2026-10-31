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

#include "BarrierUtility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

LLVM::LLVMFuncOp getBarrierBuiltin(OpBuilder &b, Location loc, StringRef name) {
  auto module =
      b.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;
  MLIRContext *ctx = b.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                          {i32, i32, i32});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto func = LLVM::LLVMFuncOp::create(b, loc, name, type);
  // Calls stay in place relative to memory operations.
  func.setPassthroughAttr(
      ArrayAttr::get(ctx, {StringAttr::get(ctx, "convergent"),
                           StringAttr::get(ctx, "nounwind")}));
  return func;
}

void createBarrierCall(OpBuilder &b, Location loc, StringRef name, Value id,
                       unsigned numThreads, unsigned fence) {
  Type i32 = b.getI32Type();
  auto func = getBarrierBuiltin(b, loc, name);
  Value args[] = {id, LLVM::ConstantOp::create(b, loc, i32, numThreads),
                  LLVM::ConstantOp::create(b, loc, i32, fence)};
  LLVM::CallOp::create(b, loc, func, args);
}

Value constId(OpBuilder &b, Location loc, unsigned id) {
  return LLVM::ConstantOp::create(b, loc, b.getI32Type(), id);
}

} // namespace

namespace mlir::triton::ppu {

void createPPUBarrierSync(OpBuilder &b, Location loc, Value id,
                          unsigned numThreads, unsigned fence) {
  createBarrierCall(b, loc, "__ppukernel_barrier_sync_cnt_void", id, numThreads,
                    fence);
}

void createPPUBarrierSync(OpBuilder &b, Location loc, unsigned id,
                          unsigned numThreads, unsigned fence) {
  createPPUBarrierSync(b, loc, constId(b, loc, id), numThreads, fence);
}

void createPPUBarrierArrive(OpBuilder &b, Location loc, Value id,
                            unsigned numThreads, unsigned fence) {
  createBarrierCall(b, loc, "__ppukernel_barrier_arrive_void", id, numThreads,
                    fence);
}

void createPPUBarrierArrive(OpBuilder &b, Location loc, unsigned id,
                            unsigned numThreads, unsigned fence) {
  createPPUBarrierArrive(b, loc, constId(b, loc, id), numThreads, fence);
}

} // namespace mlir::triton::ppu
