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
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

#ifdef __TLE__
// TLE named barriers (tle.gpu.barrier_wait/arrive without a phase) lower to PPU
// hardware barriers. They order shared memory between warp groups, so the
// issuing warp drains only its shared-memory accesses; its outstanding global
// accesses keep flowing (draining them costs ~3000 cycles per barrier), so
// unlike bar.sync on NVIDIA they do not order global memory between the
// groups. Asynchronous copies are covered by the explicit async_wait the
// producer issues before signalling.
unsigned getNumThreads(Value numThreads) {
  if (auto cst = numThreads.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return 0;
}

struct NamedBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierArriveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numThreads = getNumThreads(adaptor.getNumThreads());
    if (!numThreads)
      return rewriter.notifyMatchFailure(
          op, "PPU named barriers need a constant thread count");
    ppu::createPPUBarrierArrive(rewriter, op.getLoc(), adaptor.getBar(),
                                numThreads, ppu::kPPUBarrierFenceShared);
    rewriter.eraseOp(op);
    return success();
  }
};

struct NamedBarrierWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numThreads = getNumThreads(adaptor.getNumThreads());
    if (!numThreads)
      return rewriter.notifyMatchFailure(
          op, "PPU named barriers need a constant thread count");
    ppu::createPPUBarrierSync(rewriter, op.getLoc(), adaptor.getBar(),
                              numThreads, ppu::kPPUBarrierFenceShared);
    rewriter.eraseOp(op);
    return success();
  }
};
#endif

// The barrier Membar inserts between conflicting shared-memory accesses (also
// after every async_wait). It only has to order shared memory.
struct LocalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numThreads = triton::gpu::lookupNumWarps(op) *
                          triton::gpu::lookupThreadsPerWarp(rewriter);
    ppu::createPPUSharedMemoryBarrier(rewriter, op.getLoc(), numThreads);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::ppu::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<LocalBarrierOpConversion>(typeConverter,
                                         benefit.getBenefit() + 1);
#ifdef __TLE__
  patterns.add<NamedBarrierArriveOpConversion, NamedBarrierWaitOpConversion>(
      typeConverter, benefit);
#endif
}
