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
// groups. The kernel orders data it hands over through global memory itself
// (documented on tle.gpu.barrier_wait/arrive); the lowering does not widen the
// fence from the ops preceding a barrier, since a scan of one block misses the
// stores of earlier blocks and loop iterations, and widening every arrive
// after an unrelated output store would pay the drain for nothing.
// Asynchronous copies are covered by the explicit async_wait the producer
// issues before signalling (see BarrierUtility.h).
//
// The SDK builtin takes the thread count as a constant. TLE emits it as one
// (arrive_count on the frontend, the group size in the pipe lowering), so a
// non-constant count is reported instead of silently falling through to the
// generic legalization failure.
std::optional<unsigned> getConstantNumThreads(Value numThreads) {
  if (auto cst = numThreads.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

LogicalResult emitNonConstantThreadCount(Operation *op) {
  return op->emitError("PPU named barriers need a constant thread count");
}

struct NamedBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierArriveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto numThreads = getConstantNumThreads(adaptor.getNumThreads());
    if (!numThreads)
      return emitNonConstantThreadCount(op);
    ppu::createPPUBarrierArrive(rewriter, op.getLoc(), adaptor.getBar(),
                                *numThreads, ppu::kPPUBarrierFenceShared);
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
    auto numThreads = getConstantNumThreads(adaptor.getNumThreads());
    if (!numThreads)
      return emitNonConstantThreadCount(op);
    ppu::createPPUBarrierSync(rewriter, op.getLoc(), adaptor.getBar(),
                              *numThreads, ppu::kPPUBarrierFenceShared);
    rewriter.eraseOp(op);
    return success();
  }
};
#endif

// PPU has no mbarrier. The TLE frontend lowers its phased barrier forms
// (barrier_wait/arrive with phaseIdx, alloc_barrier with init=READY or
// expect_bytes, copy with a completion barrier) and gluon its mbarrier
// builtins to these ops. The PPU compiler rejects them on the frontend IR,
// naming the API; this is the backstop for IR that reaches the conversion
// another way, so that it fails here with the reason instead of in ppu-llc.
template <typename OpT>
struct MbarrierOpRejection : public ConvertOpToLLVMPattern<OpT> {
  using ConvertOpToLLVMPattern<OpT>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename ConvertOpToLLVMPattern<OpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError("the PPU backend has no mbarrier; use named barriers "
                        "(tle.gpu.barrier_wait/barrier_arrive without "
                        "phaseIdx)");
  }
};

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
  patterns
      .add<MbarrierOpRejection<triton::nvidia_gpu::InitBarrierOp>,
           MbarrierOpRejection<triton::nvidia_gpu::InvalBarrierOp>,
           MbarrierOpRejection<triton::nvidia_gpu::BarrierExpectOp>,
           MbarrierOpRejection<triton::nvidia_gpu::WaitBarrierOp>,
           MbarrierOpRejection<triton::nvidia_gpu::ArriveBarrierOp>,
           MbarrierOpRejection<triton::nvidia_gpu::AsyncCopyMbarrierArriveOp>>(
          typeConverter, benefit);
#ifdef __TLE__
  patterns.add<NamedBarrierArriveOpConversion, NamedBarrierWaitOpConversion>(
      typeConverter, benefit);
#endif
}
