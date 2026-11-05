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

#ifndef TRITON_PPU_BARRIER_UTILITY_H
#define TRITON_PPU_BARRIER_UTILITY_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir::triton::ppu {

// PPU hardware barriers. ppu-llc does not lower the NVVM partial-barrier
// intrinsics (llvm.nvvm.barrier / barrier.cta.*), only barrier0, so partial
// barriers are emitted as calls to the SDK builtins
//   __ppukernel_barrier_sync_cnt_void(id, numThreads, fence)
//   __ppukernel_barrier_arrive_void(id, numThreads, fence)
// which ppu-llc resolves to s.blksyn.
//
// `fence` selects what the issuing warp drains before the barrier:
//   bit 0: its shared-memory accesses, loads and stores (s.wait tsmcnt(0);
//          the tsm counter covers both);
//   bit 4: all of its outstanding global accesses (vmem.fence.blk).
// barrier0, __syncthreads and an inline `bar.sync` use 31 (both). Draining the
// global accesses costs ~3000 cycles with loads in flight, and is not needed to
// make shared memory visible to the other warps.
//
// Bit 0 also suffices after asynchronous copies into shared memory (cp.async,
// AIU loads). Their completion is what cp.async.wait_group (ttg.async_wait)
// establishes, not the tsm counter: the producer waits before it signals
// (Membar puts a barrier after every async_wait, the pipe lowering issues the
// wait itself), and the arrive with bit 0 then drains the warp's own
// shared-memory accesses, after which the copied data is visible to the
// waiting group. This rests on the SDK's cp.async model rather than on a
// documented counter interaction; the runtime tests
// test_tle_warp_specialize.py::test_async_load_in_worker_partition and
// test_tle_aiu_async_load.py cover it.
enum PPUBarrierFence : unsigned {
  kPPUBarrierFenceNone = 0,
  kPPUBarrierFenceShared = 1,
  kPPUBarrierFenceGlobal = 16,
  kPPUBarrierFenceAll = 31,
};

// Number of hardware barriers per CTA; id 0 is __syncthreads. The TTGIR passes
// that budget them repeat this value: TleLowerPipe.cpp (kNumHardwareBarriers)
// and TleAllocateNamedBarriers.cpp (kNumPhysicalNamedBarriers).
constexpr unsigned kPPUNumBarriers = 16;

void createPPUBarrierSync(OpBuilder &b, Location loc, Value id,
                          unsigned numThreads, unsigned fence);
void createPPUBarrierSync(OpBuilder &b, Location loc, unsigned id,
                          unsigned numThreads, unsigned fence);
void createPPUBarrierArrive(OpBuilder &b, Location loc, Value id,
                            unsigned numThreads, unsigned fence);
void createPPUBarrierArrive(OpBuilder &b, Location loc, unsigned id,
                            unsigned numThreads, unsigned fence);

// A CTA-wide barrier (id 0) that the lowering inserts to order shared memory:
// Membar hazards and the scratch buffers synchronized through
// TargetInfo::barrier (e.g. layout conversions). It drains only the warp's
// shared-memory accesses. Inside a warp-specialized kernel the warp
// specialization lowering retargets it to the barrier of the warp group.
void createPPUSharedMemoryBarrier(OpBuilder &b, Location loc,
                                  unsigned numThreads);

// True for a call emitted by createPPUSharedMemoryBarrier (barrier id 0). TLE
// allocates named barriers from id 1 up, so they do not use it.
bool isPPUCtaBarrierCall(LLVM::CallOp call);

} // namespace mlir::triton::ppu

#endif // TRITON_PPU_BARRIER_UTILITY_H
