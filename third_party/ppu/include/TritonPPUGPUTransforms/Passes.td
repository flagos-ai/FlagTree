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

#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"


def TritonPPUGPUAccelerateMatmul : Pass<"tritonppugpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "ppu accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., PPU tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonPPUAIULoweringPass : Pass<"tritonppu-aiu-lowering", "mlir::ModuleOp"> {
  let summary = "ppu aiu lowering";

  let description = [{
    Accelerate data movement from global memory to shared memory
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::ppu_gpu::TritonPPUGPUDialect",
                           "mlir::triton::TritonDialect"];
}


def TlePromoteAsyncLoadToAIUPass : Pass<"tle-promote-async-load-to-aiu", "mlir::ModuleOp"> {
  let summary = "Promote TLE async loads to AIU loads for PPU";

  let description = [{
    Finds tt.LoadOp with the tt.load.async attribute and tensor pointer operands,
    and replaces them with tt.AIULoadOp; only 1/2/4-byte element types are
    promoted. The AIU descriptor has no row stride: with a constant row stride
    and a constant row length, a layout it cannot express is not promoted and
    stays a plain load; when either is a run-time value, the layout is checked
    on the device where the copy is issued, and the threads copy a tile the
    descriptor cannot. Must run before rewrite_tensor_pointer.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonPPUTleLowerPipe : Pass<"tritonppu-tle-lower-pipe", "mlir::ModuleOp"> {
  let summary = "Lower TLE pipes to PPU hardware named barriers";

  let description = [{
    Lowers tle.pipe.* lifecycle ops to ttng named barrier ops that the PPU
    backend maps to hardware barriers: per stage, one full barrier for each
    reader group and one empty barrier (none for one_shot). Pipes that do not
    fit the hardware barriers left after the ids warp specialization and other
    TLE barriers take (the pipes in deeper loops come first) use counters in
    shared memory instead. Must run after the TLE barrier lowering and before
    the software pipeliner.

    Supported forms, rejected before rewriting:

    - A pipe is created and used in one kernel, after inlining. Lifecycle ops
      belong to the create with the same capacity, scope, name, field names
      and fields, where a field is its allocation plus the static offsets of
      the views taken of it: two pipes on one allocation differ by name or by
      the stages they ring through.
    - CTA-scoped pipes with one writer warp group and one reader per group,
      named or default; pipes that only order accesses inside one group lower
      to group barriers and need no hand-over.
    - A pipe handing over between warp groups, one_shot or not, is created,
      and its warp_specialize runs, once per kernel: the hardware barriers and
      the shared-memory counters keep their state across executions, so
      neither may sit inside a loop (a persistent kernel loops inside the
      partitions).
    - Capacity 1 and above; close, or one_shot without close, as on NVIDIA.
      A writer that runs its pipe twice without an intervening reader drains
      is a hang, not diagnosed. The phase of the lifecycle ops continues
      across the loops of a kernel, as on NVIDIA: the hardware barriers do not
      read it, the shared-memory counters do.
    - cp.async payloads published through pipe.writer.commit; a stage is
      published only after its copies completed.

    Performance, not correctness: a single mixed pipe (capacity 2-4, with a
    reader in the writer's group, no close or one_shot, structured control
    flow, no AIU, inline assembly or calls in the writer region) keeps its
    pending copy groups in registers and completes a group when its local
    reader reaches the slot. The FIFO schedule replaces that queue when the
    writer region is exactly a prefetch loop of min(n, depth) followed by a
    consume loop that waits, releases and refills under `position + depth < n`,
    in that order. Every other shape completes the copies at the commit. Each
    fallback emits a remark naming the reason; see
    MLIR_ENABLE_DIAGNOSTICS=remarks.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def TritonPPUTleLowerTmaCopy : Pass<"tritonppu-tle-lower-tma-copy", "mlir::ModuleOp"> {
  let summary = "Rewrite TLE tensor-descriptor copies for PPU";

  let description = [{
    Rewrites ttg.tma_copy (tle.gpu.copy with a tensor descriptor) to the
    pointer form of the transfer: a descriptor load or store paired with a
    store or load through tle.local_pointers over the whole shared buffer.
    PPU has no TMA unit; the descriptor accesses become bounds-checked pointer
    accesses in the descriptor-to-pointer rewrite, which must run afterwards.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonPPUTleZeroPadDescriptorCopies : Pass<"tritonppu-tle-zero-pad-descriptor-copies", "mlir::ModuleOp"> {
  let summary = "Zero-fill the pointer loads of TLE descriptor copies on PPU";

  let description = [{
    After the descriptor-to-pointer rewrite, the loads created by
    tritonppu-tle-lower-tma-copy fill out-of-bounds elements with
    select(padding, nan, 0). cp.async can only fill with zero, so the fill
    becomes the zero constant and a device assert checks that the descriptor
    does not pad with NaN.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
