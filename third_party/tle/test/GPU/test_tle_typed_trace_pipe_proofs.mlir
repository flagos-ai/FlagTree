// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token | FileCheck %s
// Adapted from the actual mixed-copy participant-commit regression. Trace
// metadata must preserve the original identity and async completion proof.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tt.func @traced_tma_and_completed_cpasync
// CHECK: tle.trace_event {{.*}}kind = 256
// CHECK: ttng.wait_barrier
// CHECK: tle.trace_event {{.*}}kind = 257
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_wait {{.*}}num = 0
// CHECK: ttng.barrier_expect {{.*}}, 8192
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.arrive_barrier
// CHECK-NOT: tle.pipe_trace_context
  tt.func @traced_tma_and_completed_cpasync(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    %pipe_identity_3 = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %trace = tle.pipe_trace_context %pipe_identity_3, %c0, %c0 {site = 11 : i32}
    tle.pipe.writer_acquire %trace, %q, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    %copy_group = ttg.async_commit_group tokens %copy
    ttg.async_wait %copy_group {num = 0 : i32}
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %pipe_identity_3, %q, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
// CHECK-LABEL: tt.func @traced_one_shot
// CHECK-NOT: ttng.wait_barrier
// CHECK: tle.trace_event {{.*}}kind = 263
// CHECK: ttng.arrive_barrier
// CHECK: tle.trace_event {{.*}}kind = 258
// CHECK: ttng.wait_barrier
// CHECK: tle.trace_event {{.*}}kind = 259
// CHECK: tle.trace_event {{.*}}kind = 263
// CHECK-NOT: tle.pipe_trace_context
  tt.func @traced_one_shot(%layer: i32, %iter: i32) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create {capacity = 1 : i32, pipe_name = "once", field_names = [], scope = "cta", one_shot = true}
    %writer = tle.pipe_trace_context %pipe, %layer, %iter {site = 21 : i32}
    tle.pipe.writer_acquire %writer, [%c0, %false] {capacity = 1 : i32, pipe_name = "once", field_names = [], scope = "cta"}
    tle.pipe.writer_commit %pipe, [%c0] {capacity = 1 : i32, pipe_name = "once", field_names = [], scope = "cta"}
    %reader = tle.pipe_trace_context %pipe, %layer, %iter {site = 22 : i32}
    %closed = tle.pipe.reader_wait %reader, [%c0, %false] {capacity = 1 : i32, pipe_name = "once", field_names = [], scope = "cta"}
    %release = tle.pipe_trace_context %pipe, %layer, %iter {site = 23 : i32}
    tle.pipe.reader_release %release, [%c0] {capacity = 1 : i32, pipe_name = "once", field_names = [], scope = "cta"}
    tt.return
  }
}
