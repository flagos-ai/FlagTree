// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --split-input-file --triton-tle-lower-pipe-to-nvws | FileCheck %s --check-prefix=NVWS
// RUN: triton-opt %s --split-input-file --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token | FileCheck %s --check-prefix=TTNG
// Profile scratch effects do not alias payload or barrier storage. Keep each
// explicit marker in place while retaining transport and participant proofs.

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_1_tma_copies
// NVWS: nvws.create_token
// NVWS-SAME: loadType = 2 : i32
// TTNG-LABEL: tt.func @trace_1_tma_copies
// TTNG: ttng.init_barrier {{.*}}, 1
// TTNG: tle.trace_event {{.*}}kind = 272 : i32, site = 1
// TTNG: ttng.barrier_expect {{.*}}, 8192
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273 : i32, site = 1
// TTNG: tle.trace_event {{.*}}kind = 272 : i32, site = 3
// TTNG: tle.trace_event {{.*}}kind = 273 : i32, site = 3
  tt.func @trace_1_tma_copies(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %q {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %q[%c0, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 1 : i32}
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 1 : i32}
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 3 : i32}
    tle.pipe.writer_commit %pipe, %q[%c0] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 3 : i32}
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_2_tma_copies
// NVWS: nvws.create_token
// NVWS-SAME: loadType = 2 : i32
// TTNG-LABEL: tt.func @trace_2_tma_copies
// TTNG: ttng.init_barrier {{.*}}, 1
// TTNG: tle.trace_event {{.*}}kind = 272 : i32, site = 1
// TTNG: ttng.barrier_expect {{.*}}, 16384
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273 : i32, site = 1
// TTNG: tle.trace_event {{.*}}kind = 272 : i32, site = 2
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273 : i32, site = 2
// TTNG: tle.trace_event {{.*}}kind = 272 : i32, site = 3
// TTNG: tle.trace_event {{.*}}kind = 273 : i32, site = 3
  tt.func @trace_2_tma_copies(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "tma", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %q, %g[%c0, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 1 : i32}
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 1 : i32}
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 2 : i32}
    ttg.tma_copy %desc, %g_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 2 : i32}
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 3 : i32}
    tle.pipe.writer_commit %pipe, %q, %g[%c0] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 3 : i32}
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_local_store_participants
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 64 : i32
// TTNG-LABEL: tt.func @trace_local_store_participants
// TTNG: ttng.init_barrier {{.*}}, 64
// TTNG: tt.store
// TTNG: tle.trace_event {{.*}}site = 11
// TTNG: tt.store
// TTNG: tle.trace_event {{.*}}site = 12
// TTNG: ttng.arrive_barrier {{.*}}, 64
// TTNG: participant_arrive = true
  tt.func @trace_local_store_participants(%a: !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>, %v0: tensor<64xi8, #blocked1>, %v1: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %row1 = arith.constant dense<1> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>

    %pipe_identity_0 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_0, %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable> -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %ptr1 = "tle.local_pointers"(%slot, %row1, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 11 : i32}
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 12 : i32}
    tle.pipe.writer_commit %pipe_identity_0, %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_mixed_local_tma
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 65 : i32
// TTNG-LABEL: tt.func @trace_mixed_local_tma
// TTNG: ttng.init_barrier {{.*}}, 65
// TTNG: tle.trace_event {{.*}}kind = 272
// TTNG: ttng.barrier_expect {{.*}}, 8192
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273
// TTNG: tle.trace_event {{.*}}site = 22
// TTNG: ttng.arrive_barrier {{.*}}, 64
  tt.func @trace_mixed_local_tma(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>, %v0: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>

    %pipe_identity_2 = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_2, %q, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable> -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_ptr = "tle.local_pointers"(%g_slot, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    tt.store %g_ptr, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 22 : i32}
    tle.pipe.writer_commit %pipe_identity_2, %q, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_mixed_completed_async_tma
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 65 : i32
// TTNG-LABEL: tt.func @trace_mixed_completed_async_tma
// TTNG: ttng.init_barrier {{.*}}, 65
// TTNG: tle.trace_event {{.*}}kind = 272
// TTNG: ttng.barrier_expect {{.*}}, 8192
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273
// TTNG: tle.trace_event {{.*}}site = 22
// TTNG: ttng.arrive_barrier {{.*}}, 64
  tt.func @trace_mixed_completed_async_tma(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    %pipe_identity_3 = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_3, %q, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    %copy_group = ttg.async_commit_group tokens %copy
    ttg.async_wait %copy_group {num = 0 : i32}
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 22 : i32}
    tle.pipe.writer_commit %pipe_identity_3, %q, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_mixed_cp_async_tma
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 129 : i32
// TTNG-LABEL: tt.func @trace_mixed_cp_async_tma
// TTNG: ttng.init_barrier {{.*}}, 129
// TTNG: tle.trace_event {{.*}}kind = 272
// TTNG: ttng.barrier_expect {{.*}}, 8192
// TTNG: ttng.async_tma_copy_global_to_local
// TTNG: tle.trace_event {{.*}}kind = 273
// TTNG: tle.trace_event {{.*}}site = 22
// TTNG: ttng.async_copy_mbarrier_arrive
  tt.func @trace_mixed_cp_async_tma(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    %pipe_identity_5 = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_5, %q, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 21 : i32}
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 21 : i32}
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 22 : i32}
    tle.pipe.writer_commit %pipe_identity_5, %q, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_cp_async", field_names = ["q", "g"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_completed_async_participants
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 64 : i32
// TTNG-LABEL: tt.func @trace_completed_async_participants
// TTNG: ttng.init_barrier {{.*}}, 64
// TTNG: ttg.async_copy_global_to_local
// TTNG: tle.trace_event {{.*}}site = 31
// TTNG: ttg.async_wait {{.*}}num = 0
// TTNG: tle.trace_event {{.*}}site = 32
// TTNG: ttng.arrive_barrier {{.*}}, 64
  tt.func @trace_completed_async_participants(%g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    %pipe_identity_3 = tle.pipe.create %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_3, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 273 : i32, site = 31 : i32}
    %copy_group = ttg.async_commit_group tokens %copy
    ttg.async_wait %copy_group {num = 0 : i32}
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 32 : i32}
    tle.pipe.writer_commit %pipe_identity_3, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}

// -----


#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_before_local_payload
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 64 : i32
// TTNG-LABEL: tt.func @trace_before_local_payload
// TTNG: ttng.init_barrier {{.*}}, 64
// TTNG: tt.store
// TTNG: tt.store
// TTNG: ttng.arrive_barrier {{.*}}, 64
// TTNG: participant_arrive = true
  tt.func @trace_before_local_payload(%a: !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>, %v0: tensor<64xi8, #blocked1>, %v1: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %row1 = arith.constant dense<1> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>

    %pipe_identity_0 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_0, %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable> -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %ptr1 = "tle.local_pointers"(%slot, %row1, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 41 : i32}
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    tle.pipe.writer_commit %pipe_identity_0, %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tt.return
  }
}


// -----


#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// NVWS-LABEL: tt.func @trace_before_async_payload
// NVWS: nvws.create_token
// NVWS-SAME: full_count = 64 : i32
// TTNG-LABEL: tt.func @trace_before_async_payload
// TTNG: ttng.init_barrier {{.*}}, 64
// TTNG: ttg.async_copy_global_to_local
// TTNG: ttg.async_wait {{.*}}num = 0
// TTNG: ttng.arrive_barrier {{.*}}, 64
  tt.func @trace_before_async_payload(%g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    %pipe_identity_3 = tle.pipe.create %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_3, %g[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 41 : i32}
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    %copy_group = ttg.async_commit_group tokens %copy
    ttg.async_wait %copy_group {num = 0 : i32}
    %clock = tt.elementwise_inline_asm "mov.u64 $0, %clock64;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    tle.pipe.writer_commit %pipe_identity_3, %g[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["g"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}
