// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --split-input-file --triton-tle-lower-pipe-to-nvws --verify-diagnostics
// Dedicated trace effects cannot excuse unrelated payload, missing completion,
// unknown effects, opaque/shared aliases, or operations with regions.


#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @trace_reject_mixed_async_copy_without_wait(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe_identity_13 = tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_13, %q, %g[%c0, %false] {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    ttg.async_commit_group tokens %copy
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{encountered an async copy for a non-TMA field without a proven async_wait before the pipe commit}}
    tle.pipe.writer_commit %pipe_identity_13, %q, %g[%c0] {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}


// -----


#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @trace_reject_opaque_shared_store_between_tma_and_commit(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %opaque: tensor<64x!tt.ptr<i8, 3>, #blocked1>, %v: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe_identity_17 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_17, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.store %opaque, %v : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    // expected-error @+1 {{has an opaque shared-memory store between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %pipe_identity_17, %a[%c0] {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}


// -----


#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @trace_reject_erased_shared_pointer_as_global_interleave(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %v: tensor<64xf32, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %pipe_identity_18 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_18, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %shared_ptr = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<f32, 3>, #blocked1>
    %erased_ptr = builtin.unrealized_conversion_cast %shared_ptr : tensor<64x!tt.ptr<f32, 3>, #blocked1> to tensor<64x!tt.ptr<f32>, #blocked1>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.store %erased_ptr, %v : tensor<64x!tt.ptr<f32>, #blocked1>
    // expected-error @+1 {{has unsupported interleaved op tt.store between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %pipe_identity_18, %a[%c0] {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}


// -----


#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @trace_reject_unrelated_tma_between_payload_and_commit(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %h: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe_identity_19 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe_identity_19, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %h_slot = ttg.memdesc_index %h[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %h_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{has an unrelated ttg.tma_copy between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %pipe_identity_19, %a[%c0] {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
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
  tt.func @trace_reject_unknown_effect(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %q {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %q[%c0, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %effect = tt.elementwise_inline_asm "bar.sync 0;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    // expected-error @+1 {{has unsupported interleaved op tt.elementwise_inline_asm between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %pipe, %q[%c0] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
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
  tt.func @trace_reject_region(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %q {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %q[%c0, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 51 : i32}
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    scf.if %false {
      tle.trace_event %c0, %c0, %c0, %c0 {kind = 272 : i32, site = 52 : i32}
    }
    // expected-error @+1 {{has unsupported interleaved op scf.if between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %pipe, %q[%c0] {capacity = 2 : i32, pipe_name = "tma", field_names = ["q"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}
