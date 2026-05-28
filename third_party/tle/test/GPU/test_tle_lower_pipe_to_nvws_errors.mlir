// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws -split-input-file -verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_missing_create(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // expected-error @+1 {{requires a preceding matching pipe.create}}
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_one_shot_close(%a: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{does not support close on one_shot pipe}}
    tle.pipe.writer_close %a[%c0, %false] {capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_duplicate_create(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{duplicates an existing pipe.create}}
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multiple_reader_tasks(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{requires exactly one async_task_id}}
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1, 2>, capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_named_reader_on_default_pipe(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses named reader left but pipe was created without readers}}
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_missing_reader_name_on_explicit_pipe(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], readers = ["left"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{requires reader_name because pipe was created with explicit readers}}
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_undeclared_reader_name(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], readers = ["left"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses undeclared pipe reader right}}
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_same_reader_name_different_task(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], readers = ["left"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed0 = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{but that reader already has async_task_id 1}}
    %closed1 = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_mixed_async_copy_without_wait(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %gptr: tensor<64x!tt.ptr<i8>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %q, %g[%c0, %false] {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<64xi8, #shared1, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %gptr, %g_slot : tensor<64x!tt.ptr<i8>, #blocked1> -> <64xi8, #shared1, #smem, mutable>
    ttg.async_commit_group tokens %copy
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{encountered an async copy for a non-TMA field without a proven async_wait before the pipe commit}}
    tle.pipe.writer_commit %q, %g[%c0] {capacity = 2 : i32, pipe_name = "mixed_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_partial_tma_pipe_commit(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab_tma", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a, %b[%c0, %false] {capacity = 2 : i32, pipe_name = "ab_tma", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{mixed TMA/local-store pipe commit requires proven local-store writes for the non-TMA fields}}
    tle.pipe.writer_commit %a, %b[%c0] {capacity = 2 : i32, pipe_name = "ab_tma", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_mixed_pipe_commit_transports(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_mixed", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_mixed", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a_mixed", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a_mixed", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c1] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{mixes local-store and TMA copy payload commits on the same pipe}}
    tle.pipe.writer_commit %a[%c1] {capacity = 2 : i32, pipe_name = "a_mixed", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_tma_local_store_same_root(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %v: tensor<64xf32, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "same_root", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "same_root", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %ptr = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<f32, 3>, #blocked1>
    tt.store %ptr, %v : tensor<64x!tt.ptr<f32, 3>, #blocked1>
    // expected-error @+1 {{has a ttg.tma_copy and a local-store payload targeting the same memdesc root}}
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "same_root", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_opaque_shared_store_between_tma_and_commit(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %opaque: tensor<64x!tt.ptr<i8, 3>, #blocked1>, %v: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.store %opaque, %v : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    // expected-error @+1 {{has an opaque shared-memory store between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "opaque", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_erased_shared_pointer_as_global_interleave(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %v: tensor<64xf32, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %shared_ptr = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<f32, 3>, #blocked1>
    %erased_ptr = builtin.unrealized_conversion_cast %shared_ptr : tensor<64x!tt.ptr<f32, 3>, #blocked1> to tensor<64x!tt.ptr<f32>, #blocked1>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.store %erased_ptr, %v : tensor<64x!tt.ptr<f32>, #blocked1>
    // expected-error @+1 {{has unsupported interleaved op tt.store between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "erased_as", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_unrelated_tma_between_payload_and_commit(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %h: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %h_slot = ttg.memdesc_index %h[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %h_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{has an unrelated ttg.tma_copy between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "unrelated_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_mixed_tma_cp_async_missing_copy(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "missing_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %q, %g[%c0, %false] {capacity = 2 : i32, pipe_name = "missing_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{mixed TMA/cp.async pipe commit requires proven cp.async copies for the non-TMA fields: missing a cp.async copy for at least one non-TMA field}}
    tle.pipe.writer_commit %q, %g[%c0] {capacity = 2 : i32, pipe_name = "missing_cp_async", field_names = ["q", "g"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_mixed_tma_cp_async_local_store_payload(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %q: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, %g: !ttg.memdesc<2x2x64xi8, #shared2, #smem, mutable>, %v: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    tle.pipe.create %q, %g {capacity = 2 : i32, pipe_name = "local_not_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %q, %g[%c0, %false] {capacity = 2 : i32, pipe_name = "local_not_cp_async", field_names = ["q", "g"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared2, #smem, mutable>
    %q_slot = ttg.memdesc_index %q[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %q_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %g_slot = ttg.memdesc_index %g[%c0] : !ttg.memdesc<2x2x64xi8, #shared2, #smem, mutable> -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %g_ptr = "tle.local_pointers"(%g_slot, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %g_ptr, %v : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    // expected-error @+1 {{mixed TMA/cp.async pipe commit requires proven cp.async copies for the non-TMA fields: encountered a non-TMA field local store that is not covered by cp.async mbarrier tracking}}
    tle.pipe.writer_commit %q, %g[%c0] {capacity = 2 : i32, pipe_name = "local_not_cp_async", field_names = ["q", "g"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}
