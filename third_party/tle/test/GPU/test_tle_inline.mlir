// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file -inline | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func public @inline_local_pointers
  tt.func public @inline_local_pointers(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    // CHECK-NOT: tt.call
    // CHECK: %[[PLAIN_PTR:.*]] = "tle.local_pointers"
    // CHECK: tt.atomic_rmw add, relaxed, cta, %[[PLAIN_PTR]]
    tt.call @plain_local_pointer_worker(%smem) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    // CHECK-NEXT: tt.return
    tt.return
  }

  // CHECK-NOT: @plain_local_pointer_worker
  tt.func private @plain_local_pointer_worker(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>)
      attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %ptr = "tle.local_pointers"(%smem, %c0) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %old = tt.atomic_rmw add, relaxed, cta, %ptr, %c1, %true : (!tt.ptr<i32, 3>, i32, i1) -> i32
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func public @inline_local_pointers_in_warp_specialize
  tt.func public @inline_local_pointers_in_warp_specialize(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    ttg.warp_specialize(%smem)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0
    partition0(%arg0: !ttg.memdesc<16xi32, #shared, #smem, mutable>) num_warps(4) {
      // CHECK-NOT: tt.call
      // CHECK: %[[PTR:.*]] = "tle.local_pointers"
      // CHECK: tt.atomic_rmw add, relaxed, cta, %[[PTR]]
      tt.call @ws_local_pointer_worker(%arg0) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
      // CHECK-NEXT: ttg.warp_return
      ttg.warp_return
    } : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-NOT: @ws_local_pointer_worker
  tt.func private @ws_local_pointer_worker(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>)
      attributes {noinline = false, "ttg.num-warps" = 4 : i32} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %ptr = "tle.local_pointers"(%smem, %c0) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %old = tt.atomic_rmw add, relaxed, cta, %ptr, %c1, %true : (!tt.ptr<i32, 3>, i32, i1) -> i32
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func public @preserve_noinline_local_pointers
  tt.func public @preserve_noinline_local_pointers(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    // CHECK: tt.call @noinline_local_pointer_worker
    tt.call @noinline_local_pointer_worker(%smem) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK: tt.func private @noinline_local_pointer_worker
  // CHECK: "tle.local_pointers"
  // CHECK: tt.atomic_rmw add, relaxed, cta
  tt.func private @noinline_local_pointer_worker(
      %smem: !ttg.memdesc<16xi32, #shared, #smem, mutable>)
      attributes {noinline = true} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %ptr = "tle.local_pointers"(%smem, %c0) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %old = tt.atomic_rmw add, relaxed, cta, %ptr, %c1, %true : (!tt.ptr<i32, 3>, i32, i1) -> i32
    tt.return
  }
}

// -----

module {
  llvm.func @_sink(!llvm.ptr)

  // CHECK-LABEL: tt.func public @inline_dsl_region
  tt.func public @inline_dsl_region(%arg0: !tt.ptr<i32>) {
    // CHECK-NOT: tt.call
    // CHECK: %[[DSL_RESULT:.*]] = "tle.dsl_region"
    // CHECK: tle.yield
    // CHECK: llvm.call @_sink(%[[DSL_RESULT]])
    tt.call @dsl_region_worker(%arg0) : (!tt.ptr<i32>) -> ()
    // CHECK-NEXT: tt.return
    tt.return
  }

  // CHECK-NOT: @dsl_region_worker
  tt.func private @dsl_region_worker(%arg0: !tt.ptr<i32>)
      attributes {noinline = false} {
    %ptr = "tle.dsl_region"(%arg0) ({
    ^bb0(%input: !tt.ptr<i32>):
      %raw = "tle.extract_ptr"(%input) : (!tt.ptr<i32>) -> !llvm.ptr
      "tle.yield"(%raw) : (!llvm.ptr) -> ()
    }) {arg_dialect = "llvm", output_operand_indices = array<i32: 0>,
        region_dialect = "llvm", tle_raw.source_id = "inline-test"}
        : (!tt.ptr<i32>) -> !llvm.ptr
    llvm.call @_sink(%ptr) : (!llvm.ptr) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func public @inline_pipe_lifecycle_ops
  tt.func public @inline_pipe_lifecycle_ops(
      %field: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // CHECK-NOT: tt.call
    // CHECK: tle.pipe.create
    // CHECK: tle.pipe.writer_acquire
    // CHECK: tle.pipe.writer_commit
    tt.call @pipe_lifecycle_worker(%field) : (!ttg.memdesc<4x16xf16, #shared, #smem, mutable>) -> ()
    // CHECK-NEXT: tt.return
    tt.return
  }

  // CHECK-NOT: @pipe_lifecycle_worker
  tt.func private @pipe_lifecycle_worker(
      %field: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>)
      attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %field {capacity = 4 : i32, pipe_name = "field", field_names = ["field"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %field[%c0, %false] {capacity = 4 : i32, pipe_name = "field", field_names = ["field"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %field[%c0] {capacity = 4 : i32, pipe_name = "field", field_names = ["field"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: tt.func public @inline_other_tle_ops
  tt.func public @inline_other_tle_ops() {
    // CHECK-NOT: tt.call
    // CHECK: tle.tma_store.commit_group
    tt.call @tma_store_commit() : () -> ()
    // CHECK-NEXT: tt.return
    tt.return
  }

  // CHECK-NOT: @tma_store_commit
  tt.func private @tma_store_commit() attributes {noinline = false} {
    tle.tma_store.commit_group
    tt.return
  }
}
