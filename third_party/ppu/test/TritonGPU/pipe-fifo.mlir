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

// RUN: triton-opt %s -split-input-file --tritonppu-tle-lower-pipe | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonppu-tle-lower-pipe -verify-diagnostics

// The FIFO schedule emits no remark; the pipes that keep the stage queue
// say why (checked by the second run).
// CHECK-LABEL: tt.func @signed_fifo
// CHECK: arith.constant 2147483645 : i32
// CHECK: arith.cmpi sle
// CHECK: scf.if
// CHECK: } else {
// CHECK: iter_args
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#stage = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @signed_fifo(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %phase = arith.constant false
    %ring = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    tle.pipe.create %ring {capacity = 3 : i32, field_names = ["data"], readers = ["local", "remote"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%ring, %n)
    default {
      %end = arith.minsi %n, %c2 : i32

      scf.for %i = %c0 to %end step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        tle.pipe.writer_acquire %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %tile = ttg.memdesc_index %ring[%s] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
        %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
        tle.pipe.writer_commit %ring[%s] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      scf.for %i = %c0 to %n step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        %closed = tle.pipe.reader_wait %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %ring[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %next = arith.addi %i, %c2 : i32
        %refill = arith.cmpi slt, %next, %n : i32
        scf.if %refill {
          %ns = arith.remsi %next, %c3 : i32
          tle.pipe.writer_acquire %ring[%ns, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
          %tile = ttg.memdesc_index %ring[%ns] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
          %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
          tle.pipe.writer_commit %ring[%ns] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        }
      }
      ttg.warp_yield
    }
    partition0(%remote: !ttg.memdesc<3x128xf32, #shared, #smem, mutable>, %count: i32) num_warps(4) {
      %r0 = arith.constant 0 : i32
      %r1 = arith.constant 1 : i32
      %r3 = arith.constant 3 : i32
      %rphase = arith.constant false
      scf.for %i = %r0 to %count step %r1 : i32 {
        %s = arith.remsi %i, %r3 : i32
        %closed = tle.pipe.reader_wait %remote[%s, %rphase] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %remote[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<3x128xf32, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func @unsigned_fifo
// CHECK-NOT: 2147483645
// CHECK: scf.for unsigned
// CHECK: iter_args
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#stage = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @unsigned_fifo(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %phase = arith.constant false
    %ring = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    // expected-remark @below {{pipe '<unnamed>' tracks its pending stages in a queue: the prefetch loop must run min(n, depth) signed iterations}}
    tle.pipe.create %ring {capacity = 3 : i32, field_names = ["data"], readers = ["local", "remote"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%ring, %n)
    default {
      %end = arith.minsi %n, %c2 : i32

      scf.for unsigned %i = %c0 to %end step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        tle.pipe.writer_acquire %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %tile = ttg.memdesc_index %ring[%s] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
        %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
        tle.pipe.writer_commit %ring[%s] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      scf.for unsigned %i = %c0 to %n step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        %closed = tle.pipe.reader_wait %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %ring[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %next = arith.addi %i, %c2 : i32
        %refill = arith.cmpi slt, %next, %n : i32
        scf.if %refill {
          %ns = arith.remsi %next, %c3 : i32
          tle.pipe.writer_acquire %ring[%ns, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
          %tile = ttg.memdesc_index %ring[%ns] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
          %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
          tle.pipe.writer_commit %ring[%ns] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        }
      }
      ttg.warp_yield
    }
    partition0(%remote: !ttg.memdesc<3x128xf32, #shared, #smem, mutable>, %count: i32) num_warps(4) {
      %r0 = arith.constant 0 : i32
      %r1 = arith.constant 1 : i32
      %r3 = arith.constant 3 : i32
      %rphase = arith.constant false
      scf.for %i = %r0 to %count step %r1 : i32 {
        %s = arith.remsi %i, %r3 : i32
        %closed = tle.pipe.reader_wait %remote[%s, %rphase] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %remote[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<3x128xf32, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func @copy_outside_commit
// CHECK-NOT: 2147483645
// CHECK: scf.for
// CHECK: iter_args
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#stage = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @copy_outside_commit(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %phase = arith.constant false
    %ring = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    // expected-remark @below {{pipe '<unnamed>' tracks its pending stages in a queue: the writer region contains ttg.async_copy_global_to_local outside the schedule's acquire..commit ranges}}
    tle.pipe.create %ring {capacity = 3 : i32, field_names = ["data"], readers = ["local", "remote"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%ring, %n)
    default {
      %end = arith.minsi %n, %c2 : i32
      %tile0 = ttg.memdesc_index %ring[%c0] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
      %untracked = ttg.async_copy_global_to_local %ptrs, %tile0 : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
      scf.for %i = %c0 to %end step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        tle.pipe.writer_acquire %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %tile = ttg.memdesc_index %ring[%s] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
        %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
        tle.pipe.writer_commit %ring[%s] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      scf.for %i = %c0 to %n step %c1 : i32 {
        %s = arith.remsi %i, %c3 : i32
        %closed = tle.pipe.reader_wait %ring[%s, %phase] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %ring[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        %next = arith.addi %i, %c2 : i32
        %refill = arith.cmpi slt, %next, %n : i32
        scf.if %refill {
          %ns = arith.remsi %next, %c3 : i32
          tle.pipe.writer_acquire %ring[%ns, %phase] {capacity = 3 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
          %tile = ttg.memdesc_index %ring[%ns] : !ttg.memdesc<3x128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #stage, #smem, mutable>
          %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #stage, #smem, mutable>
          tle.pipe.writer_commit %ring[%ns] {capacity = 3 : i32, field_names = ["data"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        }
      }
      ttg.warp_yield
    }
    partition0(%remote: !ttg.memdesc<3x128xf32, #shared, #smem, mutable>, %count: i32) num_warps(4) {
      %r0 = arith.constant 0 : i32
      %r1 = arith.constant 1 : i32
      %r3 = arith.constant 3 : i32
      %rphase = arith.constant false
      scf.for %i = %r0 to %count step %r1 : i32 {
        %s = arith.remsi %i, %r3 : i32
        %closed = tle.pipe.reader_wait %remote[%s, %rphase] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %remote[%s] {capacity = 3 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<3x128xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<3x128xf32, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}
