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

// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_non_power_of_two_pipe_capacity
  tt.func @lower_non_power_of_two_pipe_capacity(%a: !ttg.memdesc<6x16xf16, #shared, #smem, mutable>) {
    // CHECK: %[[TAGS:.*]] = ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<6x1xi32
    // CHECK-NOT: tensor<6x1xi32
    // CHECK-COUNT-6: ttg.local_store
    // CHECK: nvws.create_token
    // CHECK-SAME: numBuffers = 6
    tle.pipe.create %a {capacity = 6 : i32, pipe_name = "six_stage", field_names = ["a"], scope = "cta"} : !ttg.memdesc<6x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_pipe_to_nvws
  tt.func @lower_pipe_to_nvws(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false

    // CHECK: %[[TAGS:.*]] = ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<2x1xi32
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    // CHECK-SAME: numBuffers = 2
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_acquire %[[TOKEN]]
    // CHECK-SAME: async_task_id
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_acquire %[[TOKEN]]
    // CHECK: arith.constant 1 : i32
    // CHECK: ttg.local_store
    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_close %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id
    // CHECK: %[[TAG:.*]] = ttg.local_load
    // CHECK: %[[TAG_I32:.*]] = tt.unsplat %[[TAG]]
    // CHECK: %[[CLOSED:.*]] = arith.cmpi ne, %[[TAG_I32]]
    %closed = tle.pipe.reader_wait %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: scf.if %[[CLOSED]]
    scf.if %closed {
    }

    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    tle.pipe.reader_release %a[%c1] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_cpasync_pipe_commit_to_nvws
  tt.func @lower_cpasync_pipe_commit_to_nvws(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.producer_commit
    // CHECK-SAME: commitKind
    // CHECK-NOT: tle.pipe_commit_cp_async
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_one_shot_pipe_to_nvws
  tt.func @lower_one_shot_pipe_to_nvws(%a: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    // CHECK-NOT: ttg.local_alloc
    // CHECK: %[[TOKEN:.*]] = nvws.create_token {full_count = 128 : i32, loadType = 3 : i32, numBuffers = 1 : i32}
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], readers = ["left", "right"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK-NOT: nvws.producer_acquire
    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    %left_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // CHECK: arith.constant {{.*}}false
    // CHECK: scf.if
    scf.if %left_closed {
    }
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    %right_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 2>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 2>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // CHECK-NOT: nvws.consumer_release
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_multi_reader_pipe_to_nvws
  tt.func @lower_multi_reader_pipe_to_nvws(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    // CHECK-SAME: empty_count = 256 : i32
    // CHECK-SAME: full_count = 128 : i32
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id = array<i32: 1>
    %left_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    // CHECK-SAME: async_task_id = array<i32: 1>
    // CHECK-SAME: release_count = 128 : i32
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id = array<i32: 2>
    %right_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    // CHECK-SAME: async_task_id = array<i32: 2>
    // CHECK-SAME: release_count = 128 : i32
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
