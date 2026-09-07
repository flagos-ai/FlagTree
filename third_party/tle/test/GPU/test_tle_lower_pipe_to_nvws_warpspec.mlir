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

// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @writer(%pipe_argument_0: i32, %a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.writer_acquire %pipe_argument_0, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %pipe_argument_0, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  tt.func private @reader(%pipe_argument_1: i32, %a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %closed = tle.pipe.reader_wait %pipe_argument_1, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %pipe_argument_1, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_warpspec_call
  tt.func @pipe_warpspec_call(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    // CHECK: %[[TAGS:.*]] = ttg.local_alloc
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    %pipe_identity_2 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]], %[[TAGS]])
    ttg.warp_specialize(%a, %pipe_identity_2) attributes {requestedRegisters = array<i32: 240>}
    // CHECK: default
    default {
      // CHECK: nvws.producer_acquire %[[TOKEN]]
      tt.call @writer(%pipe_identity_2, %a) : (i32, !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}, %[[PART_TOKEN:.*]]: tensor<2x!nvws.token>, %[[PART_TAGS:.*]]: !ttg.memdesc<2x1xi32
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_2_p0: i32) num_warps(4) {
      // CHECK: nvws.consumer_wait %[[PART_TOKEN]]
      // CHECK: ttg.memdesc_index %[[PART_TAGS]]
      tt.call @reader(%pipe_identity_2_p0, %arg0) : (i32, !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe.reader_wait
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_warpspec_explicit_multi_reader
  tt.func @pipe_warpspec_explicit_multi_reader(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: %[[SPMC_TOKEN:.*]] = nvws.create_token
    // CHECK-SAME: empty_count = 256 : i32
    // CHECK-SAME: full_count = 128 : i32
    %pipe_identity_3 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[SPMC_TOKEN]]
    ttg.warp_specialize(%a, %pipe_identity_3) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %[[SPMC_TOKEN]]
      tle.pipe.writer_acquire %pipe_identity_3, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %pipe_identity_3, %a[%c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_3_p0: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 1>, release_count = 128 : i32}
      %closed_left = tle.pipe.reader_wait %pipe_identity_3_p0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_3_p0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg1: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_3_p1: i32) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 2>, release_count = 128 : i32}
      %closed_right = tle.pipe.reader_wait %pipe_identity_3_p1, %arg1[%p1_c0, %p1_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_3_p1, %arg1[%p1_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_multi_partition_task_ids
  tt.func @pipe_multi_partition_task_ids(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK-DAG: nvws.create_token
    // CHECK-DAG: nvws.create_token
    %pipe_identity_4 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %pipe_identity_5 = tle.pipe.create %b {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    ttg.warp_specialize(%a, %b, %pipe_identity_4, %pipe_identity_5) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 0>}
      tle.pipe.writer_acquire %pipe_identity_4, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %pipe_identity_4, %a[%c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %pipe_identity_4_p0: i32, %pipe_identity_5_p0: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      %closed_left = tle.pipe.reader_wait %pipe_identity_4_p0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_4_p0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %pipe_identity_5_p0, %arg1[%p0_c0, %p0_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %pipe_identity_5_p0, %arg1[%p0_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg2: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg3: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %pipe_identity_4_p1: i32, %pipe_identity_5_p1: i32) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>}
      %closed_score = tle.pipe.reader_wait %pipe_identity_5_p1, %arg3[%p1_c0, %p1_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_5_p1, %arg3[%p1_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, i32, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_same_partition_writer_reader
  tt.func @pipe_same_partition_writer_reader(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    // CHECK-SAME: empty_count = 128 : i32
    // CHECK-SAME: full_count = 128 : i32
    %pipe_identity_6 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]]
    ttg.warp_specialize(%a, %pipe_identity_6) attributes {requestedRegisters = array<i32: 240>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_6_p0: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.producer_commit %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 1>, release_count = 128 : i32}
      tle.pipe.writer_acquire %pipe_identity_6_p0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %pipe_identity_6_p0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      %closed = tle.pipe.reader_wait %pipe_identity_6_p0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_6_p0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // A fieldless one-shot pipe is a control handoff, so commit has no payload
  // window and may live in a warp-specialize task different from pipe.create.
  // CHECK-LABEL: tt.func @fieldless_one_shot_handoff
  tt.func @fieldless_one_shot_handoff(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: %[[HANDOFF:.*]] = nvws.create_token
    // CHECK-SAME: full_count = 128 : i32
    %pipe_identity_7 = "tle.pipe.create"() {capacity = 1 : i32, pipe_name = "handoff", field_names = [], one_shot = true, scope = "cta"} : () -> i32

    // CHECK: ttg.warp_specialize(%[[HANDOFF]])
    ttg.warp_specialize(%a, %pipe_identity_7) attributes {requestedRegisters = array<i32: 24>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_commit %[[HANDOFF]]
      "tle.pipe.writer_commit"(%pipe_identity_7, %c0) {capacity = 1 : i32, pipe_name = "handoff", field_names = [], scope = "cta"} : (i32, i32) -> ()
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_7_p0: i32) num_warps(1) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}
      %closed = "tle.pipe.reader_wait"(%pipe_identity_7_p0, %p0_c0, %p0_false) {capacity = 1 : i32, pipe_name = "handoff", field_names = [], scope = "cta"} : (i32, i32, i1) -> i1
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // A cyclic fieldless pipe uses the complete token lifecycle but carries no
  // payload memdesc through the warp-specialize ABI.
  // CHECK-LABEL: tt.func @fieldless_cyclic_handoff
  tt.func @fieldless_cyclic_handoff(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: %[[CONTROL:.*]] = nvws.create_token
    // CHECK-SAME: empty_count = 32 : i32
    // CHECK-SAME: full_count = 128 : i32
    %pipe_identity_8 = "tle.pipe.create"() {capacity = 1 : i32, pipe_name = "cyclic_handoff", field_names = [], scope = "cta"} : () -> i32

    // CHECK: ttg.warp_specialize(%[[CONTROL]])
    ttg.warp_specialize(%a, %pipe_identity_8) attributes {requestedRegisters = array<i32: 24>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %[[CONTROL]]
      // CHECK: nvws.producer_commit %[[CONTROL]]
      "tle.pipe.writer_acquire"(%pipe_identity_8, %c0, %false) {capacity = 1 : i32, pipe_name = "cyclic_handoff", field_names = [], scope = "cta"} : (i32, i32, i1) -> ()
      "tle.pipe.writer_commit"(%pipe_identity_8, %c0) {capacity = 1 : i32, pipe_name = "cyclic_handoff", field_names = [], scope = "cta"} : (i32, i32) -> ()
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_8_p0: i32) num_warps(1) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait
      // CHECK: nvws.consumer_release
      %closed = "tle.pipe.reader_wait"(%pipe_identity_8_p0, %p0_c0, %p0_false) {capacity = 1 : i32, pipe_name = "cyclic_handoff", field_names = [], scope = "cta"} : (i32, i32, i1) -> i1
      "tle.pipe.reader_release"(%pipe_identity_8_p0, %p0_c0) {capacity = 1 : i32, pipe_name = "cyclic_handoff", field_names = [], scope = "cta"} : (i32, i32) -> ()
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // A close tag is initialized by the enclosing task but stored by the
  // producer task.  Its register tensor layout must therefore follow the
  // producer partition's warp count, not the module's default warp count.
  // CHECK-LABEL: tt.func @pipe_close_different_warp_counts
  tt.func @pipe_close_different_warp_counts(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %pipe_identity_9 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    ttg.warp_specialize(%a, %pipe_identity_9) attributes {requestedRegisters = array<i32: 24>}
    default {
      // CHECK: default
      // CHECK: nvws.consumer_wait
      %closed = tle.pipe.reader_wait %pipe_identity_9, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %pipe_identity_9, %a[%c0] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %pipe_identity_9_p0: i32) num_warps(1) {
      %p0_c0 = arith.constant 0 : i32
      %p0_true = arith.constant true
      // CHECK: partition0
      // CHECK: nvws.producer_acquire
      // CHECK: ttg.local_store
      // CHECK: nvws.producer_commit
      tle.pipe.writer_close %pipe_identity_9_p0, %arg0[%p0_c0, %p0_true] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }
}
