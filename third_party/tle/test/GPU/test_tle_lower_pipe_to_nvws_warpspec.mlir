// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @writer(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  tt.func private @reader(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_warpspec_call
  tt.func @pipe_warpspec_call(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    // CHECK: %[[TAGS:.*]] = ttg.local_alloc
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]], %[[TAGS]])
    ttg.warp_specialize(%a) attributes {requestedRegisters = array<i32: 240>}
    // CHECK: default
    default {
      // CHECK: nvws.producer_acquire %[[TOKEN]]
      tt.call @writer(%a) : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}, %[[PART_TOKEN:.*]]: tensor<2x!nvws.token>, %[[PART_TAGS:.*]]: !ttg.memdesc<2x1xi32
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(4) {
      // CHECK: nvws.consumer_wait %[[PART_TOKEN]]
      // CHECK: ttg.memdesc_index %[[PART_TAGS]]
      tt.call @reader(%arg0) : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_warpspec_explicit_multi_reader
  tt.func @pipe_warpspec_explicit_multi_reader(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: %[[SPMC_TOKEN:.*]] = nvws.create_token
    // CHECK-SAME: empty_count = 256 : i32
    // CHECK-SAME: full_count = 128 : i32
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[SPMC_TOKEN]]
    ttg.warp_specialize(%a) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %[[SPMC_TOKEN]]
      tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 1>, release_count = 128 : i32}
      %closed_left = tle.pipe.reader_wait %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg1: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 2>, release_count = 128 : i32}
      %closed_right = tle.pipe.reader_wait %arg1[%p1_c0, %p1_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %arg1[%p1_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_multi_partition_task_ids
  tt.func @pipe_multi_partition_task_ids(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK-DAG: nvws.create_token
    // CHECK-DAG: nvws.create_token
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.create %b {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    ttg.warp_specialize(%a, %b) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 0>}
      tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      %closed_left = tle.pipe.reader_wait %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg1[%p0_c0, %p0_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %arg1[%p0_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg2: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg3: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>}
      %closed_score = tle.pipe.reader_wait %arg3[%p1_c0, %p1_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %arg3[%p1_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) -> ()
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
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]]
    ttg.warp_specialize(%a) attributes {requestedRegisters = array<i32: 240>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.producer_commit %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_release %{{.*}}{{.*}} {async_task_id = array<i32: 1>, release_count = 128 : i32}
      tle.pipe.writer_acquire %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      %closed = tle.pipe.reader_wait %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }
}
