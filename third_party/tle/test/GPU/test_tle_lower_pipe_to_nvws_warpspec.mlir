// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @writer(%identity: i32, %a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.writer_acquire %identity, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %identity, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  tt.func private @reader(%identity: i32, %a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %closed = tle.pipe.reader_wait %identity, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %identity, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_warpspec_call
  tt.func @pipe_warpspec_call(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    %pipe_identity = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]])
    ttg.warp_specialize(%a, %pipe_identity) attributes {requestedRegisters = array<i32: 240>}
    // CHECK: default
    default {
      // CHECK: nvws.producer_acquire
      tt.call @writer(%pipe_identity, %a) : (i32, !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}, %[[PART_TOKEN:.*]]: tensor<2x!nvws.token>)
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %identity: i32) num_warps(4) {
      // CHECK: nvws.consumer_wait %[[PART_TOKEN]]
      // CHECK-SAME: waitMode = 1 : i32
      tt.call @reader(%identity, %arg0) : (i32, !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
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
    %pipe_identity = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[SPMC_TOKEN]]
    ttg.warp_specialize(%a, %pipe_identity) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %[[SPMC_TOKEN]]
      tle.pipe.writer_acquire %pipe_identity, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %pipe_identity, %a[%c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %identity0: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>, waitMode = 1 : i32}
      // CHECK: nvws.consumer_release
      // CHECK-SAME: async_task_id = array<i32: 1>
      // CHECK-SAME: release_count = 128 : i32
      // CHECK-SAME: tle.participant_consumer_release
      %closed_left = tle.pipe.reader_wait %identity0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %identity0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg1: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %identity1: i32) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>, waitMode = 1 : i32}
      // CHECK: nvws.consumer_release
      // CHECK-SAME: async_task_id = array<i32: 2>
      // CHECK-SAME: release_count = 128 : i32
      // CHECK-SAME: tle.participant_consumer_release
      %closed_right = tle.pipe.reader_wait %identity1, %arg1[%p1_c0, %p1_false] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %identity1, %arg1[%p1_c0] {capacity = 2 : i32, pipe_name = "fanout", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
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
    %left_identity = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %score_identity = tle.pipe.create %b {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    ttg.warp_specialize(%a, %b, %left_identity, %score_identity) attributes {requestedRegisters = array<i32: 240, 168>}
    default {
      // CHECK: default
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 0>}
      tle.pipe.writer_acquire %left_identity, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %left_identity, %a[%c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %left0: i32, %score0: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>, waitMode = 1 : i32}
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      %closed_left = tle.pipe.reader_wait %left0, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %left0, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "left", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %score0, %arg1[%p0_c0, %p0_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %score0, %arg1[%p0_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg2: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %arg3: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %left1: i32, %score1: i32) num_warps(4) {
      %p1_c0 = arith.constant 0 : i32
      %p1_false = arith.constant false
      // CHECK: partition1
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 2>, waitMode = 1 : i32}
      %closed_score = tle.pipe.reader_wait %score1, %arg3[%p1_c0, %p1_false] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %score1, %arg3[%p1_c0] {capacity = 1 : i32, pipe_name = "score", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
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
    %pipe_identity = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: ttg.warp_specialize(%arg0, %[[TOKEN]]
    ttg.warp_specialize(%a, %pipe_identity) attributes {requestedRegisters = array<i32: 240>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %identity: i32) num_warps(4) {
      %p0_c0 = arith.constant 0 : i32
      %p0_false = arith.constant false
      // CHECK: partition0
      // CHECK: nvws.producer_acquire %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.producer_commit %{{.*}}{{.*}} {async_task_id = array<i32: 1>}
      // CHECK: nvws.consumer_wait %{{.*}}{{.*}} {async_task_id = array<i32: 1>, waitMode = 1 : i32}
      // CHECK: nvws.consumer_release
      // CHECK-SAME: async_task_id = array<i32: 1>
      // CHECK-SAME: release_count = 128 : i32
      // CHECK-SAME: tle.participant_consumer_release
      tle.pipe.writer_acquire %identity, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %identity, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      %closed = tle.pipe.reader_wait %identity, %arg0[%p0_c0, %p0_false] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %identity, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "same_partition", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @pipe_drain_in_one_warp_partition
  tt.func @pipe_drain_in_one_warp_partition(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    // CHECK: %[[CLOSE_TAGS:.*]] = ttg.local_alloc
    // CHECK: %[[TOKEN:.*]] = nvws.create_token
    // CHECK: %[[DRAIN:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
    // CHECK: ttng.init_barrier %[[DRAIN]], 32
    %pipe_identity = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "drain_one_warp", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    ttg.warp_specialize(%a, %pipe_identity) attributes {requestedRegisters = array<i32: 64>}
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}, %{{.*}}: tensor<2x!nvws.token>, %[[PART_CLOSE_TAGS:.*]]: !ttg.memdesc<2x1xi32
    partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>, %identity: i32) num_warps(1) {
      %p0_c0 = arith.constant 0 : i32
      %p0_c1 = arith.constant 1 : i32
      %p0_false = arith.constant false
      // CHECK: nvws.producer_acquire
      // CHECK-SAME: async_task_id = array<i32: 1>
      // CHECK: ttg.memdesc_index %[[PART_CLOSE_TAGS]]
      // CHECK: ttg.local_store
      // CHECK-SAME: tensor<1xi32
      // CHECK: nvws.producer_commit
      // CHECK-SAME: async_task_id = array<i32: 1>
      tle.pipe.writer_close %identity, %arg0[%p0_c1, %p0_false] {capacity = 2 : i32, pipe_name = "drain_one_warp", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %identity, %arg0[%p0_c0] {capacity = 2 : i32, pipe_name = "drain_one_warp", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      // CHECK: ttng.arrive_barrier %{{.*}}, 32
      // CHECK-SAME: async_task_id = array<i32: 1>
      // CHECK-SAME: participant_arrive = true
      tle.pipe.drain %identity, %arg0 {capacity = 2 : i32, pipe_name = "drain_one_warp", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>, i32) -> ()
    // CHECK-NOT: tle.pipe
    tt.return
  }
}
