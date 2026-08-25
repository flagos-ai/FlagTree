// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_non_power_of_two_capacity
  tt.func @lower_non_power_of_two_capacity(%a: !ttg.memdesc<3x16xf16, #shared, #smem, mutable>) {
    // The logical pipe and payload keep capacity 3, while private control
    // tensors are padded to 4.
    // CHECK: nvws.create_token
    // CHECK-SAME: numBuffers = 4
    // CHECK-SAME: tensor<4x!nvws.token>
    %pipe_identity_5 = tle.pipe.create %a {capacity = 3 : i32, pipe_name = "three_stage", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    // CHECK-NOT: tle.pipe
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
    %pipe_identity_4 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_acquire %[[TOKEN]]
    // CHECK-SAME: async_task_id
    tle.pipe.writer_acquire %pipe_identity_4, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_commit %pipe_identity_4, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.producer_acquire %[[TOKEN]]
    // CHECK: ttg.local_store
    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_close %pipe_identity_4, %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id
    // CHECK-SAME: waitMode = 1 : i32
    %closed = tle.pipe.reader_wait %pipe_identity_4, %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }

    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    // CHECK-SAME: tle.participant_consumer_release
    tle.pipe.reader_release %pipe_identity_4, %a[%c1] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK-NOT: tle.pipe
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_cpasync_pipe_commit_to_nvws
  tt.func @lower_cpasync_pipe_commit_to_nvws(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %pipe_identity_3 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.producer_commit
    // CHECK-SAME: commitKind
    // CHECK-NOT: tle.pipe_commit_cp_async
    tle.pipe.writer_commit %pipe_identity_3, %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_one_shot_pipe_to_nvws
  tt.func @lower_one_shot_pipe_to_nvws(%a: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    // CHECK-NOT: ttg.local_alloc
    // CHECK: %[[TOKEN:.*]] = nvws.create_token {full_count = 128 : i32, loadType = 3 : i32, numBuffers = 1 : i32}
    %pipe_identity_2 = tle.pipe.create %a {capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], readers = ["left", "right"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK-NOT: nvws.producer_acquire
    // CHECK: nvws.producer_commit %[[TOKEN]]
    tle.pipe.writer_acquire %pipe_identity_2, %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %pipe_identity_2, %a[%c0] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-NOT: waitMode
    %left_closed = tle.pipe.reader_wait %pipe_identity_2, %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    scf.if %left_closed {
    }
    tle.pipe.reader_release %pipe_identity_2, %a[%c0] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    %right_closed = tle.pipe.reader_wait %pipe_identity_2, %a[%c0, %false] {async_task_id = array<i32: 2>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %pipe_identity_2, %a[%c0] {async_task_id = array<i32: 2>, capacity = 1 : i32, pipe_name = "ready", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
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
    %pipe_identity_1 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    tle.pipe.writer_acquire %pipe_identity_1, %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %pipe_identity_1, %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id = array<i32: 1>
    // CHECK-SAME: waitMode = 1 : i32
    %left_closed = tle.pipe.reader_wait %pipe_identity_1, %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    // CHECK-SAME: async_task_id = array<i32: 1>
    // CHECK-SAME: release_count = 128 : i32
    // CHECK-SAME: tle.participant_consumer_release
    tle.pipe.reader_release %pipe_identity_1, %a[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    // CHECK: nvws.consumer_wait %[[TOKEN]]
    // CHECK-SAME: async_task_id = array<i32: 2>
    // CHECK-SAME: waitMode = 1 : i32
    %right_closed = tle.pipe.reader_wait %pipe_identity_1, %a[%c0, %false] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.consumer_release %[[TOKEN]], %{{.*}}, %arg0
    // CHECK-SAME: async_task_id = array<i32: 2>
    // CHECK-SAME: release_count = 128 : i32
    // CHECK-SAME: tle.participant_consumer_release
    tle.pipe.reader_release %pipe_identity_1, %a[%c0] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "broadcast", field_names = ["a"], reader_name = "right", scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_pipe_drain_to_named_barrier
  tt.func @lower_pipe_drain_to_named_barrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false

    // CHECK: %[[DRAIN:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
    // CHECK: ttng.init_barrier %[[DRAIN]], 256
    // CHECK: gpu.barrier
    %pipe_identity_0 = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    tle.pipe.writer_close %pipe_identity_0, %a[%c1, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: ttng.arrive_barrier %[[DRAIN]], 128 {async_task_id = array<i32: 0>, participant_arrive = true, release_fence = true}
    // CHECK: ttng.wait_barrier %[[DRAIN]], %{{.*}} {async_task_id = array<i32: 0>}
    tle.pipe.drain %pipe_identity_0, %a {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>

    %closed = tle.pipe.reader_wait %pipe_identity_0, %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %pipe_identity_0, %a[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK: ttng.arrive_barrier %[[DRAIN]], 128 {async_task_id = array<i32: 1>, participant_arrive = true, release_fence = true}
    // CHECK: ttng.wait_barrier %[[DRAIN]], %{{.*}} {async_task_id = array<i32: 1>}
    tle.pipe.drain %pipe_identity_0, %a {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "drain", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    // CHECK-NOT: tle.pipe
    tt.return
  }
}
