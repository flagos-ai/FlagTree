// RUN: triton-opt %s --nvgpu-test-ws-lower-token | FileCheck %s

#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @local_store_token
  // CHECK-NOT: nvws.
  // CHECK-COUNT-4: ttng.init_barrier {{.*}}, 128
  // CHECK: gpu.barrier
  // CHECK: arith.xori
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {{.*}}{async_task_id = array<i32: 0>, release_fence = true}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {{.*}}{async_task_id = array<i32: 1>}
  tt.func @local_store_token(%idx: i32, %phase: i1) {
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    nvws.producer_acquire %token, %idx, %phase {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32, i1
    nvws.producer_commit %token, %idx {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32
    nvws.consumer_wait %token, %idx, %phase {async_task_id = array<i32: 1>} : tensor<2x!nvws.token>, i32, i1
    nvws.consumer_release %token, %idx {async_task_id = array<i32: 1>} : tensor<2x!nvws.token>, i32
    tt.return
  }

  // CHECK-LABEL: @cpasync_commit_token
  // CHECK-NOT: nvws.
  // CHECK: ttng.async_copy_mbarrier_arrive {{.*}} {async_task_id = array<i32: 0>, noIncrement}
  tt.func @cpasync_commit_token(%idx: i32) {
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    nvws.producer_commit %token, %idx {async_task_id = array<i32: 0>, commitKind = 1 : i32} : tensor<2x!nvws.token>, i32
    tt.return
  }

  // CHECK-LABEL: @full_only_token
  // CHECK-NOT: nvws.
  // CHECK-COUNT-2: ttng.init_barrier {{.*}}, 128
  // CHECK-NOT: ttng.init_barrier
  // CHECK: gpu.barrier
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {{.*}}{async_task_id = array<i32: 0>, release_fence = true}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK-NOT: ttng.arrive_barrier {{.*}} {async_task_id = array<i32: 1>}
  tt.func @full_only_token(%idx: i32, %phase: i1) {
    %token = nvws.create_token {full_count = 128 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    nvws.producer_commit %token, %idx {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32
    nvws.consumer_wait %token, %idx, %phase {async_task_id = array<i32: 1>} : tensor<2x!nvws.token>, i32, i1
    tt.return
  }

  // CHECK-LABEL: @multi_reader_token_counts
  // CHECK-NOT: nvws.
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.init_barrier {{.*}}, 256
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.init_barrier {{.*}}, 256
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {{.*}}{async_task_id = array<i32: 1>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {{.*}}{async_task_id = array<i32: 2>}
  tt.func @multi_reader_token_counts(%idx: i32, %phase: i1) {
    %token = nvws.create_token {empty_count = 256 : i32, full_count = 128 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    nvws.consumer_wait %token, %idx, %phase {async_task_id = array<i32: 1>} : tensor<2x!nvws.token>, i32, i1
    nvws.consumer_release %token, %idx {async_task_id = array<i32: 1>, release_count = 128 : i32} : tensor<2x!nvws.token>, i32
    nvws.consumer_wait %token, %idx, %phase {async_task_id = array<i32: 2>} : tensor<2x!nvws.token>, i32, i1
    nvws.consumer_release %token, %idx {async_task_id = array<i32: 2>, release_count = 128 : i32} : tensor<2x!nvws.token>, i32
    tt.return
  }

}
