// RUN: triton-opt %s -split-input-file --nvgpu-lower-ws-tokens | FileCheck %s --implicit-check-not=nvws.
// RUN: triton-opt %s -split-input-file --nvgpu-lower-ws-tokens --nvgpu-lower-ws-tokens | FileCheck %s --implicit-check-not=nvws.

// A Blackwell automatic-WS loop without NVWS tokens must be untouched, even
// when it uses multiple CTAs. Token lowering only supports single-CTA tokens.
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @automatic_loop
  // CHECK-NOT: ttng.init_barrier
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.num_stages = 3 : i32, tt.warp_specialize
  // CHECK-NOT: ttng.init_barrier
  // CHECK: tt.return
  tt.func @automatic_loop(%base: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %ptr = tt.addptr %base, %i : !tt.ptr<i32>, i32
      %value = tt.load %ptr : !tt.ptr<i32>
      tt.store %ptr, %value : !tt.ptr<i32>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

// A function containing both explicit NVWS state and an automatic-WS loop
// must lower the token without partitioning the loop or clearing num_stages.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mixed_tokens_and_automatic_loop
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.arrive_barrier {{.*}}, 128
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.num_stages = 3 : i32, tt.warp_specialize
  // CHECK: tt.return
  tt.func @mixed_tokens_and_automatic_loop(%base: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    nvws.producer_commit %token, %c0 {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %ptr = tt.addptr %base, %i : !tt.ptr<i32>, i32
      %value = tt.load %ptr : !tt.ptr<i32>
      tt.store %ptr, %value : !tt.ptr<i32>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}
