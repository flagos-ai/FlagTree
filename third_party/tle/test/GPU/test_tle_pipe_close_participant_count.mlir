// RUN: triton-opt --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token %s | FileCheck %s

// Close publishes control data with a single elected-thread arrival. It must
// satisfy the existing data epoch, not change its proven participant count.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @participant_close
  // CHECK: ttng.init_barrier {{.*}}, 16
  // CHECK: ttng.arrive_barrier {{.*}}, 16
  // CHECK-SAME: participant_arrive = true
  // CHECK: ttng.arrive_barrier {{.*}}, 16
  // CHECK-SAME: release_fence = true
  // CHECK-NOT: participant_arrive = true
  tt.func @participant_close(%a: !ttg.memdesc<2x16xi32, #shared2, #smem, mutable>, %v: tensor<16xi32, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %pipe = tle.pipe.create %a {capacity = 2 : i32, pipe_name = "data", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xi32, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %a[%c0, %false] {capacity = 2 : i32, pipe_name = "data", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xi32, #shared2, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x16xi32, #shared2, #smem, mutable> -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%slot, %offsets) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>, tensor<16xi32, #blocked>) -> tensor<16x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %v : tensor<16x!tt.ptr<i32, 3>, #blocked>
    tle.pipe.writer_commit %pipe, %a[%c0] {capacity = 2 : i32, pipe_name = "data", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xi32, #shared2, #smem, mutable>
    tle.pipe.writer_close %pipe, %a[%c1, %false] {capacity = 2 : i32, pipe_name = "data", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xi32, #shared2, #smem, mutable>
    tt.return
  }
}
