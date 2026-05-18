// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws | FileCheck --check-prefix=NVWS %s
// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token | FileCheck --check-prefix=TTNG %s

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // NVWS-LABEL: tt.func @participant_two_local_ptr_writes
  // NVWS: %[[TOKEN:.*]] = nvws.create_token
  // NVWS-SAME: full_count = 64 : i32
  // NVWS: nvws.producer_commit %[[TOKEN]]
  // NVWS-SAME: commitKind = 3 : i32
  //
  // TTNG-LABEL: tt.func @participant_two_local_ptr_writes
  // TTNG: ttng.init_barrier {{.*}}, 64
  // TTNG: ttng.arrive_barrier {{.*}}, 64
  // TTNG-SAME: participant_arrive = true
  // TTNG-SAME: release_fence = true
  tt.func @participant_two_local_ptr_writes(%a: !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>, %v0: tensor<64xi8, #blocked1>, %v1: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %row1 = arith.constant dense<1> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>

    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable> -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%slot, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %ptr1 = "tle.local_pointers"(%slot, %row1, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "participant", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x2x64xi8, #shared3, #smem, mutable>
    tt.return
  }

  // NVWS-LABEL: tt.func @no_infer_without_stage_slot
  // NVWS: %[[TOKEN:.*]] = nvws.create_token
  // NVWS-SAME: full_count = 128 : i32
  // NVWS: nvws.producer_commit %[[TOKEN]]
  // NVWS-NOT: commitKind = 3 : i32
  // TTNG-LABEL: tt.func @no_infer_without_stage_slot
  // TTNG: ttng.init_barrier {{.*}}, 128
  // TTNG: ttng.arrive_barrier {{.*}}, 128
  // TTNG-NOT: participant_arrive = true
  tt.func @no_infer_without_stage_slot(%a: !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, %v0: tensor<64xi8, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>

    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "no_slot", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "no_slot", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%a, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "no_slot", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    tt.return
  }
}
