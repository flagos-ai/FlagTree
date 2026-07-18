// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @disjoint_async_copy
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @disjoint_async_copy(%gptr0: tensor<1x4x!tt.ptr<f32>, #blocked2>, %gptr1: tensor<1x4x!tt.ptr<f32>, #blocked2>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<1x4xf32, #blocked2>
    %mask = arith.constant dense<true> : tensor<1x4xi1, #blocked2>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x8xf32, #shared2, #smem, mutable>
    %left = ttg.memdesc_subslice %smem[0, 0] : !ttg.memdesc<1x8xf32, #shared2, #smem, mutable> -> !ttg.memdesc<1x4xf32, #shared2, #smem, mutable, 1x8>
    %right = ttg.memdesc_subslice %smem[0, 4] : !ttg.memdesc<1x8xf32, #shared2, #smem, mutable> -> !ttg.memdesc<1x4xf32, #shared2, #smem, mutable, 1x8>
    ttg.async_copy_global_to_local %gptr0, %left mask %mask other %c0 : tensor<1x4x!tt.ptr<f32>, #blocked2> -> <1x4xf32, #shared2, #smem, mutable, 1x8>
    ttg.async_copy_global_to_local %gptr1, %right mask %mask other %c0 : tensor<1x4x!tt.ptr<f32>, #blocked2> -> <1x4xf32, #shared2, #smem, mutable, 1x8>
    tt.return
  }

  // CHECK-LABEL: llvm.func @overlap_async_copy
  // CHECK: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @overlap_async_copy(%gptr0: tensor<1x4x!tt.ptr<f32>, #blocked2>, %gptr1: tensor<1x4x!tt.ptr<f32>, #blocked2>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<1x4xf32, #blocked2>
    %mask = arith.constant dense<true> : tensor<1x4xi1, #blocked2>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x8xf32, #shared2, #smem, mutable>
    %left0 = ttg.memdesc_subslice %smem[0, 0] : !ttg.memdesc<1x8xf32, #shared2, #smem, mutable> -> !ttg.memdesc<1x4xf32, #shared2, #smem, mutable, 1x8>
    %left1 = ttg.memdesc_subslice %smem[0, 0] : !ttg.memdesc<1x8xf32, #shared2, #smem, mutable> -> !ttg.memdesc<1x4xf32, #shared2, #smem, mutable, 1x8>
    ttg.async_copy_global_to_local %gptr0, %left0 mask %mask other %c0 : tensor<1x4x!tt.ptr<f32>, #blocked2> -> <1x4xf32, #shared2, #smem, mutable, 1x8>
    ttg.async_copy_global_to_local %gptr1, %left1 mask %mask other %c0 : tensor<1x4x!tt.ptr<f32>, #blocked2> -> <1x4xf32, #shared2, #smem, mutable, 1x8>
    tt.return
  }

  // CHECK-LABEL: llvm.func @disjoint_local_ptr_rows
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @disjoint_local_ptr_rows(%v0: tensor<64xi8, #blocked1>, %v1: tensor<64xi8, #blocked1>) {
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %row1 = arith.constant dense<1> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %ptr1 = "tle.local_pointers"(%smem, %row1, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.return
  }

  // CHECK-LABEL: llvm.func @overlap_local_ptr_rows
  // CHECK: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @overlap_local_ptr_rows(%v0: tensor<64xi8, #blocked1>, %v1: tensor<64xi8, #blocked1>) {
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    %ptr1 = "tle.local_pointers"(%smem, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.return
  }

  // CHECK-LABEL: llvm.func @plain_arrive_after_local_store_keeps_cta_barrier
  // CHECK: nvvm.barrier0
  // CHECK: "membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 128;", "b,r"
  // CHECK: llvm.return
  tt.func @plain_arrive_after_local_store_keeps_cta_barrier(%barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %v0: tensor<64xi8, #blocked1>) {
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    ttng.arrive_barrier %barrier, 128 {release_fence = true} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: llvm.func @participant_arrive_after_local_store_uses_lane_fences
  // CHECK-NOT: nvvm.barrier0
  // CHECK: "@$0 membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r"
  // CHECK: llvm.return
  tt.func @participant_arrive_after_local_store_uses_lane_fences(%barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %v0: tensor<64xi8, #blocked1>) {
    %row0 = arith.constant dense<0> : tensor<64xi32, #blocked1>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi8, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem, %row0, %offs) : (!ttg.memdesc<2x64xi8, #shared2, #smem, mutable>, tensor<64xi32, #blocked1>, tensor<64xi32, #blocked1>) -> tensor<64x!tt.ptr<i8, 3>, #blocked1>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked1>
    ttng.arrive_barrier %barrier, 64 {participant_arrive = true, release_fence = true} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    tt.return
  }
}
