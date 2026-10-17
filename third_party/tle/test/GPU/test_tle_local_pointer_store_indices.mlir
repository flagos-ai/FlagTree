// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-stores | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A full-shaped pointer tensor can still implement a dynamic scatter.
  // CHECK-LABEL: tt.func @dynamic_scatter
  // CHECK: "tle.local_pointers"
  // CHECK-NOT: ttg.local_store
  // CHECK: tt.store
  tt.func @dynamic_scatter(%value: tensor<64xi32, #blocked>, %indices: tensor<64xi32, #blocked>, %mask: tensor<64xi1, #blocked>) {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %indices) : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %value, %mask : tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func @reversed_scatter
  // CHECK: "tle.local_pointers"
  // CHECK-NOT: ttg.local_store
  // CHECK: tt.store
  tt.func @reversed_scatter(%value: tensor<64xi32, #blocked>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %last = arith.constant dense<63> : tensor<64xi32, #blocked>
    %indices = arith.subi %last, %range : tensor<64xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %indices) : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func @identity_store
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @identity_store(%value: tensor<64xi32, #blocked>) {
    %indices = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %indices) : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @identity_store_2d
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @identity_store_2d(%value: tensor<8x8xi32, #blocked>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %rows = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %cols = tt.broadcast %c2 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<8x8xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %rows, %cols) : (!ttg.memdesc<8x8xi32, #shared, #smem, mutable>, tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>) -> tensor<8x8x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %value : tensor<8x8x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
  // CHECK-LABEL: tt.func @transposed_store_2d
  // CHECK: "tle.local_pointers"
  // CHECK-NOT: ttg.local_store
  // CHECK: tt.store
  tt.func @transposed_store_2d(%value: tensor<8x8xi32, #blocked>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %rows = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %cols = tt.broadcast %c2 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<8x8xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %cols, %rows) : (!ttg.memdesc<8x8xi32, #shared, #smem, mutable>, tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>) -> tensor<8x8x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %value : tensor<8x8x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}
