// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-stores | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_store_with_known_full_tile_mask
  // CHECK-NOT: ttg.local_load
  // CHECK-NOT: arith.select
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @masked_store_with_known_full_tile_mask(%value: tensor<64x256xbf16, #blocked>) {
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %pid_h = tt.get_program_id z : i32
    %group_h = arith.remsi %pid_h, %c2_i32 : i32
    %h_base = arith.muli %group_h, %c64_i32 : i32
    %h_base_s = tt.splat %h_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %h_limit_s = tt.splat %c128_i32 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_h = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %h_idx = arith.addi %h_base_s, %offs_h : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_h_1d = arith.cmpi slt, %h_idx, %h_limit_s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_h_2d = tt.expand_dims %mask_h_1d {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
    %mask_h = tt.broadcast %mask_h_2d : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %d_limit_s = tt.splat %c512_i32 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_d = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_1d = arith.cmpi slt, %offs_d, %d_limit_s : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_2d = tt.expand_dims %mask_d_1d {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi1, #blocked>
    %mask_d = tt.broadcast %mask_d_2d : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %mask = arith.andi %mask_h, %mask_d : tensor<64x256xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x256xbf16, #shared, #smem, mutable>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.store %ptr, %value, %mask : tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_store_with_partial_tile_mask
  // CHECK: ttg.local_load
  // CHECK: arith.select
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @masked_store_with_partial_tile_mask(%value: tensor<64x256xbf16, #blocked>) {
    %c255_i32 = arith.constant 255 : i32
    %d_limit_s = tt.splat %c255_i32 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_d = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_1d = arith.cmpi slt, %offs_d, %d_limit_s : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_2d = tt.expand_dims %mask_d_1d {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi1, #blocked>
    %mask = tt.broadcast %mask_d_2d : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x256xbf16, #shared, #smem, mutable>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.store %ptr, %value, %mask : tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}
