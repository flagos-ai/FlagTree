// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-async-stores | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @async_store_static_subslice
  tt.func @async_store_static_subslice(%gptr: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %c128 = arith.constant 128 : i32
    %c128t = tt.splat %c128 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col.off = arith.addi %col, %c128t : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col.off {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %rowb, %colb) : (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %v = tt.load %gptr : tensor<64x128x!tt.ptr<bf16>, #blocked>
    // CHECK: %[[SUB:.*]] = ttg.memdesc_subslice %[[BASE:.*]][0, 128]
    // CHECK: %[[TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[SUB]] {tle.local_ptr_async_store}
    // CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[TOK]]
    // CHECK: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
    // CHECK-NOT: tt.store
    tt.store %ptr, %v : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @async_store_static_subslice_with_value_convert
  tt.func @async_store_static_subslice_with_value_convert(%gptr: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %c128 = arith.constant 128 : i32
    %c128t = tt.splat %c128 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_alt}>>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_alt}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_alt}>> -> tensor<64x1xi32, #blocked_alt>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked_alt> -> tensor<64x128xi32, #blocked_alt>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_alt}>>
    %col.off = arith.addi %col, %c128t : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_alt}>>
    %col2d = tt.expand_dims %col.off {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_alt}>> -> tensor<1x128xi32, #blocked_alt>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked_alt> -> tensor<64x128xi32, #blocked_alt>
    %ptr = "tle.local_pointers"(%smem, %rowb, %colb) : (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked_alt>, tensor<64x128xi32, #blocked_alt>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked_alt>
    %v = tt.load %gptr : tensor<64x128x!tt.ptr<bf16>, #blocked>
    %v.cvt = ttg.convert_layout %v : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked_alt>
    // CHECK: %[[SUB:.*]] = ttg.memdesc_subslice %[[BASE:.*]][0, 128]
    // CHECK: %[[TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[SUB]] {tle.local_ptr_async_store}
    // CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[TOK]]
    // CHECK: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
    // CHECK-NOT: ttg.convert_layout
    // CHECK-NOT: tt.store
    tt.store %ptr, %v.cvt : tensor<64x128x!tt.ptr<bf16, 3>, #blocked_alt>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_cluster_shared_load_store
  tt.func @preserve_cluster_shared_load_store(%rptr: tensor<64x!tt.ptr<i32, 7>, #blocked>) {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %local = "tle.local_pointers"(%smem, %offs) : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // CHECK-NOT: ttg.async_copy_global_to_local
    // CHECK: %[[V:.*]] = tt.load %{{.*}} : tensor<64x!tt.ptr<i32, 7>, #blocked>
    // CHECK: tt.store %{{.*}}, %[[V]]
    %v = tt.load %rptr : tensor<64x!tt.ptr<i32, 7>, #blocked>
    tt.store %local, %v : tensor<64x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}
