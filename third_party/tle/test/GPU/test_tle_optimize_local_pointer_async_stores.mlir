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

#load = #ttg.blocked<{sizePerThread = [1, 1, 1, 8], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [1, 8, 1, 1], order = [3, 0, 1, 2]}>
#local = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 4, 8], warpsPerCTA = [1, 8, 1, 1], order = [3, 2, 1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [3, 2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @async_store_singleton_axis
  tt.func @async_store_singleton_axis(%gptr: tensor<1x256x4x8x!tt.ptr<bf16>, #load>) {
    // Canonicalization folds arange(0, 1) and its broadcasts to this splat.
    %idx0 = arith.constant dense<0> : tensor<1x256x4x8xi32, #local>

    %idx1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #local}>}>}>>
    %idx1.1 = tt.expand_dims %idx1 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #local}>}>}>> -> tensor<1x256xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #local}>}>>
    %idx1.2 = tt.expand_dims %idx1.1 {axis = 2 : i32} : tensor<1x256xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #local}>}>> -> tensor<1x256x1xi32, #ttg.slice<{dim = 3, parent = #local}>>
    %idx1.3 = tt.expand_dims %idx1.2 {axis = 3 : i32} : tensor<1x256x1xi32, #ttg.slice<{dim = 3, parent = #local}>> -> tensor<1x256x1x1xi32, #local>
    %idx1.full = tt.broadcast %idx1.3 : tensor<1x256x1x1xi32, #local> -> tensor<1x256x4x8xi32, #local>

    %idx2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 3, parent = #local}>}>}>>
    %idx2.1 = tt.expand_dims %idx2 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 3, parent = #local}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 3, parent = #local}>}>>
    %idx2.2 = tt.expand_dims %idx2.1 {axis = 0 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 3, parent = #local}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 3, parent = #local}>>
    %idx2.3 = tt.expand_dims %idx2.2 {axis = 3 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 3, parent = #local}>> -> tensor<1x1x4x1xi32, #local>
    %idx2.full = tt.broadcast %idx2.3 : tensor<1x1x4x1xi32, #local> -> tensor<1x256x4x8xi32, #local>

    %idx3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #local}>}>}>>
    %idx3.1 = tt.expand_dims %idx3 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #local}>}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #local}>}>>
    %idx3.2 = tt.expand_dims %idx3.1 {axis = 0 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #local}>}>> -> tensor<1x1x8xi32, #ttg.slice<{dim = 0, parent = #local}>>
    %idx3.3 = tt.expand_dims %idx3.2 {axis = 0 : i32} : tensor<1x1x8xi32, #ttg.slice<{dim = 0, parent = #local}>> -> tensor<1x1x1x8xi32, #local>
    %idx3.full = tt.broadcast %idx3.3 : tensor<1x1x1x8xi32, #local> -> tensor<1x256x4x8xi32, #local>

    // CHECK: %[[SMEM:.*]] = ttg.local_alloc
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x256x4x8xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %idx0, %idx1.full, %idx2.full, %idx3.full) : (!ttg.memdesc<1x256x4x8xbf16, #shared, #smem, mutable>, tensor<1x256x4x8xi32, #local>, tensor<1x256x4x8xi32, #local>, tensor<1x256x4x8xi32, #local>, tensor<1x256x4x8xi32, #local>) -> tensor<1x256x4x8x!tt.ptr<bf16, 3>, #local>
    %value = tt.load %gptr : tensor<1x256x4x8x!tt.ptr<bf16>, #load>
    %value.local = ttg.convert_layout %value : tensor<1x256x4x8xbf16, #load> -> tensor<1x256x4x8xbf16, #local>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[SMEM]] {tle.local_ptr_async_store}
    // CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[TOK]]
    // CHECK: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
    // CHECK-NOT: tt.store
    tt.store %ptr, %value.local : tensor<1x256x4x8x!tt.ptr<bf16, 3>, #local>
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @independent_async_stores_keep_single_token_chains
  tt.func @independent_async_stores_keep_single_token_chains(%gptr0: tensor<64x128x!tt.ptr<bf16>, #blocked>, %gptr1: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr0 = "tle.local_pointers"(%smem0, %rowb, %colb) : (!ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%smem1, %rowb, %colb) : (!ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %v0 = tt.load %gptr0 : tensor<64x128x!tt.ptr<bf16>, #blocked>
    // CHECK: %[[TOK0:.*]] = ttg.async_copy_global_to_local %{{.*}}, %{{.*}} {tle.local_ptr_async_store}
    // CHECK: %[[COMMIT0:.*]] = ttg.async_commit_group tokens %[[TOK0]]
    // CHECK: ttg.async_wait %[[COMMIT0]] {num = 0 : i32}
    tt.store %ptr0, %v0 : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %v1 = tt.load %gptr1 : tensor<64x128x!tt.ptr<bf16>, #blocked>
    // CHECK: %[[TOK1:.*]] = ttg.async_copy_global_to_local %{{.*}}, %{{.*}} {tle.local_ptr_async_store}
    // CHECK: %[[COMMIT1:.*]] = ttg.async_commit_group tokens %[[TOK1]]
    // CHECK: ttg.async_wait %[[COMMIT1]] {num = 0 : i32}
    // CHECK-NOT: tt.store
    tt.store %ptr1, %v1 : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @async_store_pipe_commit
  tt.func @async_store_pipe_commit(%gptr0: tensor<64x128x!tt.ptr<bf16>, #blocked>, %gptr1: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %field0 = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>
    %field1 = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>
    %pipe_identity_28 = tle.pipe.create %field0, %field1 {capacity = 2 : i32, pipe_name = "async", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>
    %slot0 = ttg.memdesc_index %field0[%c0] : !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>
    %slot1 = ttg.memdesc_index %field1[%c0] : !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%slot0) : (!ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%slot1) : (!ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %v0 = tt.load %gptr0 : tensor<64x128x!tt.ptr<bf16>, #blocked>
    // CHECK: ttg.async_copy_global_to_local
    tt.store %ptr0, %v0 : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %v1 = tt.load %gptr1 : tensor<64x128x!tt.ptr<bf16>, #blocked>
    // CHECK: ttg.async_copy_global_to_local
    tt.store %ptr1, %v1 : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK-NOT: ttg.async_commit_group
    // CHECK-NOT: ttg.async_wait
    // CHECK: tle.pipe.writer_commit
    // CHECK-SAME: tle.pipe_commit_cp_async
    tle.pipe.writer_commit %pipe_identity_28, %field0, %field1[%c0] {capacity = 2 : i32, pipe_name = "async", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<2x64x128xbf16, #shared3, #smem, mutable>
    tt.return
  }
}
