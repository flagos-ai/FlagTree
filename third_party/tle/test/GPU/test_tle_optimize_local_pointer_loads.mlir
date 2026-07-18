// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @load_static_subslice
  tt.func @load_static_subslice() -> tensor<64x128xbf16, #blocked> {
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
    // CHECK: %[[SUB:.*]] = ttg.memdesc_subslice %[[BASE:.*]][0, 128]
    // CHECK: %[[LOAD:.*]] = ttg.local_load %[[SUB]]
    // CHECK-NOT: tt.load
    %v = tt.load %ptr : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return %v : tensor<64x128xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @load_full_view
  tt.func @load_full_view() -> tensor<64x128xbf16, #blocked> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x128xbf16, #shared, #smem, mutable>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK-NOT: ttg.memdesc_subslice
    // CHECK: ttg.local_load %[[BASE:.*]]
    // CHECK-NOT: tt.load
    %v = tt.load %ptr : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return %v : tensor<64x128xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_load_is_not_rewritten
  tt.func @masked_load_is_not_rewritten(%mask: tensor<64x128xi1, #blocked>) -> tensor<64x128xbf16, #blocked> {
    %other = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x128xbf16, #shared, #smem, mutable>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: tt.load
    // CHECK-NOT: ttg.local_load
    %v = tt.load %ptr, %mask, %other : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return %v : tensor<64x128xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @rematerialize_local_ptr_load_for_convert_layout
  tt.func @rematerialize_local_ptr_load_for_convert_layout(%slot_scalar: i32) -> tensor<64xi32, #blocked_alt> {
    %c0 = arith.constant dense<0> : tensor<64xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    %mask = arith.cmpi sge, %ids, %c0 : tensor<64xi32, #blocked>
    %safe = arith.select %mask, %ids, %c0 : tensor<64xi1, #blocked>, tensor<64xi32, #blocked>
    // CHECK-SAME: -> tensor<64xi32, #[[TARGET:[A-Za-z0-9_]+]]>
    // CHECK: tt.splat %{{.*}} : i32 -> tensor<64xi32, #[[TARGET]]>
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #[[TARGET]]>
    // CHECK: %[[PTR:.*]] = "tle.local_pointers"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ttg.memdesc<2x64xi32, #{{[A-Za-z0-9_]+}}, #smem, mutable>, tensor<64xi32, #[[TARGET]]>, tensor<64xi32, #[[TARGET]]>) -> tensor<64x!tt.ptr<i32, 3>, #[[TARGET]]>
    // CHECK: %[[IDS:.*]] = tt.load %[[PTR]] : tensor<64x!tt.ptr<i32, 3>, #[[TARGET]]>
    // CHECK: %[[MASK:.*]] = arith.cmpi sge, %[[IDS]], %{{.*}} : tensor<64xi32, #[[TARGET]]>
    // CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[IDS]], %{{.*}} : tensor<64xi1, #[[TARGET]]>, tensor<64xi32, #[[TARGET]]>
    // CHECK-NEXT: tt.return %[[SAFE]]
    // CHECK-NOT: ttg.convert_layout
    %out = ttg.convert_layout %safe : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#vec = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#ptr_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#ptr_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @rematerialize_local_ptr_load_for_addptr_convert
  tt.func @rematerialize_local_ptr_load_for_addptr_convert(%slot_scalar: i32, %gptr: !tt.ptr<bf16>) -> tensor<64x1x!tt.ptr<bf16>, #ptr_dst> {
    %c0 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %stride = arith.constant dense<576> : tensor<64x1xi64, #ptr_src>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %idx_ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>) -> tensor<64x!tt.ptr<i32, 3>, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %ids = tt.load %idx_ptr : tensor<64x!tt.ptr<i32, 3>, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %mask = arith.cmpi sge, %ids, %c0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %safe = arith.select %mask, %ids, %c0 : tensor<64xi1, #ttg.slice<{dim = 1, parent = #ptr_src}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %safe64 = arith.extsi %safe : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ptr_src}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #ptr_src}>>
    %safe2d = tt.expand_dims %safe64 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ptr_src}>> -> tensor<64x1xi64, #ptr_src>
    %offset = arith.muli %safe2d, %stride : tensor<64x1xi64, #ptr_src>
    %base = tt.splat %gptr : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #ptr_src>
    %ptr = tt.addptr %base, %offset : tensor<64x1x!tt.ptr<bf16>, #ptr_src>, tensor<64x1xi64, #ptr_src>
    // CHECK-SAME: -> tensor<64x1x!tt.ptr<bf16>, #[[DST:[A-Za-z0-9_]+]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[PTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<64x1x!tt.ptr<bf16>, #[[DST]]>, tensor<64x1xi64, #[[DST]]>
    // CHECK-NEXT: tt.return %[[PTR]]
    %out = ttg.convert_layout %ptr : tensor<64x1x!tt.ptr<bf16>, #ptr_src> -> tensor<64x1x!tt.ptr<bf16>, #ptr_dst>
    tt.return %out : tensor<64x1x!tt.ptr<bf16>, #ptr_dst>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_dynamic_index_convert_layout
  tt.func @preserve_dynamic_index_convert_layout(%slot: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %c0 = arith.constant dense<0> : tensor<64xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    %mask = arith.cmpi sge, %ids, %c0 : tensor<64xi32, #blocked>
    %safe = arith.select %mask, %ids, %c0 : tensor<64xi1, #blocked>, tensor<64xi32, #blocked>
    // CHECK-SAME: (%{{.*}}: tensor<64xi32, #[[SRC:[A-Za-z0-9_]+]]>) -> tensor<64xi32, #[[DST:[A-Za-z0-9_]+]]>
    // CHECK: "tle.local_pointers"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ttg.memdesc<2x64xi32, #{{[A-Za-z0-9_]+}}, #smem, mutable>, tensor<64xi32, #[[SRC]]>, tensor<64xi32, #[[SRC]]>) -> tensor<64x!tt.ptr<i32, 3>, #[[SRC]]>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<64xi32, #[[SRC]]> -> tensor<64xi32, #[[DST]]>
    %out = ttg.convert_layout %safe : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}
