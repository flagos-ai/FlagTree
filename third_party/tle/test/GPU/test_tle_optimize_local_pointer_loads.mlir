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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 5}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @load_rank_collapsed_full_view
  tt.func @load_rank_collapsed_full_view() -> tensor<64x128xbf16, #rhs> {
    %zero = arith.constant dense<0> : tensor<64x128xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %zero, %zero, %rowb, %zero, %colb) : (!ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %value = tt.load %ptr : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK-NOT: ttg.memdesc_subslice
    // CHECK: %[[VIEW:.*]] = ttg.memdesc_reshape %[[BASE:.*]] : !ttg.memdesc<1x1x64x1x128xbf16, #{{.*}}, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #{{.*}}, #smem, mutable>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %[[VIEW]] : {{.*}} -> tensor<64x128xbf16, #ttg.dot_op<
    // CHECK-NOT: tt.load
    // CHECK-NOT: ttg.convert_layout
    %out = ttg.convert_layout %value : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #rhs>
    tt.return %out : tensor<64x128xbf16, #rhs>
  }

  // CHECK-LABEL: tt.func @load_rank_collapsed_tail_view
  tt.func @load_rank_collapsed_tail_view() -> tensor<32x128xbf16, #rhs> {
    %zero = arith.constant dense<0> : tensor<32x128xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>
    %tail = ttg.memdesc_subslice %smem[0, 0, 32, 0, 0] : !ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x1x32x1x128xbf16, #shared, #smem, mutable, 1x1x64x1x128>
    %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%tail, %zero, %zero, %rowb, %zero, %colb) : (!ttg.memdesc<1x1x32x1x128xbf16, #shared, #smem, mutable, 1x1x64x1x128>, tensor<32x128xi32, #blocked>, tensor<32x128xi32, #blocked>, tensor<32x128xi32, #blocked>, tensor<32x128xi32, #blocked>, tensor<32x128xi32, #blocked>) -> tensor<32x128x!tt.ptr<bf16, 3>, #blocked>
    %value = tt.load %ptr : tensor<32x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: %[[FULL:.*]] = ttg.memdesc_reshape %[[BASE:.*]] : !ttg.memdesc<1x1x64x1x128xbf16, #{{.*}}, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #{{.*}}, #smem, mutable>
    // CHECK: %[[TAIL:.*]] = ttg.memdesc_subslice %[[FULL]][32, 0] : {{.*}} -> !ttg.memdesc<32x128xbf16, #{{.*}}, #smem, mutable, 64x128>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %[[TAIL]] : {{.*}} -> tensor<32x128xbf16, #ttg.dot_op<
    // CHECK-NOT: tt.load
    // CHECK-NOT: ttg.convert_layout
    %out = ttg.convert_layout %value : tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #rhs>
    tt.return %out : tensor<32x128xbf16, #rhs>
  }

  // CHECK-LABEL: tt.func @do_not_collapse_unmarked_inline_asm
  tt.func @do_not_collapse_unmarked_inline_asm() -> tensor<64x128xbf16, #rhs> {
    %zero = arith.constant dense<0> : tensor<64x128xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col_opaque = tt.elementwise_inline_asm "mov.u32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %col : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col_opaque {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %zero, %zero, %rowb, %zero, %colb) : (!ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: %[[VALUE:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[VALUE]]
    %value = tt.load %ptr : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %out = ttg.convert_layout %value : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #rhs>
    tt.return %out : tensor<64x128xbf16, #rhs>
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
