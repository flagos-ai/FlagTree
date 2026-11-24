// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads | FileCheck %s --check-prefixes=COMMON,OFF

// With the option (FLAGTREE_KEEP_SMEM_READ_ORDER), a local pointer load is
// re-issued at the layout conversion that uses it only when the conversion is
// in the same block and nothing in between may write the buffer the load
// reads: a write through a descriptor of the same allocation (through views),
// a store through a shared pointer, or an op whose effects are unknown. A
// write to another allocation or to global memory does not hold it back.
// Without the option the load is re-issued regardless.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @rematerialize_without_write_in_between
  tt.func @rematerialize_without_write_in_between(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @keep_convert_after_write
  tt.func @keep_convert_after_write(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON-NEXT: tt.store
    // OFF: tt.store
    tt.store %ptr, %v : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // The conversion follows the store: re-issuing the load there would read the new data.
    // ON: ttg.convert_layout %[[IDS]]
    // OFF: tt.load
    // OFF-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // A write to another allocation does not change what the load reads: the
  // load is re-issued either way.
  // COMMON-LABEL: tt.func @rematerialize_past_write_of_other_alloc
  tt.func @rematerialize_past_write_of_other_alloc(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %other = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared1, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: ttg.local_store
    ttg.local_store %v, %other : tensor<64xi32, #blocked> -> !ttg.memdesc<64xi32, #shared1, #smem, mutable>
    // COMMON: tt.load
    // COMMON-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // The store writes a slot of the buffer the load reads through a view: the
  // same allocation, so the conversion stays.
  // COMMON-LABEL: tt.func @keep_convert_after_write_through_view
  tt.func @keep_convert_after_write_through_view(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %c1 = arith.constant 1 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    %slot1 = ttg.memdesc_index %smem[%c1] : !ttg.memdesc<2x64xi32, #shared, #smem, mutable> -> !ttg.memdesc<64xi32, #shared1, #smem, mutable>
    // COMMON: ttg.local_store
    ttg.local_store %v, %slot1 : tensor<64xi32, #blocked> -> !ttg.memdesc<64xi32, #shared1, #smem, mutable>
    // ON: ttg.convert_layout %[[IDS]]
    // OFF: tt.load
    // OFF-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // A store to global memory does not change what the load reads: the load is
  // re-issued either way.
  // COMMON-LABEL: tt.func @rematerialize_past_global_store
  tt.func @rematerialize_past_global_store(%slot_scalar: i32, %v: tensor<64xi32, #blocked>, %gptr: tensor<64x!tt.ptr<i32>, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: tt.store
    tt.store %gptr, %v : tensor<64x!tt.ptr<i32>, #blocked>
    // COMMON: tt.load
    // COMMON-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // A store through the pointers of a tle.local_pointers of another allocation
  // writes another buffer: the load is re-issued at the conversion either way,
  // after the store.
  // COMMON-LABEL: tt.func @rematerialize_past_local_pointer_store_of_other_alloc
  tt.func @rematerialize_past_local_pointer_store_of_other_alloc(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %other = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %other_ptr = "tle.local_pointers"(%other, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: tt.store
    tt.store %other_ptr, %v : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: tt.load
    // COMMON-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // A store through a pointer whose tle.local_pointers is not seen (here it
  // goes through pointer arithmetic) may write any shared buffer, so the
  // conversion stays even though the pointer is of another allocation.
  // COMMON-LABEL: tt.func @keep_convert_after_store_through_pointer_arithmetic
  tt.func @keep_convert_after_store_through_pointer_arithmetic(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %other = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %other_ptr = "tle.local_pointers"(%other, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %moved_ptr = tt.addptr %other_ptr, %slot : tensor<64x!tt.ptr<i32, 3>, #blocked>, tensor<64xi32, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: tt.store
    tt.store %moved_ptr, %v : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: ttg.convert_layout %[[IDS]]
    // OFF: tt.load
    // OFF-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // An op whose effects are unknown (gpu.barrier declares none) may write
  // anything: the conversion stays.
  // COMMON-LABEL: tt.func @keep_convert_after_unknown_effects
  tt.func @keep_convert_after_unknown_effects(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: gpu.barrier
    gpu.barrier
    // ON: ttg.convert_layout %[[IDS]]
    // OFF: tt.load
    // OFF-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // The conversion is in a loop body while the load is outside: the loop may
  // write the buffer before the conversion runs, so the conversion stays.
  // COMMON-LABEL: tt.func @keep_convert_in_other_block
  tt.func @keep_convert_in_other_block(%slot_scalar: i32, %v: tensor<64xi32, #blocked>, %n: i32) -> tensor<64xi32, #blocked_alt> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0> : tensor<64xi32, #blocked_alt>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON: scf.for
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64xi32, #blocked_alt>) : i32 {
      // ON: ttg.convert_layout %[[IDS]]
      // OFF: tt.load
      // OFF-NOT: ttg.convert_layout
      %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
      %sum = arith.addi %acc, %out : tensor<64xi32, #blocked_alt>
      scf.yield %sum : tensor<64xi32, #blocked_alt>
    }
    tt.return %r : tensor<64xi32, #blocked_alt>
  }
}
