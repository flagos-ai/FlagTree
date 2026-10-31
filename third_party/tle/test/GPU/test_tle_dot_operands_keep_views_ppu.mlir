// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s

// On PPU, a view of a shared buffer that is read transposed as a dot operand
// keeps the buffer's encoding even for the plain default layout: the
// allocation and its other slots keep that encoding, so retagging the view
// would read the same bytes with another layout. Only a memdesc_trans view is
// added.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  // CHECK-LABEL: @keep_ring_slot_ppu
  tt.func @keep_ring_slot_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c1 = arith.constant 1 : i32
    // CHECK: %[[RING:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #[[$ENC]], #smem, mutable>
    %ring = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    // CHECK: %[[SLOT:.*]] = ttg.memdesc_index %[[RING]][%{{.*}}] : !ttg.memdesc<2x64x64xbf16, #[[$ENC]], #smem, mutable> -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %slot = ttg.memdesc_index %ring[%c1] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %load = ttg.local_load %slot : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[SLOT]] {order = array<i32: 1, 0>}
    // CHECK: ttg.local_load %[[TRANS]]
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #blocked_t>
    %b = ttg.convert_layout %trans : tensor<64x64xbf16, #blocked_t> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %q, %b, %cst : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// On PPU, a buffer with a non-trivial swizzled layout keeps it whether or not
// the kernel writes the buffer with local_store: the layout was chosen for
// its writers (an async copy through a local pointer, here none at all).

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
  // CHECK-LABEL: @keep_swizzled_without_store_ppu
  tt.func @keep_swizzled_without_store_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[ALLOC]] {order = array<i32: 1, 0>}
    // CHECK: ttg.local_load %[[TRANS]]
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #blocked_t>
    %b = ttg.convert_layout %trans : tensor<64x64xbf16, #blocked_t> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %q, %b, %cst : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// On PPU, a plain default-layout buffer that also has a view (here a subslice)
// keeps its encoding when it is read transposed: retagging the allocation
// alone would leave the view on the old encoding for the same bytes.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  // CHECK-LABEL: @keep_alloc_with_view_ppu
  tt.func @keep_alloc_with_view_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> (tensor<64x64xf32, #mma>, tensor<64x32xbf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_subslice %[[ALLOC]][0, 32] : !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable> -> !ttg.memdesc<64x32xbf16, #[[$ENC]], #smem, mutable, 64x64>
    %half = ttg.memdesc_subslice %alloc [0, 32] : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 64x64>
    %h = ttg.local_load %half : !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 64x64> -> tensor<64x32xbf16, #blocked>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[ALLOC]] {order = array<i32: 1, 0>}
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #blocked_t>
    %b = ttg.convert_layout %trans : tensor<64x64xbf16, #blocked_t> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %q, %b, %cst : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    tt.return %d, %h : tensor<64x64xf32, #mma>, tensor<64x32xbf16, #blocked>
  }
}

// -----

// On PPU, a buffer with a view keeps its encoding when it is read transposed
// even if its allocation has an initial value: the view stays on that encoding.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  // CHECK-LABEL: @keep_initialized_alloc_with_view_ppu
  tt.func @keep_initialized_alloc_with_view_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %t: tensor<64x64xbf16, #blocked>) -> (tensor<64x64xf32, #mma>, tensor<64x32xbf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %{{.*}} : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem>
    %alloc = ttg.local_alloc %t : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
    // CHECK: ttg.memdesc_subslice %[[ALLOC]][0, 32] : !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem> -> !ttg.memdesc<64x32xbf16, #[[$ENC]], #smem, 64x64>
    %half = ttg.memdesc_subslice %alloc [0, 32] : !ttg.memdesc<64x64xbf16, #shared, #smem> -> !ttg.memdesc<64x32xbf16, #shared, #smem, 64x64>
    %h = ttg.local_load %half : !ttg.memdesc<64x32xbf16, #shared, #smem, 64x64> -> tensor<64x32xbf16, #blocked>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x64xbf16, #shared, #smem> -> tensor<64x64xbf16, #blocked>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[ALLOC]] {order = array<i32: 1, 0>}
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #blocked_t>
    %b = ttg.convert_layout %trans : tensor<64x64xbf16, #blocked_t> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %q, %b, %cst : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    tt.return %d, %h : tensor<64x64xf32, #mma>, tensor<64x32xbf16, #blocked>
  }
}
