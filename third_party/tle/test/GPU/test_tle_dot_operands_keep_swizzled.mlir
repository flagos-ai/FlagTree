// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s

// On PPU, a kernel-written buffer with an explicit (non-trivial) swizzled layout
// keeps its encoding when it is read transposed as a dot operand: only a
// memdesc_trans view is added.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
  // CHECK-LABEL: @keep_stored_swizzled_ppu
  tt.func @keep_stored_swizzled_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
                %kv: tensor<64x64xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{.*}}, %[[ALLOC]] : tensor<64x64xbf16, #{{.*}}> -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    ttg.local_store %kv, %alloc : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
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

// On PPU, the plain default layout (max phase 1) is still retagged for the
// transposed read.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
  // CHECK-LABEL: @retag_plain_default_ppu
  tt.func @retag_plain_default_ppu(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
                %kv: tensor<64x64xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{.*}}, %[[ALLOC]] : tensor<64x64xbf16, #{{.*}}> -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    ttg.local_store %kv, %alloc : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
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

// Other targets keep retagging explicit swizzled layouts.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$ENC:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
  // CHECK-LABEL: @retag_stored_swizzled_cuda
  tt.func @retag_stored_swizzled_cuda(%q: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
                %kv: tensor<64x64xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{.*}}, %[[ALLOC]] : tensor<64x64xbf16, #{{.*}}> -> !ttg.memdesc<64x64xbf16, #[[$ENC]], #smem, mutable>
    ttg.local_store %kv, %alloc : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[ALLOC]] {order = array<i32: 1, 0>}
    // CHECK: ttg.local_load %[[TRANS]]
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #blocked_t>
    %b = ttg.convert_layout %trans : tensor<64x64xbf16, #blocked_t> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %q, %b, %cst : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}
