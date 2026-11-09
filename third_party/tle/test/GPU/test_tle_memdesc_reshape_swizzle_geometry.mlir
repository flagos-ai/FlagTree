// RUN: triton-opt %s -split-input-file --tritongpu-optimize-dot-operands | FileCheck %s

// The transposed FP8 core has eight rows. Preserving the first dimension
// during a 3-D reshape does not preserve the collapsed physical row extent.
#src = #ttg.blocked<{sizePerThread = [4, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_broadcast_subtile
  // CHECK: %[[RESHAPE:.*]] = tt.reshape
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %[[RESHAPE]]
  // CHECK-NOT: ttg.memdesc_reshape
  // CHECK: tt.return %[[ALLOC]]
  tt.func @fp8_broadcast_subtile(%x: tensor<128x4x4xf8E4M3FN, #src>) -> !ttg.memdesc<128x16xf8E4M3FN, #shared, #ttg.shared_memory> {
    %r = tt.reshape %x : tensor<128x4x4xf8E4M3FN, #src> -> tensor<128x16xf8E4M3FN, #dst>
    %a = ttg.local_alloc %r : (tensor<128x16xf8E4M3FN, #dst>) -> !ttg.memdesc<128x16xf8E4M3FN, #shared, #ttg.shared_memory>
    tt.return %a : !ttg.memdesc<128x16xf8E4M3FN, #shared, #ttg.shared_memory>
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @legal_outer_reshape
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %arg0 : (tensor<2x64x128xf8E4M3FN
  // CHECK: %[[RESHAPE:.*]] = ttg.memdesc_reshape %[[ALLOC]]
  // CHECK: tt.return %[[RESHAPE]]
  tt.func @legal_outer_reshape(%x: tensor<2x64x128xf8E4M3FN, #src>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory> {
    %r = tt.reshape %x : tensor<2x64x128xf8E4M3FN, #src> -> tensor<128x128xf8E4M3FN, #dst>
    %a = ttg.local_alloc %r : (tensor<128x128xf8E4M3FN, #dst>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>
    tt.return %a : !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>
  }
}
