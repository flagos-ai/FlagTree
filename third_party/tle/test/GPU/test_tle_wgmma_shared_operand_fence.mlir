// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion=compute-capability=90 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @insert_for_wgmma_shared_operand
  tt.func @insert_for_wgmma_shared_operand(
      %a: tensor<64x512xbf16, #dot>,
      %b_init: tensor<512x64xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc
    %b_smem = ttg.local_alloc %b_init : (tensor<512x64xbf16, #blocked>) -> !ttg.memdesc<512x64xbf16, #shared, #smem, mutable>

    // CHECK-NEXT: tle.wgmma_shared_operand_fence %[[B_SMEM]]
    // CHECK-NEXT: ttng.warp_group_dot %arg0, %[[B_SMEM]], {{.*}}
    %out = ttng.warp_group_dot %a, %b_smem, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #dot> * !ttg.memdesc<512x64xbf16, #shared, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>
#shared_in = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_out = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @insert_for_async_copy_wgmma_view
  tt.func @insert_for_async_copy_wgmma_view(
      %a: tensor<64x64xbf16, #dot1>,
      %src: tensor<64x64x!tt.ptr<bf16>, #blocked1>) -> tensor<64x64xf32, #mma1> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma1>
    %base = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable>
    %slot = ttg.memdesc_index %base[%c0] : !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable>
    %tok = ttg.async_copy_global_to_local %src, %slot : tensor<64x64x!tt.ptr<bf16>, #blocked1> -> <64x64xbf16, #shared_in, #smem, mutable>
    %tok2 = ttg.async_commit_group tokens %tok
    %view = tle.memdesc_wgmma_view %slot {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view
    // CHECK-NEXT: tle.wgmma_shared_operand_fence %[[VIEW]]
    // CHECK-NEXT: ttng.warp_group_dot %arg0, %[[VIEW]], {{.*}}
    %out = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32} : tensor<64x64xbf16, #dot1> * !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable> -> tensor<64x64xf32, #mma1>
    tt.return %out : tensor<64x64xf32, #mma1>
  }
}
