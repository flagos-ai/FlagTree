// RUN: triton-opt %s -triton-nvidia-gpu-fence-insertion=compute-capability=90 --canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared_in = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_out = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_async_copy_wgmma_fence
  tt.func @preserve_async_copy_wgmma_fence(
      %a: tensor<64x64xbf16, #dot>,
      %src: tensor<64x64x!tt.ptr<bf16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %base = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable>
    %slot = ttg.memdesc_index %base[%c0] : !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable>
    %tok = ttg.async_copy_global_to_local %src, %slot : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared_in, #smem, mutable>
    %tok2 = ttg.async_commit_group tokens %tok
    %view = tle.memdesc_wgmma_view %slot {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable>
    // CHECK: tle.wgmma_shared_operand_fence
    // CHECK-NEXT: ttng.warp_group_dot
    %out = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32} : tensor<64x64xbf16, #dot> * !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
