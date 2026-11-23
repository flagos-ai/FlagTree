// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.shared = 8192 : i32
  tt.func @alias_does_not_allocate_new_smem(%arg0: tensor<64x64xbf16, #blocked>) {
    // CHECK: ttg.local_alloc
    // CHECK-SAME: allocation.offset = 0 : i32
    %src = ttg.local_alloc %arg0 : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: tle.memdesc_alias
    // CHECK-NOT: allocation.offset = 8192 : i32
    %alias = tle.memdesc_alias %src {offset_bytes = 0 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable>
    tt.return
  }
}
