// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @wgmma_wait_memdesc_result_alias
  // CHECK: %[[ALLOC:.+]] = ttg.local_alloc
  // CHECK: %[[SLOT:.+]] = ttg.memdesc_index %[[ALLOC]]
  // CHECK: %[[WAIT:.+]] = ttng.warp_group_dot_wait %[[SLOT]]
  // CHECK: ttg.local_load %[[WAIT]]
  tt.func @wgmma_wait_memdesc_result_alias() {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %wait = ttng.warp_group_dot_wait %slot {pendings = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %value = ttg.local_load %wait : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @select_memdesc_result_alias
  // CHECK: %[[SLOT0:.+]] = ttg.memdesc_index
  // CHECK: %[[SLOT1:.+]] = ttg.memdesc_index
  // CHECK: %[[SEL:.+]] = arith.select %{{.+}}, %[[SLOT0]], %[[SLOT1]]
  // CHECK: ttg.local_load %[[SEL]]
  tt.func @select_memdesc_result_alias(%pred: i1) {
    %c0 = arith.constant 0 : i32
    %alloc0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable>
    %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable>
    %slot0 = ttg.memdesc_index %alloc0[%c0] : !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %slot1 = ttg.memdesc_index %alloc1[%c0] : !ttg.memdesc<1x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %selected = arith.select %pred, %slot0, %slot1 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %value = ttg.local_load %selected : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    tt.return
  }
}
