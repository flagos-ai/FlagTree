// RUN: triton-opt %s -split-input-file --tritongpu-optimize-dot-operands | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_b_from_existing_smem
  tt.func @reuse_transposed_wgmma_b_from_existing_smem(
      %a: tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
      %b_init: tensor<64x512xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc
    %b_smem = ttg.local_alloc %b_init : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_smem : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #blocked>
    %b_t = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x512xbf16, #blocked> -> tensor<512x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SMEM]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<512x64xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_alloc = ttg.local_alloc %b_t : (tensor<512x64xbf16, #blocked1>) -> !ttg.memdesc<512x64xbf16, #shared, #smem>
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<512x64xbf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_wgmma_a_from_existing_smem
  tt.func @reuse_wgmma_a_from_existing_smem(
      %a_init: tensor<64x512xbf16, #blocked>,
      %b: !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc
    %a_smem = ttg.local_alloc %a_init : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %a = ttg.local_load %a_smem : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    // CHECK-NOT: ttg.local_load %[[A_SMEM]]
    // CHECK: ttng.warp_group_dot %[[A_SMEM]], %arg1, {{.*}} {
    // CHECK-SAME: tle.explicit_wgmma_commit
    // CHECK-SAME: } : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
    %out = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, tle.explicit_wgmma_commit} : tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_b_from_indexed_smem
  tt.func @reuse_transposed_wgmma_b_from_indexed_smem(
      %a: tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_slot : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #blocked>
    %b_t = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x512xbf16, #blocked> -> tensor<512x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SLOT]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<512x64xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_alloc = ttg.local_alloc %b_t : (tensor<512x64xbf16, #blocked1>) -> !ttg.memdesc<512x64xbf16, #shared, #smem>
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<512x64xbf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_wgmma_b_alloc_from_indexed_smem
  tt.func @reuse_wgmma_b_alloc_from_indexed_smem(
      %a: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x256xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>

    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<64x256xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[B_SLOT]], {{.*}}
    %b_alloc = ttg.local_alloc %b : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x256xbf16, #shared, #smem> -> tensor<64x256xf32, #mma>
    tt.return %out : tensor<64x256xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_b_from_subsliced_smem_preserves_alloc_shape
  tt.func @reuse_transposed_wgmma_b_from_subsliced_smem_preserves_alloc_shape(
      %a: tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable>
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SUB:.+]] = ttg.memdesc_subslice %{{.*}}[0, 256] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512>
    %b_sub = ttg.memdesc_subslice %b_slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512>
    %b = ttg.local_load %b_sub : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512> -> tensor<64x256xbf16, #blocked>
    %b_t = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x256xbf16, #blocked> -> tensor<256x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SUB]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable, 512x64>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<256x64xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_alloc = ttg.local_alloc %b_t : (tensor<256x64xbf16, #blocked1>) -> !ttg.memdesc<256x64xbf16, #shared1, #smem>
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32} : tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<256x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_canonicalized_transposed_wgmma_b_from_subsliced_smem_preserves_alloc_shape
  tt.func @reuse_canonicalized_transposed_wgmma_b_from_subsliced_smem_preserves_alloc_shape(
      %a: tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable>
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SUB:.+]] = ttg.memdesc_subslice %{{.*}}[0, 256] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512>
    %b_sub = ttg.memdesc_subslice %b_slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512>
    %b = ttg.local_load %b_sub : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512> -> tensor<64x256xbf16, #blocked>
    %b_alloc = ttg.local_alloc %b : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SUB]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable, 512x64>
    // CHECK-NOT: ttg.memdesc_trans
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_trans = ttg.memdesc_trans %b_alloc {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem> -> !ttg.memdesc<256x64xbf16, #shared1, #smem>
    %out = ttng.warp_group_dot %a, %b_trans, %acc {inputPrecision = 0 : i32} : tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<256x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_canonicalized_transposed_wgmma_b_from_transposed_staging_alloc
  tt.func @reuse_canonicalized_transposed_wgmma_b_from_transposed_staging_alloc(
      %a: tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>
    %b_alloc = ttg.local_alloc %b : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared1, #smem>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SLOT]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.memdesc_trans
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_trans = ttg.memdesc_trans %b_alloc {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared1, #smem> -> !ttg.memdesc<256x64xbf16, #shared, #smem>
    %out = ttng.warp_group_dot %a, %b_trans, %acc {inputPrecision = 0 : i32} : tensor<64x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<256x64xbf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_a_from_indexed_smem_with_wait
  tt.func @reuse_transposed_wgmma_a_from_indexed_smem_with_wait(
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>) -> tensor<256x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>

    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[A_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %a_slot = ttg.memdesc_index %a_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %a = ttg.local_load %a_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>
    %a_t = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x256xbf16, #blocked> -> tensor<256x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[A_SLOT]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<256x64xbf16
    // CHECK: %[[DOT:.+]] = ttng.warp_group_dot %[[VIEW]], %arg0, {{.*}} {
    // CHECK-SAME: tle.explicit_wgmma_commit
    // CHECK-SAME: }
    %a_alloc = ttg.local_alloc %a_t : (tensor<256x64xbf16, #blocked1>) -> !ttg.memdesc<256x64xbf16, #shared1, #smem>
    %out = ttng.warp_group_dot %a_alloc, %b, %acc {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit} : !ttg.memdesc<256x64xbf16, #shared1, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<256x64xf32, #mma>

    // CHECK: ttng.warp_group_dot_wait %[[DOT]], %[[VIEW]] {pendings = 0 : i32}
    %wait:2 = ttng.warp_group_dot_wait %out, %a_alloc {pendings = 0 : i32} : tensor<256x64xf32, #mma>, !ttg.memdesc<256x64xbf16, #shared1, #smem>
    tt.return %wait#0 : tensor<256x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_a_register_load_from_indexed_smem
  tt.func @reuse_transposed_wgmma_a_register_load_from_indexed_smem(
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>) -> tensor<256x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>

    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[A_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %a_slot = ttg.memdesc_index %a_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %a = ttg.local_load %a_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>
    %a_t = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x256xbf16, #blocked> -> tensor<256x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[A_SLOT]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<256x64xbf16
    // CHECK-NOT: ttg.local_load {{.*}} -> tensor<256x64xbf16, #ttg.dot_op
    // CHECK: ttng.warp_group_dot %[[VIEW]], %arg0, {{.*}}
    %a_alloc = ttg.local_alloc %a_t : (tensor<256x64xbf16, #blocked1>) -> !ttg.memdesc<256x64xbf16, #shared1, #smem>
    %a_dot = ttg.local_load %a_alloc : !ttg.memdesc<256x64xbf16, #shared1, #smem> -> tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %out = ttng.warp_group_dot %a_dot, %b, %acc {inputPrecision = 0 : i32} : tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<256x64xf32, #mma>
    tt.return %out : tensor<256x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_a_convert_from_indexed_smem
  tt.func @reuse_transposed_wgmma_a_convert_from_indexed_smem(
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>) -> tensor<256x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>

    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[A_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %a_slot = ttg.memdesc_index %a_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %a = ttg.local_load %a_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>
    %a_t = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x256xbf16, #blocked> -> tensor<256x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[A_SLOT]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.convert_layout {{.*}} -> tensor<256x64xbf16, #ttg.dot_op
    // CHECK: ttng.warp_group_dot %[[VIEW]], %arg0, {{.*}}
    %a_dot = ttg.convert_layout %a_t : tensor<256x64xbf16, #blocked1> -> tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %out = ttng.warp_group_dot %a_dot, %b, %acc {inputPrecision = 0 : i32} : tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<256x64xf32, #mma>
    tt.return %out : tensor<256x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_wgmma_b_alloc_from_for_if_carried_smem
  tt.func @reuse_wgmma_b_alloc_from_for_if_carried_smem(
      %a: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x256xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable>
    %slot0 = ttg.memdesc_index %b_smem[%c0_i32] : !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %res:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%slot_arg = %slot0, %acc_arg = %acc) -> (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x256xf32, #mma>) {
      // CHECK: %[[B_SUB:.+]] = ttg.memdesc_subslice {{.*}}[0, 0]
      %b_sub = ttg.memdesc_subslice %slot_arg[0, 0] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512>
      %b = ttg.local_load %b_sub : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable, 64x512> -> tensor<64x256xbf16, #blocked>

      // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<64x256xbf16
      // CHECK: ttng.warp_group_dot {{.*}}, %[[B_SUB]], {{.*}}
      %b_alloc = ttg.local_alloc %b : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>
      %out = ttng.warp_group_dot %a, %b_alloc, %acc_arg {inputPrecision = 0 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x256xbf16, #shared, #smem> -> tensor<64x256xf32, #mma>

      %slot1 = ttg.memdesc_index %b_smem[%c1_i32] : !ttg.memdesc<2x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      %cond = arith.cmpi slt, %iv, %c1 : index
      %next_slot = scf.if %cond -> (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) {
        scf.yield %slot1 : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      } else {
        scf.yield %slot_arg : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      }
      scf.yield %next_slot, %out : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x256xf32, #mma>
    }
    tt.return %res#1 : tensor<64x256xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reuse_wgmma_b_alloc_merges_redundant_waits
  tt.func @reuse_wgmma_b_alloc_merges_redundant_waits(
      %a: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x256xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #mma>

    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable>
    // CHECK: %[[B_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    %b_slot = ttg.memdesc_index %b_smem[%c0] : !ttg.memdesc<2x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_slot : !ttg.memdesc<64x256xbf16, #shared, #smem, mutable> -> tensor<64x256xbf16, #blocked>

    %b_alloc = ttg.local_alloc %b : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>
    // CHECK: %[[DOT:.+]] = ttng.warp_group_dot {{.*}}, %[[B_SLOT]], {{.*}}
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32, isAsync = true} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x256xbf16, #shared, #smem> -> tensor<64x256xf32, #mma>
    // CHECK-NEXT: %[[WAIT:.+]]:2 = ttng.warp_group_dot_wait %[[DOT]], %[[B_SLOT]]
    // CHECK-NOT: ttng.warp_group_dot_wait
    // CHECK: tt.return %[[WAIT]]#0
    %wait0:2 = ttng.warp_group_dot_wait %out, %b_alloc {pendings = 0 : i32} : tensor<64x256xf32, #mma>, !ttg.memdesc<64x256xbf16, #shared, #smem>
    %wait1:2 = ttng.warp_group_dot_wait %out, %b_alloc {pendings = 0 : i32} : tensor<64x256xf32, #mma>, !ttg.memdesc<64x256xbf16, #shared, #smem>
    tt.return %wait0#0 : tensor<64x256xf32, #mma>
  }
}
