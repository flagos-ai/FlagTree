// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

// Consecutive static extracts used to claim the same scratch allocation even
// though lowering only selects registers. MembarAnalysis then inserted a CTA
// barrier between the extracts. Check both scratch and the resulting code.
#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @extract_static_rows_no_barrier
  // CHECK-NOT: nvvm.barrier
  // CHECK-NOT: !llvm.ptr<3>
  // CHECK-NOT: ld.shared
  // CHECK-NOT: st.shared
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK-NOT: nvvm.barrier
  // CHECK-NOT: !llvm.ptr<3>
  // CHECK-NOT: ld.shared
  // CHECK-NOT: st.shared
  // CHECK: llvm.return
  tt.func @extract_static_rows_no_barrier(%src: tensor<16x128xf32, #row>, %out: tensor<1x128x!tt.ptr<f32>, #row>) {
    %c0 = arith.constant 0 : i32
    %c15 = arith.constant 15 : i32
    %first = tle.extract_tile %src[%c0] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    %last = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    %sum = arith.addf %first, %last : tensor<1x128xf32, #row>
    tt.store %out, %sum : tensor<1x128x!tt.ptr<f32>, #row>
    tt.return
  }
}

// -----

// A runtime index still requires the shared-memory relay and its barriers.
#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @extract_dynamic_row_keeps_shared
  // CHECK: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.load {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK: llvm.return
  tt.func @extract_dynamic_row_keeps_shared(%src: tensor<16x128xf32, #row>, %idx: i32, %out: tensor<1x128x!tt.ptr<f32>, #row>) {
    %tile = tle.extract_tile %src[%idx] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    tt.store %out, %tile : tensor<1x128x!tt.ptr<f32>, #row>
    tt.return
  }
}

// -----

#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @extract_static_unaligned_row_keeps_shared
  // CHECK: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.load {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK: llvm.return
  tt.func @extract_static_unaligned_row_keeps_shared(%src: tensor<16x128xf32, #rows>, %out: tensor<1x128x!tt.ptr<f32>, #rows>) {
    %c15 = arith.constant 15 : i32
    %tile = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #rows>, i32 -> tensor<1x128xf32, #rows>
    tt.store %out, %tile : tensor<1x128x!tt.ptr<f32>, #rows>
    tt.return
  }
}

// -----

#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @extract_static_changed_lane_layout_keeps_shared
  // CHECK: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.load {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK: llvm.return
  tt.func @extract_static_changed_lane_layout_keeps_shared(%src: tensor<16x128xf32, #row>, %out: tensor<1x128x!tt.ptr<f32>, #rows>) {
    %c15 = arith.constant 15 : i32
    %tile = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #rows>
    tt.store %out, %tile : tensor<1x128x!tt.ptr<f32>, #rows>
    tt.return
  }
}
