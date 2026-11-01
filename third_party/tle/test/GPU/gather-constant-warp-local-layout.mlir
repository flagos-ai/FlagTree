// RUN: triton-opt %s -tritongpu-optimize-thread-locality | FileCheck %s

// Uniform indices select one column per row. An already warp-local MMA gather
// needs no Shared layout round-trip merely to minimize dynamic-index shuffles.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 8, 32]}>
#cross = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 16, 32]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @uniform_warp_local
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.gather {{.*}}efficient_layout{{.*}} -> tensor<64x1xf32, #mma>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func @uniform_warp_local(%src: tensor<64x8xf32, #mma>) -> tensor<64x1xf32, #mma> {
    %index = arith.constant dense<0> : tensor<64x1xi32, #mma>
    %value = tt.gather %src[%index] {axis = 1 : i32} : (tensor<64x8xf32, #mma>, tensor<64x1xi32, #mma>) -> tensor<64x1xf32, #mma>
    tt.return %value : tensor<64x1xf32, #mma>
  }

  // Dynamic indices retain the ordinary locality optimization.
  // CHECK-LABEL: tt.func @dynamic_indices
  // CHECK: ttg.convert_layout
  // CHECK: tt.gather {{.*}}efficient_layout
  tt.func @dynamic_indices(%src: tensor<64x8xf32, #mma>, %index: tensor<64x1xi32, #mma>) -> tensor<64x1xf32, #mma> {
    %value = tt.gather %src[%index] {axis = 1 : i32} : (tensor<64x8xf32, #mma>, tensor<64x1xi32, #mma>) -> tensor<64x1xf32, #mma>
    tt.return %value : tensor<64x1xf32, #mma>
  }

  // A uniform index cannot make a cross-warp source warp-local.
  // CHECK-LABEL: tt.func @cross_warp_source
  // CHECK: ttg.convert_layout
  // CHECK: tt.gather {{.*}}efficient_layout
  tt.func @cross_warp_source(%src: tensor<64x32xf32, #cross>) -> tensor<64x1xf32, #cross> {
    %index = arith.constant dense<0> : tensor<64x1xi32, #cross>
    %value = tt.gather %src[%index] {axis = 1 : i32} : (tensor<64x32xf32, #cross>, tensor<64x1xi32, #cross>) -> tensor<64x1xf32, #cross>
    tt.return %value : tensor<64x1xf32, #cross>
  }
}
