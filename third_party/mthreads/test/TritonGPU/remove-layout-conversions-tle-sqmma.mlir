// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=false" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true" | FileCheck %s
// The established TLE SQMMA writeback adapter is independent of RLC enhancement.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, tle.enable_encoding_rematerialization} {
  // CHECK-LABEL: tt.func @sqmma_store(
  // CHECK-SAME: %[[ACC:[a-zA-Z0-9_]+]]: tensor<16x64xf32, #{{[^>]+}}>
  // CHECK-NOT: ttg.convert_layout %[[ACC]]
  // CHECK: tt.store {{.*}}, %[[ACC]], {{.*}} : tensor<16x64x!tt.ptr<f32>, #{{[^>]+}}>
  tt.func @sqmma_store(%acc: tensor<16x64xf32, #sqmma>, %ptrs: tensor<16x64x!tt.ptr<f32>, #blocked>, %mask: tensor<16x64xi1, #blocked>) {
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #sqmma> -> tensor<16x64xf32, #blocked>
    tt.store %ptrs, %value, %mask : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
