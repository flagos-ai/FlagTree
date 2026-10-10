// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=false" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=15" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, tle.enable_encoding_rematerialization} {
  // CHECK-LABEL: tt.func @sqmma_store(
  // CHECK: %[[VALUE:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #blocked}
  // CHECK: tt.store {{.*}}, %[[VALUE]], {{.*}} : tensor<16x64x!tt.ptr<f32>, #blocked>
  tt.func @sqmma_store(%acc: tensor<16x64xf32, #sqmma>, %ptrs: tensor<16x64x!tt.ptr<f32>, #blocked>, %mask: tensor<16x64xi1, #blocked>) {
    %value = ttg.convert_layout %acc {"tle.explicit_encoding.0" = #blocked} : tensor<16x64xf32, #sqmma> -> tensor<16x64xf32, #blocked>
    tt.store %ptrs, %value, %mask : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
 // CHECK-LABEL: tt.func @volatile_duplicate(
 // CHECK: tt.load {{.*}} {isVolatile = true}
 // CHECK-NOT: tt.load
 // CHECK: tt.return
 tt.func @volatile_duplicate(%base: !tt.ptr<f32>) -> (tensor<64xf32, #src>, tensor<64xf32, #dst>) {
  %i = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #src>
  %b = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #src>
  %p = tt.addptr %b, %i : tensor<64x!tt.ptr<f32>, #src>, tensor<64xi32, #src>
  %v = tt.load %p {isVolatile = true} : tensor<64x!tt.ptr<f32>, #src>
  %c = ttg.convert_layout %v : tensor<64xf32, #src> -> tensor<64xf32, #dst>
  tt.return %v, %c : tensor<64xf32, #src>, tensor<64xf32, #dst>
 }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
 // CHECK-LABEL: tt.func @repeated_address_remat(
 // CHECK: tt.load
 // CHECK: tt.load
 // CHECK-NOT: ttg.convert_layout
 // CHECK: tt.return
 tt.func @repeated_address_remat(%base: !tt.ptr<f32>) -> (tensor<256xf32, #src>, tensor<256xf32, #dst>) {
  %b = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #src>
  %v = tt.load %b : tensor<256x!tt.ptr<f32>, #src>
  %c = ttg.convert_layout %v : tensor<256xf32, #src> -> tensor<256xf32, #dst>
  tt.return %v, %c : tensor<256xf32, #src>, tensor<256xf32, #dst>
 }
}
