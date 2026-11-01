// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8 enable-source-remat=false' | FileCheck %s

// An explicit MMA or dot-operand source is not a blocked encoding. Expanding
// its rank needs a compatible slice for this use without changing the source.
#mma2 = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#mma3 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
#operand = #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 2}>
// CHECK: #[[$MMA2:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
// CHECK: #[[$MMA3:.*]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>

module {
  // CHECK-LABEL: tt.func @mma2_expand_leading_axis
  tt.func @mma2_expand_leading_axis(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %p = tt.splat %input : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
    %a = tt.load %p : tensor<16x64x!tt.ptr<f32>>
    %value = tle.gpu.set_layout %a {target_encoding = #mma2} : tensor<16x64xf32> -> tensor<16x64xf32>
    // CHECK: %[[SOURCE:.*]] = ttg.convert_layout %{{.*}} : tensor<16x64xf32, #[[$MMA2]]> -> tensor<16x64xf32, #ttg.slice<{dim = 0, parent = {{.*}}}>>
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SOURCE]] {axis = 0 : i32{{.*}}} : tensor<16x64xf32, {{.*}}> -> tensor<1x16x64xf32, {{.*}}>
    %expanded = tt.expand_dims %value {axis = 0 : i32} : tensor<16x64xf32> -> tensor<1x16x64xf32>
    %pointers = tt.splat %output : !tt.ptr<f32> -> tensor<1x16x64x!tt.ptr<f32>>
    // CHECK: %[[OUTPUT:.*]] = ttg.convert_layout %[[EXPANDED]]
    // CHECK: tt.store {{.*}}, %[[OUTPUT]]
    tt.store %pointers, %expanded : tensor<1x16x64x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func @mma3_expand_trailing_axis
  tt.func @mma3_expand_trailing_axis(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %p = tt.splat %input : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %a = tt.load %p : tensor<64x64x!tt.ptr<f32>>
    %value = tle.gpu.set_layout %a {target_encoding = #mma3} : tensor<64x64xf32> -> tensor<64x64xf32>
    // CHECK: %[[SOURCE:.*]] = ttg.convert_layout %{{.*}} : tensor<64x64xf32, #[[$MMA3]]> -> tensor<64x64xf32, #ttg.slice<{dim = 2, parent = {{.*}}}>>
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SOURCE]] {axis = 2 : i32{{.*}}} : tensor<64x64xf32, {{.*}}> -> tensor<64x64x1xf32, {{.*}}>
    %expanded = tt.expand_dims %value {axis = 2 : i32} : tensor<64x64xf32> -> tensor<64x64x1xf32>
    %pointers = tt.splat %output : !tt.ptr<f32> -> tensor<64x64x1x!tt.ptr<f32>>
    // CHECK: %[[OUTPUT:.*]] = ttg.convert_layout %[[EXPANDED]]
    // CHECK: tt.store {{.*}}, %[[OUTPUT]]
    tt.store %pointers, %expanded : tensor<64x64x1x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_operand_expand_middle_axis
  tt.func @dot_operand_expand_middle_axis(%input: !tt.ptr<bf16>, %output: !tt.ptr<bf16>) {
    %p = tt.splat %input : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
    %a = tt.load %p : tensor<16x64x!tt.ptr<bf16>>
    %value = tle.gpu.set_layout %a {target_encoding = #operand} : tensor<16x64xbf16> -> tensor<16x64xbf16>
    // CHECK: %[[SOURCE:.*]] = ttg.convert_layout %{{.*}} : tensor<16x64xbf16, #ttg.dot_op<{{.*}}>> -> tensor<16x64xbf16, #ttg.slice<{dim = 1, parent = {{.*}}}>>
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SOURCE]] {axis = 1 : i32{{.*}}} : tensor<16x64xbf16, {{.*}}> -> tensor<16x1x64xbf16, {{.*}}>
    %expanded = tt.expand_dims %value {axis = 1 : i32} : tensor<16x64xbf16> -> tensor<16x1x64xbf16>
    %pointers = tt.splat %output : !tt.ptr<bf16> -> tensor<16x1x64x!tt.ptr<bf16>>
    // CHECK: %[[OUTPUT:.*]] = ttg.convert_layout %[[EXPANDED]]
    // CHECK: tt.store {{.*}}, %[[OUTPUT]]
    tt.store %pointers, %expanded : tensor<16x1x64x!tt.ptr<bf16>>
    tt.return
  }
}
