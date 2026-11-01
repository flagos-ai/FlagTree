// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4 enable-source-remat=false' | FileCheck %s

// A size-one dimension still carries duplicated register/lane coordinates.
// Reshape must preserve that view before converting to a consumer's layout.
#source = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#consumer = #ttg.slice<{dim = 1, parent = #source}>

module {
  // CHECK-LABEL: tt.func @reshape_to_explicit_consumer
  tt.func @reshape_to_explicit_consumer(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %ip = tt.splat %input : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %value = tt.load %ip : tensor<64x1x!tt.ptr<f32>>
    %source = tle.gpu.set_layout %value {target_encoding = #source} : tensor<64x1xf32> -> tensor<64x1xf32>
    // CHECK: %[[VIEW:.*]] = tt.reshape {{.*}} : tensor<64x1xf32, {{.*}}> -> tensor<64xf32, {{.*}}>
    %view = tt.reshape %source : tensor<64x1xf32> -> tensor<64xf32>
    // CHECK: %[[CONVERTED:.*]] = ttg.convert_layout %[[VIEW]]
    %consumer = tle.gpu.set_layout %view {target_encoding = #consumer} : tensor<64xf32> -> tensor<64xf32>
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    // CHECK: tt.store {{.*}}, %[[CONVERTED]]
    tt.store %op, %consumer : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
