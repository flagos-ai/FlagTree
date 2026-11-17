// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

// A consumer's explicit matrix layout must be materialized after a reduction.
// The reduction itself still returns the slice layout of its rank-three input.
#sink = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#other = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#producer = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func @reduce_to_explicit_matrix
  tt.func @reduce_to_explicit_matrix(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<4x4x128x!tt.ptr<f32>>
    %values = tt.load %base : tensor<4x4x128x!tt.ptr<f32>>
    // CHECK: %[[REDUCED:.*]] = "tt.reduce"
    %reduced = "tt.reduce"(%values) <{axis = 2 : i32}> ({
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x4x128xf32>) -> tensor<4x4xf32>
    // CHECK: }) {{.*}} : (tensor<4x4x128xf32, {{.*}}>) -> tensor<4x4xf32, #ttg.slice<{dim = 2, parent = {{.*}}}>>
    // CHECK: %[[CONVERTED:.*]] = ttg.convert_layout %[[REDUCED]]
    %hint = tle.gpu.set_layout %reduced {target_encoding = #sink} : tensor<4x4xf32> -> tensor<4x4xf32>
    %out = tt.splat %output : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    // CHECK: tt.store %{{.*}}, %[[CONVERTED]]
    // CHECK-NOT: builtin.unrealized_conversion_cast
    tt.store %out, %hint : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }

  // Each result needs its own conversion, not one encoding for the entire op.
  // CHECK-LABEL: tt.func @reduce_to_two_explicit_layouts
  tt.func @reduce_to_two_explicit_layouts(%input: !tt.ptr<f32>, %sum_out: !tt.ptr<f32>, %max_out: !tt.ptr<f32>) {
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<4x4x128x!tt.ptr<f32>>
    %values = tt.load %base : tensor<4x4x128x!tt.ptr<f32>>
    // CHECK: %[[PAIR:.*]]:2 = "tt.reduce"
    %pair:2 = "tt.reduce"(%values, %values) <{axis = 2 : i32}> ({
    ^bb0(%sum_left: f32, %max_left: f32, %sum_right: f32, %max_right: f32):
      %sum = arith.addf %sum_left, %sum_right : f32
      %max = arith.maximumf %max_left, %max_right : f32
      tt.reduce.return %sum, %max : f32, f32
    }) : (tensor<4x4x128xf32>, tensor<4x4x128xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    // CHECK: %[[SUM:.*]] = ttg.convert_layout %[[PAIR]]#0
    // CHECK: %[[MAX:.*]] = ttg.convert_layout %[[PAIR]]#1
    %sum_hint = tle.gpu.set_layout %pair#0 {target_encoding = #sink} : tensor<4x4xf32> -> tensor<4x4xf32>
    %max_hint = tle.gpu.set_layout %pair#1 {target_encoding = #other} : tensor<4x4xf32> -> tensor<4x4xf32>
    %sum_ptr = tt.splat %sum_out : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %max_ptr = tt.splat %max_out : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    // CHECK: tt.store %{{.*}}, %[[SUM]]
    // CHECK: tt.store %{{.*}}, %[[MAX]]
    tt.store %sum_ptr, %sum_hint : tensor<4x4x!tt.ptr<f32>>
    tt.store %max_ptr, %max_hint : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }

  // Pre-encoded operands must not make an incompatible hinted result legal.
  // CHECK-LABEL: tt.func @preencoded_reduction
  tt.func @preencoded_reduction(%input: tensor<4x4x128xf32, #producer>, %output: !tt.ptr<f32>) {
    // CHECK: %[[REDUCED:.*]] = "tt.reduce"
    %reduced = "tt.reduce"(%input) <{axis = 2 : i32}> ({
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x4x128xf32, #producer>) -> tensor<4x4xf32, #ttg.slice<{dim = 2, parent = #producer}>>
    // CHECK: %[[CONVERTED:.*]] = ttg.convert_layout %[[REDUCED]]
    %hint = tle.gpu.set_layout %reduced {target_encoding = #sink} : tensor<4x4xf32, #ttg.slice<{dim = 2, parent = #producer}>> -> tensor<4x4xf32, #ttg.slice<{dim = 2, parent = #producer}>>
    %out = tt.splat %output : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>, #ttg.slice<{dim = 2, parent = #producer}>>
    // CHECK: tt.store %{{.*}}, %[[CONVERTED]]
    tt.store %out, %hint : tensor<4x4x!tt.ptr<f32>, #ttg.slice<{dim = 2, parent = #producer}>>
    tt.return
  }
}
