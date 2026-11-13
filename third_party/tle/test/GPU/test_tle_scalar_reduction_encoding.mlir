// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

// A rank-one reduction returns a scalar, not a rank-zero tensor with a
// SliceEncoding. An explicit source layout must not be propagated past this
// tensor/scalar boundary, including a scalar re-broadcast to another layout.
#one = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#matrix = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#keys = #ttg.slice<{dim = 0, parent = #matrix}>

module {
  // CHECK-LABEL: tt.func @integer_count
  tt.func @integer_count(%out: !tt.ptr<i32>) {
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %hint = tle.gpu.set_layout %indices {target_encoding = #one} : tensor<128xi32> -> tensor<128xi32>
    // CHECK: %[[SUM:.*]] = "tt.reduce"
    %sum = "tt.reduce"(%hint) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %v = arith.addi %a, %b : i32
      tt.reduce.return %v : i32
    }) : (tensor<128xi32>) -> i32
    // CHECK: }) : (tensor<128xi32, {{.*}}>) -> i32
    // CHECK: tt.store %{{.*}}, %[[SUM]] : !tt.ptr<i32>
    tt.store %out, %sum : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: tt.func @slice_sum_and_broadcast
  tt.func @slice_sum_and_broadcast(%input: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %indices : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %values = tt.load %ptrs : tensor<128x!tt.ptr<f32>>
    %hint = tle.gpu.set_layout %values {target_encoding = #keys} : tensor<128xf32> -> tensor<128xf32>
    // CHECK: %[[SUM:.*]] = "tt.reduce"
    %sum = "tt.reduce"(%hint) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %v = arith.addf %a, %b : f32
      tt.reduce.return %v : f32
    }) : (tensor<128xf32>) -> f32
    // CHECK: }) : (tensor<128xf32, #ttg.slice<{{.*}}>>) -> f32
    // CHECK: tt.splat %[[SUM]] {{.*}} : f32 -> tensor<128xf32, #blocked{{[0-9]*}}>
    %broadcast = tt.splat %sum : f32 -> tensor<128xf32>
    %other = tle.gpu.set_layout %broadcast {target_encoding = #one} : tensor<128xf32> -> tensor<128xf32>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %indices : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %out_ptrs, %other : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  // Tensor-valued reductions must still propagate the requested parent layout.
  // CHECK-LABEL: tt.func @matrix_row_sum
  tt.func @matrix_row_sum(%input: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x128x!tt.ptr<f32>>
    %values = tt.load %base : tensor<8x128x!tt.ptr<f32>>
    %hint = tle.gpu.set_layout %values {target_encoding = #matrix} : tensor<8x128xf32> -> tensor<8x128xf32>
    // CHECK: %[[SUM:.*]] = "tt.reduce"
    %sum = "tt.reduce"(%hint) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %v = arith.addf %a, %b : f32
      tt.reduce.return %v : f32
    }) : (tensor<8x128xf32>) -> tensor<8xf32>
    // CHECK: }) {{.*}} : (tensor<8x128xf32, {{.*}}>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = {{.*}}}>>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    // CHECK: tt.store %{{.*}}, %[[SUM]]
    tt.store %out_ptrs, %sum : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
