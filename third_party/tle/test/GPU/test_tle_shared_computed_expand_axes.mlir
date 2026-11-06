// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s
// A computed index may be shared by both matrix axes. ExpandDims must
// convert each use to its required slice while preserving the matrix hint.
// CHECK-LABEL: tt.func @shared_computed_axes
// CHECK: arith.addi
// CHECK: ttg.convert_layout
// CHECK: tt.expand_dims
// CHECK: tt.expand_dims
// CHECK: tt.store
#matrix = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
module {
 tt.func @shared_computed_axes(%offset: i32, %out: !tt.ptr<i32>) {
  %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %splat = tt.splat %offset : i32 -> tensor<4xi32>
  %index = arith.addi %range, %splat : tensor<4xi32>
  %rows = tt.expand_dims %index {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cols = tt.expand_dims %index {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rb = tt.broadcast %rows : tensor<4x1xi32> -> tensor<4x4xi32>
  %cb = tt.broadcast %cols : tensor<1x4xi32> -> tensor<4x4xi32>
  %zero = arith.constant dense<0> : tensor<4x4xi32>
  %hint = tle.gpu.set_layout %zero {target_encoding = #matrix} : tensor<4x4xi32> -> tensor<4x4xi32>
  %sum = arith.addi %rb, %cb : tensor<4x4xi32>
  %value = arith.addi %sum, %hint : tensor<4x4xi32>
  %ptr = tt.splat %out : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
  tt.store %ptr, %value : tensor<4x4x!tt.ptr<i32>>
  tt.return
 }
}
