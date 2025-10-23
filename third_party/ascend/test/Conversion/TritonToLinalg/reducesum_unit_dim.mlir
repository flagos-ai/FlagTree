// RUN: triton-adapter-opt --triton-to-linalg %s --split-input-file | FileCheck %s -check-prefixes=CHECK
module {
  // CHECK-LABEL: @triton_addptr_f32
  tt.func public @triton_addptr_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_1 = arith.constant dense<256> : tensor<256xi32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = arith.cmpi slt, %0, %cst_1 : tensor<256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<256x!tt.ptr<f32, 1>>
    %3 = tt.addptr %2, %0 : tensor<256x!tt.ptr<f32, 1>>, tensor<256xi32>
    %4 = tt.load %3, %1, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>>
    %5 = arith.select %1, %4, %cst_0 : tensor<256xi1>, tensor<256xf32>
    %6 = "tt.reduce"(%5) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %11 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %11 : f32
    }) : (tensor<256xf32>) -> f32
    %7 = arith.addf %6, %cst : f32
    //CHECK: %[[TENSOR:.*]] = tensor.empty() : tensor<1xf32>
    //CHECK: %[[RES:.*]] = linalg.fill ins(%[[VAL:.*]] : f32) outs(%[[TENSOR]] : tensor<1xf32>) -> tensor<1xf32>
    %8 = tt.splat %7 : f32 -> tensor<1xf32>
    //CHECK: %[[OUTPTR:.*]] = memref.reinterpret_cast %[[ARG_3:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    %9 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f32, 1>, i32
    %10 = tt.splat %9 : !tt.ptr<f32, 1> -> tensor<1x!tt.ptr<f32, 1>>
    //CHECK: bufferization.materialize_in_destination %[[VAL_1:.*]] in writable %[[OUTPTR]] : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    tt.store %10, %8 {cache = 1 : i32, evict = 1 : i32} : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: @triton_addptr_1x1xf32
  tt.func public @triton_addptr_1x1xf32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x256xf32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %3 = scf.for %arg3 = %c0_i32 to %c4096_i32 step %c256_i32 iter_args(%arg4 = %cst) -> (tensor<1x256xf32>)  : i32 {
      %8 = tt.splat %arg3 : i32 -> tensor<1x256xi32>
      %9 = arith.addi %8, %1 : tensor<1x256xi32>
      %10 = tt.addptr %2, %9 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
      %11 = tt.load %10 evictionPolicy = evict_first : tensor<1x256x!tt.ptr<f32>>
      %12 = arith.addf %arg4, %11 : tensor<1x256xf32>
      scf.yield %12 : tensor<1x256xf32>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %8 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %8 : f32
    }) : (tensor<1x256xf32>) -> tensor<1xf32>
    //CHECK: %[[RES:.*]] = tensor.expand_shape %[[VAL:.*]] {{\[\[}}0, 1]] output_shape {{\[}}1, 1] : tensor<1xf32> into tensor<1x1xf32>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    //CHECK: %[[OUTPTR:.*]] = memref.reinterpret_cast %[[ARG_1:.*]] to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1]>>
    %6 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f32>, i32
    %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
    //CHECK: bufferization.materialize_in_destination %[[RES]] in writable %[[OUTPTR]] : (tensor<1x1xf32>, memref<1x1xf32, strided<[1, 1]>>) -> ()
    tt.store %7, %5 : tensor<1x1x!tt.ptr<f32>>
    tt.return
  }
}