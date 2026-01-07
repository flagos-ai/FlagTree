// RUN: triton-adapter-opt %s --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// dtype: bool
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<i8>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xi8>, tensor<4xi32>) -> tensor<4xi8> 
    %9 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xi8>, tensor<4xi32>, i32) -> tensor<4xi8>

// -----

// dtype: float16
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f16>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<f16>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xf16>, tensor<4xi32>) -> tensor<4xf16> 
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f16>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<f16>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xf16>, tensor<4xi32>, i32) -> tensor<4xf16>

// -----

// dtype: float32
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<f32>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xf32>, tensor<4xi32>) -> tensor<4xf32> 
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<f32>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xf32>, tensor<4xi32>, i32) -> tensor<4xf32>

// -----

// dtype: bfloat16
module {
  tt.func public @triton_func(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<bf16>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xbf16>, tensor<4xi32>) -> tensor<4xbf16> 
    %9 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<bf16>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xbf16>, tensor<4xi32>, i32) -> tensor<4xbf16>

// -----

// dtype: float8_e4m3
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x!tt.ptr<f8E4M3FN>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f8E4M3FN>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<f8E4M3FN>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xf8E4M3FN>, tensor<4xi32>) -> tensor<4xf8E4M3FN> 
    %9 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E4M3FN>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xf8E4M3FN>, tensor<4xi32>, i32) -> tensor<4xf8E4M3FN>

// -----

// dtype: float8_e5m2
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<8x!tt.ptr<f8E5M2>> 
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f8E5M2>>, tensor<8xi32> 
    %3 = tt.load %2 : tensor<8x!tt.ptr<f8E5M2>> 
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> 
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>> 
    %8 = tt.gather %3[%7] {axis = 0 : i32} : (tensor<8xf8E5M2>, tensor<4xi32>) -> tensor<4xf8E5M2> 
    %9 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>> 
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32> 
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E5M2>> 
    tt.return 
  } 
} 

// CHECK: %[[RESULT:.*]] = call @triton_gather(%[[SOURCE:.*]], %[[INDICES:.*]], %[[DIMENSION:.*]]) : (tensor<8xf8E5M2>, tensor<4xi32>, i32) -> tensor<4xf8E5M2>