// RUN: triton-adapter-opt %s --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// dtype: uint8 & int8
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<i8>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<i8>> 
    %6 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<i8>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<i8>> 
    tt.return 
  } 
}

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xi8> to memref<4xi8, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi8>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xi8, strided<[1], offset: 5>> to memref<4xi8>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xi8>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xi8> to memref<4xi8, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xi8>, memref<4xi8, strided<[1]>>) -> ()

// -----

// dtype: uint16 & int16
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<i16>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<i16>> 
    %6 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<i16>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xi16> to memref<4xi16, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi16>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xi16, strided<[1], offset: 5>> to memref<4xi16>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xi16>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xi16> to memref<4xi16, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xi16>, memref<4xi16, strided<[1]>>) -> ()

// -----

// dtype: uint32 & int32
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<i32>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<i32>> 
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<i32>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi32>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xi32, strided<[1], offset: 5>> to memref<4xi32>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xi32>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xi32>, memref<4xi32, strided<[1]>>) -> ()


// -----

// dtype: uint64 & int64
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<i64>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<i64>> 
    %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<i64>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xi64> to memref<4xi64, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi64>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xi64, strided<[1], offset: 5>> to memref<4xi64>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xi64>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xi64> to memref<4xi64, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xi64>, memref<4xi64, strided<[1]>>) -> ()

// -----

// dtype: bool
module {
  tt.func public @triton_func(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %1 = tt.splat %0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>> 
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %3 = arith.extsi %2 : tensor<4xi32> to tensor<4xi64> 
    %4 = arith.addi %3, %cst : tensor<4xi64> 
    %5 = tt.addptr %1, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi64> 
    %6 = tt.load %5 : tensor<4x!tt.ptr<i8>> 
    %7 = tt.bitcast %arg1 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %8 = tt.splat %7 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>> 
    %9 = tt.addptr %8, %3 : tensor<4x!tt.ptr<i8>>, tensor<4xi64>
    tt.store %9, %6 : tensor<4x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xi8> to memref<4xi8, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi8>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xi8, strided<[1], offset: 5>> to memref<4xi8>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xi8>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xi8> to memref<4xi8, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xi8>, memref<4xi8, strided<[1]>>) -> ()

// -----

// dtype: float16
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<f16>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<f16>> 
    %6 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<f16>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<f16>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xf16> to memref<4xf16, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf16>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xf16, strided<[1], offset: 5>> to memref<4xf16>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xf16>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xf16> to memref<4xf16, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xf16>, memref<4xf16, strided<[1]>>) -> ()


// -----

// dtype: float32
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<f32>> 
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<f32>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<f32>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xf32, strided<[1], offset: 5>> to memref<4xf32>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xf32>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()


// -----

// dtype: bfloat16
module {
  tt.func public @triton_func(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<bf16>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<bf16>> 
    %6 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<bf16>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xbf16> to memref<4xbf16, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xbf16>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xbf16, strided<[1], offset: 5>> to memref<4xbf16>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xbf16>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xbf16> to memref<4xbf16, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xbf16>, memref<4xbf16, strided<[1]>>) -> ()


// -----

// dtype: float8_e5m2
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<f8E5M2>> 
    %6 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<f8E5M2>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xf8E5M2> to memref<4xf8E5M2, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf8E5M2>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xf8E5M2, strided<[1], offset: 5>> to memref<4xf8E5M2>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xf8E5M2>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xf8E5M2> to memref<4xf8E5M2, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xf8E5M2>, memref<4xf8E5M2, strided<[1]>>) -> ()


// -----

// dtype: float8_e4m3
module {
  tt.func public @triton_func(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<4xi64> 
    %0 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = arith.extsi %1 : tensor<4xi32> to tensor<4xi64> 
    %3 = arith.addi %2, %cst : tensor<4xi64> 
    %4 = tt.addptr %0, %3 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi64> 
    %5 = tt.load %4 : tensor<4x!tt.ptr<f8E4M3FN>> 
    %6 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>> 
    %7 = tt.addptr %6, %2 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi64> 
    tt.store %7, %5 : tensor<4x!tt.ptr<f8E4M3FN>> 
    tt.return 
  } 
} 

// CHECK: %[[REINT_CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [5], sizes: [4], strides: [1] : memref<?xf8E4M3FN> to memref<4xf8E4M3FN, strided<[1], offset: 5>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf8E4M3FN>
// CHECK: memref.copy %[[REINT_CAST0]], %[[ALLOC]] : memref<4xf8E4M3FN, strided<[1], offset: 5>> to memref<4xf8E4M3FN>
// CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4xf8E4M3FN>
// CHECK: %[[REINT_CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [4], strides: [1] : memref<?xf8E4M3FN> to memref<4xf8E4M3FN, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL0]] in writable %[[REINT_CAST1]] : (tensor<4xf8E4M3FN>, memref<4xf8E4M3FN, strided<[1]>>) -> ()

