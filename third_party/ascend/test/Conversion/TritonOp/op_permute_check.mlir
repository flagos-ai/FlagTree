// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s
// bfloat16

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<bf16> ) {
    %c8_i32 = arith.constant 8 : i32 
    %cst = arith.constant dense<2> : tensor<8x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %2 = arith.muli %0, %c8_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x8xi32> 
    %5 = arith.addi %4, %3 : tensor<1x8xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x8x!tt.ptr<bf16>> 
    %7 = tt.addptr %6, %5 : tensor<1x8x!tt.ptr<bf16>>, tensor<1x8xi32> 
    %8 = tt.load %7 : tensor<1x8x!tt.ptr<bf16>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x8xbf16> -> tensor<8x1xbf16> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %11 = arith.muli %10, %cst : tensor<8x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<8x1xi32> 
    %13 = arith.addi %11, %12 : tensor<8x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x1x!tt.ptr<bf16>> 
    %15 = tt.addptr %14, %13 : tensor<8x1x!tt.ptr<bf16>>, tensor<8x1xi32> 
    tt.store %15, %9 : tensor<8x1x!tt.ptr<bf16>> 
    tt.return 
  } 
}

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x8xbf16, strided<[8, 1], offset: ?>> to memref<1x8xbf16>

// -----
// uint8

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<i8>  , %arg1: !tt.ptr<i8> )  {
    %c256_i32 = arith.constant 256 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %2 = arith.muli %0, %c256_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x256xi32> 
    %5 = arith.addi %4, %3 : tensor<1x256xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x256x!tt.ptr<i8>> 
    %7 = tt.addptr %6, %5 : tensor<1x256x!tt.ptr<i8>>, tensor<1x256xi32> 
    %8 = tt.load %7 : tensor<1x256x!tt.ptr<i8>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x256xi8> -> tensor<256x1xi8> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32> 
    %11 = tt.splat %0 : i32 -> tensor<256x1xi32> 
    %12 = arith.addi %10, %11 : tensor<256x1xi32> 
    %13 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>> 
    %14 = tt.addptr %13, %12 : tensor<256x1x!tt.ptr<i8>>, tensor<256x1xi32> 
    tt.store %14, %9 : tensor<256x1x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x256xi8, strided<[256, 1], offset: ?>> to memref<1x256xi8>

// -----
// uint16
module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> ) {
    %c2_i32 = arith.constant 2 : i32 
    %cst = arith.constant dense<2> : tensor<2x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %2 = arith.muli %0, %c2_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x2xi32> 
    %5 = arith.addi %4, %3 : tensor<1x2xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x2x!tt.ptr<i16>> 
    %7 = tt.addptr %6, %5 : tensor<1x2x!tt.ptr<i16>>, tensor<1x2xi32> 
    %8 = tt.load %7 : tensor<1x2x!tt.ptr<i16>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x2xi16> -> tensor<2x1xi16> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %11 = arith.muli %10, %cst : tensor<2x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<2x1xi32> 
    %13 = arith.addi %11, %12 : tensor<2x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x1x!tt.ptr<i16>> 
    %15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i16>>, tensor<2x1xi32> 
    tt.store %15, %9 : tensor<2x1x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x2xi16, strided<[2, 1], offset: ?>> to memref<1x2xi16>

// -----
// uint32

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> ) {
    %c8_i32 = arith.constant 8 : i32 
    %cst = arith.constant dense<8> : tensor<8x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %2 = arith.muli %0, %c8_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x8xi32> 
    %5 = arith.addi %4, %3 : tensor<1x8xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x8x!tt.ptr<i32>> 
    %7 = tt.addptr %6, %5 : tensor<1x8x!tt.ptr<i32>>, tensor<1x8xi32> 
    %8 = tt.load %7 : tensor<1x8x!tt.ptr<i32>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x8xi32> -> tensor<8x1xi32> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %11 = arith.muli %10, %cst : tensor<8x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<8x1xi32> 
    %13 = arith.addi %11, %12 : tensor<8x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>> 
    %15 = tt.addptr %14, %13 : tensor<8x1x!tt.ptr<i32>>, tensor<8x1xi32> 
    tt.store %15, %9 : tensor<8x1x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x8xi32, strided<[8, 1], offset: ?>> to memref<1x8xi32>

// -----
// uint64

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64> )  {
    %c125_i32 = arith.constant 125 : i32 
    %cst = arith.constant dense<2> : tensor<125x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = arith.muli %0, %c125_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<125xi32> -> tensor<1x125xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x125xi32> 
    %5 = arith.addi %4, %3 : tensor<1x125xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x125x!tt.ptr<i64>> 
    %7 = tt.addptr %6, %5 : tensor<1x125x!tt.ptr<i64>>, tensor<1x125xi32> 
    %8 = tt.load %7 : tensor<1x125x!tt.ptr<i64>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x125xi64> -> tensor<125x1xi64> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<125xi32> -> tensor<125x1xi32> 
    %11 = arith.muli %10, %cst : tensor<125x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<125x1xi32> 
    %13 = arith.addi %11, %12 : tensor<125x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<125x1x!tt.ptr<i64>> 
    %15 = tt.addptr %14, %13 : tensor<125x1x!tt.ptr<i64>>, tensor<125x1xi32> 
    tt.store %15, %9 : tensor<125x1x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x125xi64, strided<[125, 1], offset: ?>> to memref<1x125xi64>


// -----
// bool
module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<i1> , %arg1: !tt.ptr<i1> ) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 
    %1 = tt.addptr %arg1, %0 : !tt.ptr<i1>, i32   
    %2 = tt.bitcast %1 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %3 = tt.splat %2 : !tt.ptr<i8> -> tensor<1x1x!tt.ptr<i8>> 
    %4 = tt.load %3 : tensor<1x1x!tt.ptr<i8>> 
    %5 = tt.trans %4 {order = array<i32: 1, 0>} : tensor<1x1xi8> -> tensor<1x1xi8> 
    %6 = tt.addptr %arg0, %0 : !tt.ptr<i1>, i32 
    %7 = tt.bitcast %6 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %8 = tt.splat %7 : !tt.ptr<i8> -> tensor<1x1x!tt.ptr<i8>> 
    tt.store %8, %5 : tensor<1x1x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK: memref.copy %[[X:.*]], %[[Y:.*]] : memref<1x1xi8, strided<[1, 1], offset: ?>> to memref<1x1xi8>


// -----
// f8E4M3FN

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32 
    %cst = arith.constant dense<2> : tensor<8x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %2 = arith.muli %0, %c8_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x8xi32> 
    %5 = arith.addi %4, %3 : tensor<1x8xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<1x8x!tt.ptr<f8E4M3FN>> 
    %7 = tt.addptr %6, %5 : tensor<1x8x!tt.ptr<f8E4M3FN>>, tensor<1x8xi32> 
    %8 = tt.load %7 : tensor<1x8x!tt.ptr<f8E4M3FN>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x8xf8E4M3FN> -> tensor<8x1xf8E4M3FN> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %11 = arith.muli %10, %cst : tensor<8x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<8x1xi32> 
    %13 = arith.addi %11, %12 : tensor<8x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x1x!tt.ptr<f8E4M3FN>> 
    %15 = tt.addptr %14, %13 : tensor<8x1x!tt.ptr<f8E4M3FN>>, tensor<8x1xi32> 
    tt.store %15, %9 : tensor<8x1x!tt.ptr<f8E4M3FN>> 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @fn_npu_021
// CHECK-SAME: %arg2: memref<?xf8E4M3FN>
// CHECK-SAME: %arg3: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [%[[Y:.*]]], sizes: [1, 8], strides: [8, 1] : memref<?xf8E4M3FN> to memref<1x8xf8E4M3FN, strided<[8, 1], offset: ?>>


// -----
// f8E5M2

module {
  tt.func public @fn_npu_021(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32 
    %cst = arith.constant dense<2> : tensor<8x1xi32> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %2 = arith.muli %0, %c8_i32 : i32 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.splat %2 : i32 -> tensor<1x8xi32> 
    %5 = arith.addi %4, %3 : tensor<1x8xi32> 
    %6 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<1x8x!tt.ptr<f8E5M2>> 
    %7 = tt.addptr %6, %5 : tensor<1x8x!tt.ptr<f8E5M2>>, tensor<1x8xi32> 
    %8 = tt.load %7 : tensor<1x8x!tt.ptr<f8E5M2>> 
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<1x8xf8E5M2> -> tensor<8x1xf8E5M2> 
    %10 = tt.expand_dims %1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %11 = arith.muli %10, %cst : tensor<8x1xi32> 
    %12 = tt.splat %0 : i32 -> tensor<8x1xi32> 
    %13 = arith.addi %11, %12 : tensor<8x1xi32> 
    %14 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<8x1x!tt.ptr<f8E5M2>> 
    %15 = tt.addptr %14, %13 : tensor<8x1x!tt.ptr<f8E5M2>>, tensor<8x1xi32> 
    tt.store %15, %9 : tensor<8x1x!tt.ptr<f8E5M2>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_021
// CHECK-SAME: %arg2: memref<?xf8E5M2>
// CHECK-SAME: %arg3: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [%[[Y:.*]]], sizes: [1, 8], strides: [8, 1] : memref<?xf8E5M2> to memref<1x8xf8E5M2, strided<[8, 1], offset: ?>>