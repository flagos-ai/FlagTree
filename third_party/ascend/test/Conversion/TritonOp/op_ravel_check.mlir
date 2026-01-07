// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s

// bfloat16
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<bf16> ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<4> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32> 
    %6 = arith.muli %5, %cst_0 : tensor<2x1x1xi32> 
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %9 = arith.muli %8, %cst : tensor<1x8x1xi32> 
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x8x1xi32> 
    %11 = tt.broadcast %9 : tensor<1x8x1xi32> -> tensor<2x8x1xi32> 
    %12 = arith.addi %10, %11 : tensor<2x8x1xi32> 
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %15 = tt.broadcast %12 : tensor<2x8x1xi32> -> tensor<2x8x4xi32> 
    %16 = tt.broadcast %14 : tensor<1x1x4xi32> -> tensor<2x8x4xi32> 
    %17 = arith.addi %15, %16 : tensor<2x8x4xi32> 
    %18 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<2x8x4x!tt.ptr<bf16>> 
    %19 = tt.addptr %18, %17 : tensor<2x8x4x!tt.ptr<bf16>>, tensor<2x8x4xi32> 
    %20 = tt.load %19 : tensor<2x8x4x!tt.ptr<bf16>> 
    %21 = tt.reshape %20 allow_reorder : tensor<2x8x4xbf16> -> tensor<64xbf16> 
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %23 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>> 
    %24 = tt.addptr %23, %22 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32> 
    tt.store %24, %21 : tensor<64x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xbf16>
// CHECK-SAME: %arg1: memref<?xbf16>

// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1]>>

// ----
// uint8

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> , %arg1: !tt.ptr<i8> )  {
    %cst = arith.constant dense<16> : tensor<1x256x1xi32> 
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32> 
    %4 = arith.muli %3, %cst : tensor<1x256x1xi32> 
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %7 = tt.broadcast %4 : tensor<1x256x1xi32> -> tensor<1x256x16xi32> 
    %8 = tt.broadcast %6 : tensor<1x1x16xi32> -> tensor<1x256x16xi32> 
    %9 = arith.addi %7, %8 : tensor<1x256x16xi32> 
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x256x16x!tt.ptr<i8>> 
    %11 = tt.addptr %10, %9 : tensor<1x256x16x!tt.ptr<i8>>, tensor<1x256x16xi32> 
    %12 = tt.load %11 : tensor<1x256x16x!tt.ptr<i8>> 
    %13 = tt.reshape %12 allow_reorder : tensor<1x256x16xi8> -> tensor<4096xi8> 
    %14 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32> 
    %15 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4096x!tt.ptr<i8>> 
    %16 = tt.addptr %15, %14 : tensor<4096x!tt.ptr<i8>>, tensor<4096xi32> 
    tt.store %16, %13 : tensor<4096x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>

// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [4096], strides: [1] : memref<?xi8> to memref<4096xi8, strided<[1]>>

// ----
// uint16

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> )  {
    %cst = arith.constant dense<3> : tensor<1x2x1xi32> 
    %cst_0 = arith.constant dense<3> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_0 : tensor<2x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32> 
    %8 = arith.muli %7, %cst : tensor<1x2x1xi32> 
    %9 = tt.broadcast %5 : tensor<2x1x1xi32> -> tensor<2x2x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x2x1xi32> -> tensor<2x2x1xi32> 
    %11 = arith.addi %9, %10 : tensor<2x2x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32> 
    %14 = tt.broadcast %11 : tensor<2x2x1xi32> -> tensor<2x2x3xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x3xi32> -> tensor<2x2x3xi32> 
    %16 = arith.addi %14, %15 : tensor<2x2x3xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x2x3x!tt.ptr<i16>> 
    %18 = tt.addptr %17, %16 : tensor<2x2x3x!tt.ptr<i16>>, tensor<2x2x3xi32> 
    %19 = tt.load %18 : tensor<2x2x3x!tt.ptr<i16>> 
    %20 = tt.reshape %19 allow_reorder : tensor<2x2x3xi16> -> tensor<12xi16> 
    %21 = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32> 
    %22 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<12x!tt.ptr<i16>> 
    %23 = tt.addptr %22, %21 : tensor<12x!tt.ptr<i16>>, tensor<12xi32> 
    tt.store %23, %20 : tensor<12x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>

// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [12], strides: [1] : memref<?xi16> to memref<12xi16, strided<[1]>>


// ----
// uint32
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32> 
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i32>> 
    %20 = tt.reshape %19 allow_reorder : tensor<8x8x4xi32> -> tensor<256xi32> 
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %22 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>> 
    %23 = tt.addptr %22, %21 : tensor<256x!tt.ptr<i32>>, tensor<256xi32> 
    tt.store %23, %20 : tensor<256x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>

// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [256], strides: [1] : memref<?xi32> to memref<256xi32, strided<[1]>>

// ----
// uint64
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> , %arg1: !tt.ptr<i64> ) {
    %cst = arith.constant dense<4> : tensor<1x125x1xi32> 
    %cst_0 = arith.constant dense<4> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<125> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32> 
    %6 = arith.muli %5, %cst_0 : tensor<2x1x1xi32> 
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<125xi32> -> tensor<1x125xi32> 
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x125xi32> -> tensor<1x125x1xi32> 
    %9 = arith.muli %8, %cst : tensor<1x125x1xi32> 
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x125x1xi32> 
    %11 = tt.broadcast %9 : tensor<1x125x1xi32> -> tensor<2x125x1xi32> 
    %12 = arith.addi %10, %11 : tensor<2x125x1xi32> 
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %15 = tt.broadcast %12 : tensor<2x125x1xi32> -> tensor<2x125x4xi32> 
    %16 = tt.broadcast %14 : tensor<1x1x4xi32> -> tensor<2x125x4xi32> 
    %17 = arith.addi %15, %16 : tensor<2x125x4xi32> 
    %18 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x125x4x!tt.ptr<i64>> 
    %19 = tt.addptr %18, %17 : tensor<2x125x4x!tt.ptr<i64>>, tensor<2x125x4xi32> 
    %20 = tt.load %19 : tensor<2x125x4x!tt.ptr<i64>> 
    %21 = tt.reshape %20 allow_reorder : tensor<2x125x4xi64> -> tensor<1000xi64>  
    %22 = tt.make_range {end = 1000 : i32, start = 0 : i32} : tensor<1000xi32> 
    %23 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1000x!tt.ptr<i64>> 
    %24 = tt.addptr %23, %22 : tensor<1000x!tt.ptr<i64>>, tensor<1000xi32> 
    tt.store %24, %21 : tensor<1000x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>

// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [1000], strides: [1] : memref<?xi64> to memref<1000xi64, strided<[1]>>

// ----
// bool

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i1> , %arg1: !tt.ptr<i1> )  {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %3 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<1x1x2x!tt.ptr<i1>> 
    %4 = tt.addptr %3, %2 : tensor<1x1x2x!tt.ptr<i1>>, tensor<1x1x2xi32> 
    %5 = tt.bitcast %4 : tensor<1x1x2x!tt.ptr<i1>> -> tensor<1x1x2x!tt.ptr<i8>> 
    %6 = tt.load %5 : tensor<1x1x2x!tt.ptr<i8>> 
    %7 = tt.reshape %6 allow_reorder : tensor<1x1x2xi8> -> tensor<2xi8> 
    %8 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<2x!tt.ptr<i1>> 
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<i1>>, tensor<2xi32> 
    %10 = tt.bitcast %9 : tensor<2x!tt.ptr<i1>> -> tensor<2x!tt.ptr<i8>> 
    tt.store %10, %7 : tensor<2x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>

// CHECK:  %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [2], strides: [1] : memref<?xi8> to memref<2xi8, strided<[1]>>


// -----
// f8E4M3FN

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<2x1x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32>
    %6 = arith.muli %5, %cst_0 : tensor<2x1x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %9 = arith.muli %8, %cst : tensor<1x8x1xi32>
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x8x1xi32>
    %11 = tt.broadcast %9 : tensor<1x8x1xi32> -> tensor<2x8x1xi32>
    %12 = arith.addi %10, %11 : tensor<2x8x1xi32>
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %15 = tt.broadcast %12 : tensor<2x8x1xi32> -> tensor<2x8x4xi32>
    %16 = tt.broadcast %14 : tensor<1x1x4xi32> -> tensor<2x8x4xi32>
    %17 = arith.addi %15, %16 : tensor<2x8x4xi32>
    %18 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<2x8x4x!tt.ptr<f8E4M3FN>>
    %19 = tt.addptr %18, %17 : tensor<2x8x4x!tt.ptr<f8E4M3FN>>, tensor<2x8x4xi32>
    %20 = tt.load %19 : tensor<2x8x4x!tt.ptr<f8E4M3FN>>
    %21 = tt.reshape %20 allow_reorder : tensor<2x8x4xf8E4M3FN> -> tensor<64xf8E4M3FN>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %23 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x!tt.ptr<f8E4M3FN>>
    %24 = tt.addptr %23, %22 : tensor<64x!tt.ptr<f8E4M3FN>>, tensor<64xi32>
    tt.store %24, %21 : tensor<64x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}


// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [64], strides: [1] : memref<?xf8E4M3FN> to memref<64xf8E4M3FN, strided<[1]>>


// -----
// f8E5M2

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<2x1x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32>
    %6 = arith.muli %5, %cst_0 : tensor<2x1x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %9 = arith.muli %8, %cst : tensor<1x8x1xi32>
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x8x1xi32>
    %11 = tt.broadcast %9 : tensor<1x8x1xi32> -> tensor<2x8x1xi32>
    %12 = arith.addi %10, %11 : tensor<2x8x1xi32>
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %15 = tt.broadcast %12 : tensor<2x8x1xi32> -> tensor<2x8x4xi32>
    %16 = tt.broadcast %14 : tensor<1x1x4xi32> -> tensor<2x8x4xi32>
    %17 = arith.addi %15, %16 : tensor<2x8x4xi32>
    %18 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<2x8x4x!tt.ptr<f8E5M2>>
    %19 = tt.addptr %18, %17 : tensor<2x8x4x!tt.ptr<f8E5M2>>, tensor<2x8x4xi32>
    %20 = tt.load %19 : tensor<2x8x4x!tt.ptr<f8E5M2>>
    %21 = tt.reshape %20 allow_reorder : tensor<2x8x4xf8E5M2> -> tensor<64xf8E5M2>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %23 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<64x!tt.ptr<f8E5M2>>
    %24 = tt.addptr %23, %22 : tensor<64x!tt.ptr<f8E5M2>>, tensor<64xi32>
    tt.store %24, %21 : tensor<64x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [64], strides: [1] : memref<?xf8E5M2> to memref<64xf8E5M2, strided<[1]>>