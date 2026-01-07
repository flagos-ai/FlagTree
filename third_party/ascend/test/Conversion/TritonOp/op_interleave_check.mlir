// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s
// bfloat16

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_2 = arith.constant dense<4> : tensor<8x1x1xi32> 
    %cst_3 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_3 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_2 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst_1 : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x8x4x!tt.ptr<bf16>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<bf16>>, tensor<8x8x4xi32> 
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<bf16>> 
    %20 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<8x8x4x!tt.ptr<bf16>> 
    %21 = tt.addptr %20, %16 : tensor<8x8x4x!tt.ptr<bf16>>, tensor<8x8x4xi32> 
    %22 = tt.load %21 : tensor<8x8x4x!tt.ptr<bf16>> 
    %23 = tt.join %19, %22 : tensor<8x8x4xbf16> -> tensor<8x8x4x2xbf16> 
    %24 = tt.reshape %23 : tensor<8x8x4x2xbf16> -> tensor<8x8x8xbf16> 
    %25 = arith.muli %5, %cst_0 : tensor<8x1x1xi32> 
    %26 = arith.muli %8, %cst : tensor<1x8x1xi32> 
    %27 = tt.broadcast %25 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %28 = tt.broadcast %26 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %29 = arith.addi %27, %28 : tensor<8x8x1xi32> 
    %30 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %31 = tt.broadcast %29 : tensor<8x8x1xi32> -> tensor<8x8x8xi32> 
    %32 = tt.broadcast %30 : tensor<1x1x8xi32> -> tensor<8x8x8xi32> 
    %33 = arith.addi %31, %32 : tensor<8x8x8xi32> 
    %34 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x8x8x!tt.ptr<bf16>> 
    %35 = tt.addptr %34, %33 : tensor<8x8x8x!tt.ptr<bf16>>, tensor<8x8x8xi32> 
    tt.store %35, %24 : tensor<8x8x8x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xbf16>
// CHECK-SAME: %arg1: memref<?xbf16>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xbf16> to memref<8x8x4xbf16, strided<[32, 4, 1]>>


// -----
// uint8

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc("test_general_interleave.py":183:0)) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x256x1xi32> 
    %cst_0 = arith.constant dense<16> : tensor<1x256x1xi32> 
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32> 
    %4 = arith.muli %3, %cst_0 : tensor<1x256x1xi32> 
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %7 = tt.broadcast %4 : tensor<1x256x1xi32> -> tensor<1x256x16xi32> 
    %8 = tt.broadcast %6 : tensor<1x1x16xi32> -> tensor<1x256x16xi32> 
    %9 = arith.addi %7, %8 : tensor<1x256x16xi32> 
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x256x16x!tt.ptr<i8>> 
    %11 = tt.addptr %10, %9 : tensor<1x256x16x!tt.ptr<i8>>, tensor<1x256x16xi32> 
    %12 = tt.load %11 : tensor<1x256x16x!tt.ptr<i8>> 
    %13 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<1x256x16x!tt.ptr<i8>> 
    %14 = tt.addptr %13, %9 : tensor<1x256x16x!tt.ptr<i8>>, tensor<1x256x16xi32> 
    %15 = tt.load %14 : tensor<1x256x16x!tt.ptr<i8>> 
    %16 = tt.join %12, %15 : tensor<1x256x16xi8> -> tensor<1x256x16x2xi8> 
    %17 = tt.reshape %16 : tensor<1x256x16x2xi8> -> tensor<1x256x32xi8> 
    %18 = arith.muli %4, %cst : tensor<1x256x1xi32> 
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32> 
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32> 
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32> 
    %22 = tt.broadcast %18 : tensor<1x256x1xi32> -> tensor<1x256x32xi32> 
    %23 = tt.broadcast %21 : tensor<1x1x32xi32> -> tensor<1x256x32xi32> 
    %24 = arith.addi %22, %23 : tensor<1x256x32xi32> 
    %25 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x256x32x!tt.ptr<i8>> 
    %26 = tt.addptr %25, %24 : tensor<1x256x32x!tt.ptr<i8>>, tensor<1x256x32xi32> 
    tt.store %26, %17 : tensor<1x256x32x!tt.ptr<i8>> 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [1, 256, 16], strides: [4096, 16, 1] : memref<?xi8> to memref<1x256x16xi8, strided<[4096, 16, 1]>>

// -----
// uint16

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x2x1xi32> 
    %cst_0 = arith.constant dense<3> : tensor<1x2x1xi32> 
    %cst_1 = arith.constant dense<3> : tensor<2x1x1xi32> 
    %cst_2 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %4 = arith.muli %3, %cst_2 : tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32> 
    %8 = arith.muli %7, %cst_0 : tensor<1x2x1xi32> 
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
    %20 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<2x2x3x!tt.ptr<i16>> 
    %21 = tt.addptr %20, %16 : tensor<2x2x3x!tt.ptr<i16>>, tensor<2x2x3xi32> 
    %22 = tt.load %21 : tensor<2x2x3x!tt.ptr<i16>> 
    %23 = tt.join %19, %22 : tensor<2x2x3xi16> -> tensor<2x2x3x2xi16> 
    %24 = tt.reshape %23 : tensor<2x2x3x2xi16> -> tensor<2x2x6xi16> 
    %25 = arith.muli %5, %cst_2 : tensor<2x1x1xi32> 
    %26 = arith.muli %8, %cst : tensor<1x2x1xi32> 
    %27 = tt.broadcast %25 : tensor<2x1x1xi32> -> tensor<2x2x1xi32> 
    %28 = tt.broadcast %26 : tensor<1x2x1xi32> -> tensor<2x2x1xi32> 
    %29 = arith.addi %27, %28 : tensor<2x2x1xi32> 
    %30 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32> 
    %31 = tt.expand_dims %30 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32> 
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<1x6xi32> -> tensor<1x1x6xi32> 
    %33 = tt.broadcast %29 : tensor<2x2x1xi32> -> tensor<2x2x6xi32> 
    %34 = tt.broadcast %32 : tensor<1x1x6xi32> -> tensor<2x2x6xi32> 
    %35 = arith.addi %33, %34 : tensor<2x2x6xi32> 
    %36 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x2x6x!tt.ptr<i16>> 
    %37 = tt.addptr %36, %35 : tensor<2x2x6x!tt.ptr<i16>>, tensor<2x2x6xi32> 
    tt.store %37, %24 : tensor<2x2x6x!tt.ptr<i16>> 
    tt.return 
  } 
}

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [2, 2, 3], strides: [6, 3, 1] : memref<?xi16> to memref<2x2x3xi16, strided<[6, 3, 1]>>

// -----
// uint32

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_2 = arith.constant dense<4> : tensor<8x1x1xi32> 
    %cst_3 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_3 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_2 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst_1 : tensor<1x8x1xi32> 
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
    %20 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>> 
    %21 = tt.addptr %20, %16 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32> 
    %22 = tt.load %21 : tensor<8x8x4x!tt.ptr<i32>> 
    %23 = tt.join %19, %22 : tensor<8x8x4xi32> -> tensor<8x8x4x2xi32> 
    %24 = tt.reshape %23 : tensor<8x8x4x2xi32> -> tensor<8x8x8xi32> 
    %25 = arith.muli %5, %cst_0 : tensor<8x1x1xi32> 
    %26 = arith.muli %8, %cst : tensor<1x8x1xi32> 
    %27 = tt.broadcast %25 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %28 = tt.broadcast %26 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %29 = arith.addi %27, %28 : tensor<8x8x1xi32> 
    %30 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %31 = tt.broadcast %29 : tensor<8x8x1xi32> -> tensor<8x8x8xi32> 
    %32 = tt.broadcast %30 : tensor<1x1x8xi32> -> tensor<8x8x8xi32> 
    %33 = arith.addi %31, %32 : tensor<8x8x8xi32> 
    %34 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x8x!tt.ptr<i32>> 
    %35 = tt.addptr %34, %33 : tensor<8x8x8x!tt.ptr<i32>>, tensor<8x8x8xi32> 
    tt.store %35, %24 : tensor<8x8x8x!tt.ptr<i32>> 
    tt.return 
  } 
}  


// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi32> to memref<8x8x4xi32, strided<[32, 4, 1]>>

// -----
// uint64

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x125x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<4> : tensor<1x125x1xi32> 
    %cst_2 = arith.constant dense<4> : tensor<2x1x1xi32> 
    %cst_3 = arith.constant dense<125> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_3 : tensor<2x1x1xi32> 
    %6 = arith.muli %5, %cst_2 : tensor<2x1x1xi32> 
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<125xi32> -> tensor<1x125xi32> 
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x125xi32> -> tensor<1x125x1xi32> 
    %9 = arith.muli %8, %cst_1 : tensor<1x125x1xi32> 
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
    %21 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<2x125x4x!tt.ptr<i64>> 
    %22 = tt.addptr %21, %17 : tensor<2x125x4x!tt.ptr<i64>>, tensor<2x125x4xi32> 
    %23 = tt.load %22 : tensor<2x125x4x!tt.ptr<i64>> 
    %24 = tt.join %20, %23 : tensor<2x125x4xi64> -> tensor<2x125x4x2xi64> 
    %25 = tt.reshape %24 : tensor<2x125x4x2xi64> -> tensor<2x125x8xi64> 
    %26 = arith.muli %6, %cst_0 : tensor<2x1x1xi32> 
    %27 = arith.muli %9, %cst : tensor<1x125x1xi32> 
    %28 = tt.broadcast %26 : tensor<2x1x1xi32> -> tensor<2x125x1xi32> 
    %29 = tt.broadcast %27 : tensor<1x125x1xi32> -> tensor<2x125x1xi32> 
    %30 = arith.addi %28, %29 : tensor<2x125x1xi32> 
    %31 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %34 = tt.broadcast %30 : tensor<2x125x1xi32> -> tensor<2x125x8xi32> 
    %35 = tt.broadcast %33 : tensor<1x1x8xi32> -> tensor<2x125x8xi32> 
    %36 = arith.addi %34, %35 : tensor<2x125x8xi32> 
    %37 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x125x8x!tt.ptr<i64>> 
    %38 = tt.addptr %37, %36 : tensor<2x125x8x!tt.ptr<i64>>, tensor<2x125x8xi32> 
    tt.store %38, %25 : tensor<2x125x8x!tt.ptr<i64>> 
    tt.return 
  } 
} 
 

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [2, 125, 4], strides: [500, 4, 1] : memref<?xi64> to memref<2x125x4xi64, strided<[500, 4, 1]>> 


// -----
// bool

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %3 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<1x1x2x!tt.ptr<i1>> 
    %4 = tt.addptr %3, %2 : tensor<1x1x2x!tt.ptr<i1>>, tensor<1x1x2xi32> 
    %5 = tt.bitcast %4 : tensor<1x1x2x!tt.ptr<i1>> -> tensor<1x1x2x!tt.ptr<i8>> 
    %6 = tt.load %5 : tensor<1x1x2x!tt.ptr<i8>> 
    %7 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<1x1x2x!tt.ptr<i1>> 
    %8 = tt.addptr %7, %2 : tensor<1x1x2x!tt.ptr<i1>>, tensor<1x1x2xi32> 
    %9 = tt.bitcast %8 : tensor<1x1x2x!tt.ptr<i1>> -> tensor<1x1x2x!tt.ptr<i8>> 
    %10 = tt.load %9 : tensor<1x1x2x!tt.ptr<i8>> 
    %11 = tt.join %6, %10 : tensor<1x1x2xi8> -> tensor<1x1x2x2xi8> 
    %12 = tt.reshape %11 : tensor<1x1x2x2xi8> -> tensor<1x1x4xi8> 
    %13 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %16 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1x1x4x!tt.ptr<i1>> 
    %17 = tt.addptr %16, %15 : tensor<1x1x4x!tt.ptr<i1>>, tensor<1x1x4xi32> 
    %18 = tt.bitcast %17 : tensor<1x1x4x!tt.ptr<i1>> -> tensor<1x1x4x!tt.ptr<i8>> 
    tt.store %18, %12 : tensor<1x1x4x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]] =  memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [1, 1, 2], strides: [2, 2, 1] : memref<?xi8> to memref<1x1x2xi8, strided<[2, 2, 1]>>

// -----
// f8E4M3FN

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_2 = arith.constant dense<4> : tensor<8x1x1xi32> 
    %cst_3 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_3 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_2 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst_1 : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<8x8x4x!tt.ptr<f8E4M3FN>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<f8E4M3FN>>, tensor<8x8x4xi32> 
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<f8E4M3FN>> 
    %20 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<8x8x4x!tt.ptr<f8E4M3FN>> 
    %21 = tt.addptr %20, %16 : tensor<8x8x4x!tt.ptr<f8E4M3FN>>, tensor<8x8x4xi32> 
    %22 = tt.load %21 : tensor<8x8x4x!tt.ptr<f8E4M3FN>> 
    %23 = tt.join %19, %22 : tensor<8x8x4xf8E4M3FN> -> tensor<8x8x4x2xf8E4M3FN> 
    %24 = tt.reshape %23 : tensor<8x8x4x2xf8E4M3FN> -> tensor<8x8x8xf8E4M3FN> 
    %25 = arith.muli %5, %cst_0 : tensor<8x1x1xi32> 
    %26 = arith.muli %8, %cst : tensor<1x8x1xi32> 
    %27 = tt.broadcast %25 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %28 = tt.broadcast %26 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %29 = arith.addi %27, %28 : tensor<8x8x1xi32> 
    %30 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %31 = tt.broadcast %29 : tensor<8x8x1xi32> -> tensor<8x8x8xi32> 
    %32 = tt.broadcast %30 : tensor<1x1x8xi32> -> tensor<8x8x8xi32> 
    %33 = arith.addi %31, %32 : tensor<8x8x8xi32> 
    %34 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x8x8x!tt.ptr<f8E4M3FN>> 
    %35 = tt.addptr %34, %33 : tensor<8x8x8x!tt.ptr<f8E4M3FN>>, tensor<8x8x8xi32> 
    tt.store %35, %24 : tensor<8x8x8x!tt.ptr<f8E4M3FN>> 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] =  memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xf8E4M3FN> to memref<8x8x4xf8E4M3FN, strided<[32, 4, 1]>>


// -----
// f8E5M2

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x256x1xi32> 
    %cst_0 = arith.constant dense<16> : tensor<1x256x1xi32> 
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32> 
    %4 = arith.muli %3, %cst_0 : tensor<1x256x1xi32> 
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %7 = tt.broadcast %4 : tensor<1x256x1xi32> -> tensor<1x256x16xi32> 
    %8 = tt.broadcast %6 : tensor<1x1x16xi32> -> tensor<1x256x16xi32> 
    %9 = arith.addi %7, %8 : tensor<1x256x16xi32> 
    %10 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<1x256x16x!tt.ptr<f8E5M2>> 
    %11 = tt.addptr %10, %9 : tensor<1x256x16x!tt.ptr<f8E5M2>>, tensor<1x256x16xi32> 
    %12 = tt.load %11 : tensor<1x256x16x!tt.ptr<f8E5M2>> 
    %13 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<1x256x16x!tt.ptr<f8E5M2>> 
    %14 = tt.addptr %13, %9 : tensor<1x256x16x!tt.ptr<f8E5M2>>, tensor<1x256x16xi32> 
    %15 = tt.load %14 : tensor<1x256x16x!tt.ptr<f8E5M2>> 
    %16 = tt.join %12, %15 : tensor<1x256x16xf8E5M2> -> tensor<1x256x16x2xf8E5M2> 
    %17 = tt.reshape %16 : tensor<1x256x16x2xf8E5M2> -> tensor<1x256x32xf8E5M2> 
    %18 = arith.muli %4, %cst : tensor<1x256x1xi32> 
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32> 
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32> 
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32> 
    %22 = tt.broadcast %18 : tensor<1x256x1xi32> -> tensor<1x256x32xi32> 
    %23 = tt.broadcast %21 : tensor<1x1x32xi32> -> tensor<1x256x32xi32> 
    %24 = arith.addi %22, %23 : tensor<1x256x32xi32> 
    %25 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<1x256x32x!tt.ptr<f8E5M2>> 
    %26 = tt.addptr %25, %24 : tensor<1x256x32x!tt.ptr<f8E5M2>>, tensor<1x256x32xi32> 
    tt.store %26, %17 : tensor<1x256x32x!tt.ptr<f8E5M2>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] =  memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [1, 256, 16], strides: [4096, 16, 1] : memref<?xf8E5M2> to memref<1x256x16xf8E5M2, strided<[4096, 16, 1]>> 