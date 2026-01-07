// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s


//bfloat16

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<bf16> ) {
    %cst = arith.constant dense<64> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<16x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<1x64x1xi32> 
    %cst_2 = arith.constant dense<16> : tensor<2x1x1xi32> 
    %cst_3 = arith.constant dense<64> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_3 : tensor<2x1x1xi32> 
    %6 = arith.muli %5, %cst_2 : tensor<2x1x1xi32> 
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x64xi32> -> tensor<1x64x1xi32> 
    %9 = arith.muli %8, %cst_1 : tensor<1x64x1xi32> 
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x64x1xi32> 
    %11 = tt.broadcast %9 : tensor<1x64x1xi32> -> tensor<2x64x1xi32> 
    %12 = arith.addi %10, %11 : tensor<2x64x1xi32> 
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %15 = tt.broadcast %12 : tensor<2x64x1xi32> -> tensor<2x64x16xi32> 
    %16 = tt.broadcast %14 : tensor<1x1x16xi32> -> tensor<2x64x16xi32> 
    %17 = arith.addi %15, %16 : tensor<2x64x16xi32> 
    %18 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<2x64x16x!tt.ptr<bf16>> 
    %19 = tt.addptr %18, %17 : tensor<2x64x16x!tt.ptr<bf16>>, tensor<2x64x16xi32> 
    %20 = tt.load %19 : tensor<2x64x16x!tt.ptr<bf16>> 
    %21 = tt.reshape %20 : tensor<2x64x16xbf16> -> tensor<16x128xbf16> 
    %22 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %23 = arith.muli %22, %cst_0 : tensor<16x1xi32> 
    %24 = arith.muli %23, %cst : tensor<16x1xi32> 
    %25 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32> 
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32> 
    %27 = tt.broadcast %24 : tensor<16x1xi32> -> tensor<16x128xi32> 
    %28 = tt.broadcast %26 : tensor<1x128xi32> -> tensor<16x128xi32> 
    %29 = arith.addi %27, %28 : tensor<16x128xi32> 
    %30 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>> 
    %31 = tt.addptr %30, %29 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32> 
    tt.store %31, %21 : tensor<16x128x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xbf16>
// CHECK-SAME: %arg1: memref<?xbf16>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x64x16xbf16>, tensor<2xi64>) -> tensor<16x128xbf16>

// -----
//uint8

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> , %arg1: !tt.ptr<i8> )  {
    %cst = arith.constant dense<4> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<16x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<1x4x1xi32> 
    %cst_2 = arith.constant dense<16> : tensor<2x1x1xi32> 
    %cst_3 = arith.constant dense<4> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_3 : tensor<2x1x1xi32> 
    %6 = arith.muli %5, %cst_2 : tensor<2x1x1xi32> 
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32> 
    %9 = arith.muli %8, %cst_1 : tensor<1x4x1xi32> 
    %10 = tt.broadcast %6 : tensor<2x1x1xi32> -> tensor<2x4x1xi32> 
    %11 = tt.broadcast %9 : tensor<1x4x1xi32> -> tensor<2x4x1xi32> 
    %12 = arith.addi %10, %11 : tensor<2x4x1xi32> 
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %15 = tt.broadcast %12 : tensor<2x4x1xi32> -> tensor<2x4x16xi32> 
    %16 = tt.broadcast %14 : tensor<1x1x16xi32> -> tensor<2x4x16xi32> 
    %17 = arith.addi %15, %16 : tensor<2x4x16xi32> 
    %18 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<2x4x16x!tt.ptr<i8>> 
    %19 = tt.addptr %18, %17 : tensor<2x4x16x!tt.ptr<i8>>, tensor<2x4x16xi32> 
    %20 = tt.load %19 : tensor<2x4x16x!tt.ptr<i8>> 
    %21 = tt.reshape %20 : tensor<2x4x16xi8> -> tensor<16x8xi8>
    %22 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %23 = arith.muli %22, %cst_0 : tensor<16x1xi32> 
    %24 = arith.muli %23, %cst : tensor<16x1xi32> 
    %25 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %27 = tt.broadcast %24 : tensor<16x1xi32> -> tensor<16x8xi32> 
    %28 = tt.broadcast %26 : tensor<1x8xi32> -> tensor<16x8xi32> 
    %29 = arith.addi %27, %28 : tensor<16x8xi32> 
    %30 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<16x8x!tt.ptr<i8>> 
    %31 = tt.addptr %30, %29 : tensor<16x8x!tt.ptr<i8>>, tensor<16x8xi32> 
    tt.store %31, %21 : tensor<16x8x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x4x16xi8>, tensor<2xi64>) -> tensor<16x8xi8>

// -----
// uint16
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> ) {
    %cst = arith.constant dense<2> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<16> : tensor<1x2x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<2x1x1xi32> 
    %cst_2 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
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
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %14 = tt.broadcast %11 : tensor<2x2x1xi32> -> tensor<2x2x16xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x16xi32> -> tensor<2x2x16xi32> 
    %16 = arith.addi %14, %15 : tensor<2x2x16xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x2x16x!tt.ptr<i16>> 
    %18 = tt.addptr %17, %16 : tensor<2x2x16x!tt.ptr<i16>>, tensor<2x2x16xi32> 
    %19 = tt.load %18 : tensor<2x2x16x!tt.ptr<i16>> 
    %20 = tt.reshape %19 : tensor<2x2x16xi16> -> tensor<16x4xi16> 
    %21 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %22 = arith.muli %21, %cst : tensor<16x1xi32> 
    %23 = arith.muli %22, %cst : tensor<16x1xi32> 
    %24 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %26 = tt.broadcast %23 : tensor<16x1xi32> -> tensor<16x4xi32> 
    %27 = tt.broadcast %25 : tensor<1x4xi32> -> tensor<16x4xi32> 
    %28 = arith.addi %26, %27 : tensor<16x4xi32> 
    %29 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<16x4x!tt.ptr<i16>> 
    %30 = tt.addptr %29, %28 : tensor<16x4x!tt.ptr<i16>>, tensor<16x4xi32> 
    tt.store %30, %20 : tensor<16x4x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x2x16xi16>, tensor<2xi64>) -> tensor<16x4xi16>

// -----
// uint32
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32>) {
    %cst = arith.constant dense<8> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<16> : tensor<1x8x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<8x1x1xi32> 
    %cst_2 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_2 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst_0 : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x16xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x16xi32> -> tensor<8x8x16xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x16xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x16x!tt.ptr<i32>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x16x!tt.ptr<i32>>, tensor<8x8x16xi32> 
    %19 = tt.load %18 : tensor<8x8x16x!tt.ptr<i32>> 
    %20 = tt.reshape %19 : tensor<8x8x16xi32> -> tensor<16x64xi32> 
    %21 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %22 = arith.muli %21, %cst : tensor<16x1xi32> 
    %23 = arith.muli %22, %cst : tensor<16x1xi32> 
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %26 = tt.broadcast %23 : tensor<16x1xi32> -> tensor<16x64xi32> 
    %27 = tt.broadcast %25 : tensor<1x64xi32> -> tensor<16x64xi32> 
    %28 = arith.addi %26, %27 : tensor<16x64xi32> 
    %29 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x64x!tt.ptr<i32>> 
    %30 = tt.addptr %29, %28 : tensor<16x64x!tt.ptr<i32>>, tensor<16x64xi32> 
    tt.store %30, %20 : tensor<16x64x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<8x8x16xi32>, tensor<2xi64>) -> tensor<16x64xi32>

// -----
// uint64
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> , %arg1: !tt.ptr<i64> )  {
    %cst = arith.constant dense<125> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<16x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<1x125x1xi32> 
    %cst_2 = arith.constant dense<16> : tensor<2x1x1xi32> 
    %cst_3 = arith.constant dense<125> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
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
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %15 = tt.broadcast %12 : tensor<2x125x1xi32> -> tensor<2x125x16xi32> 
    %16 = tt.broadcast %14 : tensor<1x1x16xi32> -> tensor<2x125x16xi32> 
    %17 = arith.addi %15, %16 : tensor<2x125x16xi32> 
    %18 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x125x16x!tt.ptr<i64>> 
    %19 = tt.addptr %18, %17 : tensor<2x125x16x!tt.ptr<i64>>, tensor<2x125x16xi32> 
    %20 = tt.load %19 : tensor<2x125x16x!tt.ptr<i64>> 
    %21 = tt.reshape %20 : tensor<2x125x16xi64> -> tensor<16x250xi64> 
    %22 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %23 = arith.muli %22, %cst_0 : tensor<16x1xi32> 
    %24 = arith.muli %23, %cst : tensor<16x1xi32> 
    %25 = tt.make_range {end = 250 : i32, start = 0 : i32} : tensor<250xi32> 
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<250xi32> -> tensor<1x250xi32> 
    %27 = tt.broadcast %24 : tensor<16x1xi32> -> tensor<16x250xi32> 
    %28 = tt.broadcast %26 : tensor<1x250xi32> -> tensor<16x250xi32> 
    %29 = arith.addi %27, %28 : tensor<16x250xi32> 
    %30 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<16x250x!tt.ptr<i64>> 
    %31 = tt.addptr %30, %29 : tensor<16x250x!tt.ptr<i64>>, tensor<16x250xi32> 
    tt.store %31, %21 : tensor<16x250x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x125x16xi64>, tensor<2xi64>) -> tensor<16x250xi64>

// -----
// bool

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i1> , %arg1: !tt.ptr<i1> ) {
    %cst = arith.constant dense<2> : tensor<16x1xi32> 
    %cst_0 = arith.constant dense<16> : tensor<1x2x1xi32> 
    %cst_1 = arith.constant dense<16> : tensor<2x1x1xi32> 
    %cst_2 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
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
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32> 
    %14 = tt.broadcast %11 : tensor<2x2x1xi32> -> tensor<2x2x16xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x16xi32> -> tensor<2x2x16xi32> 
    %16 = arith.addi %14, %15 : tensor<2x2x16xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<2x2x16x!tt.ptr<i1>> 
    %18 = tt.addptr %17, %16 : tensor<2x2x16x!tt.ptr<i1>>, tensor<2x2x16xi32> 
    %19 = tt.bitcast %18 : tensor<2x2x16x!tt.ptr<i1>> -> tensor<2x2x16x!tt.ptr<i8>> 
    %20 = tt.load %19 : tensor<2x2x16x!tt.ptr<i8>> 
    %21 = tt.reshape %20 : tensor<2x2x16xi8> -> tensor<16x4xi8> 
    %22 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> 
    %23 = arith.muli %22, %cst : tensor<16x1xi32> 
    %24 = arith.muli %23, %cst : tensor<16x1xi32> 
    %25 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %27 = tt.broadcast %24 : tensor<16x1xi32> -> tensor<16x4xi32> 
    %28 = tt.broadcast %26 : tensor<1x4xi32> -> tensor<16x4xi32> 
    %29 = arith.addi %27, %28 : tensor<16x4xi32> 
    %30 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<16x4x!tt.ptr<i1>> 
    %31 = tt.addptr %30, %29 : tensor<16x4x!tt.ptr<i1>>, tensor<16x4xi32> 
    %32 = tt.bitcast %31 : tensor<16x4x!tt.ptr<i1>> -> tensor<16x4x!tt.ptr<i8>> 
    tt.store %32, %21 : tensor<16x4x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x2x16xi8>, tensor<2xi64>) -> tensor<16x4xi8>

// -----
// f8E4M3FN

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<4x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_2 = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_3 = arith.constant dense<8> : tensor<2x1x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %5 = arith.muli %4, %cst_3 : tensor<2x1x1xi32>
    %6 = arith.muli %5, %cst_2 : tensor<2x1x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %9 = arith.muli %8, %cst_1 : tensor<1x8x1xi32>
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
    %21 = tt.reshape %20 : tensor<2x8x4xf8E4M3FN> -> tensor<4x16xf8E4M3FN>
    %22 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %23 = arith.muli %22, %cst_0 : tensor<4x1xi32>
    %24 = arith.muli %23, %cst : tensor<4x1xi32>
    %25 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %27 = tt.broadcast %24 : tensor<4x1xi32> -> tensor<4x16xi32>
    %28 = tt.broadcast %26 : tensor<1x16xi32> -> tensor<4x16xi32>
    %29 = arith.addi %27, %28 : tensor<4x16xi32>
    %30 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x16x!tt.ptr<f8E4M3FN>>
    %31 = tt.addptr %30, %29 : tensor<4x16x!tt.ptr<f8E4M3FN>>, tensor<4x16xi32>
    tt.store %31, %21 : tensor<4x16x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x8x4xf8E4M3FN>, tensor<2xi64>) -> tensor<4x16xf8E4M3FN>


// -----
// f8E5M2
module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<4x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_2 = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_3 = arith.constant dense<8> : tensor<2x1x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %5 = arith.muli %4, %cst_3 : tensor<2x1x1xi32>
    %6 = arith.muli %5, %cst_2 : tensor<2x1x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %9 = arith.muli %8, %cst_1 : tensor<1x8x1xi32>
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
    %21 = tt.reshape %20 : tensor<2x8x4xf8E5M2> -> tensor<4x16xf8E5M2>
    %22 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %23 = arith.muli %22, %cst_0 : tensor<4x1xi32>
    %24 = arith.muli %23, %cst : tensor<4x1xi32>
    %25 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %27 = tt.broadcast %24 : tensor<4x1xi32> -> tensor<4x16xi32>
    %28 = tt.broadcast %26 : tensor<1x16xi32> -> tensor<4x16xi32>
    %29 = arith.addi %27, %28 : tensor<4x16xi32>
    %30 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x16x!tt.ptr<f8E5M2>>
    %31 = tt.addptr %30, %29 : tensor<4x16x!tt.ptr<f8E5M2>>, tensor<4x16xi32>
    tt.store %31, %21 : tensor<4x16x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] = tensor.reshape %[[X:.*]](%[[Y:.*]]) : (tensor<2x8x4xf8E5M2>, tensor<2xi64>) -> tensor<4x16xf8E5M2>