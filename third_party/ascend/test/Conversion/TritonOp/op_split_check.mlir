// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s

// dtype uint64

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64> ) {
    %cst = arith.constant dense<125> : tensor<2x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<1x125x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %cst_2 = arith.constant dense<125> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %4 = arith.muli %3, %cst_2 : tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<2x1x1xi32> 
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<125xi32> -> tensor<1x125xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x125xi32> -> tensor<1x125x1xi32> 
    %8 = arith.muli %7, %cst_0 : tensor<1x125x1xi32> 
    %9 = tt.broadcast %5 : tensor<2x1x1xi32> -> tensor<2x125x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x125x1xi32> -> tensor<2x125x1xi32> 
    %11 = arith.addi %9, %10 : tensor<2x125x1xi32> 
    %12 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %14 = tt.broadcast %11 : tensor<2x125x1xi32> -> tensor<2x125x2xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x2xi32> -> tensor<2x125x2xi32> 
    %16 = arith.addi %14, %15 : tensor<2x125x2xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x125x2x!tt.ptr<i64>> 
    %18 = tt.addptr %17, %16 : tensor<2x125x2x!tt.ptr<i64>>, tensor<2x125x2xi32> 
    %19 = tt.load %18 : tensor<2x125x2x!tt.ptr<i64>> 
    %outLHS, %outRHS = tt.split %19 : tensor<2x125x2xi64> -> tensor<2x125xi64> 
    %20 = arith.muli %2, %cst : tensor<2x1xi32> 
    %21 = tt.broadcast %20 : tensor<2x1xi32> -> tensor<2x125xi32> 
    %22 = tt.broadcast %6 : tensor<1x125xi32> -> tensor<2x125xi32> 
    %23 = arith.addi %21, %22 : tensor<2x125xi32> 
    %24 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x125x!tt.ptr<i64>> 
    %25 = tt.addptr %24, %23 : tensor<2x125x!tt.ptr<i64>>, tensor<2x125xi32> 
    tt.store %25, %outLHS : tensor<2x125x!tt.ptr<i64>> 
    %26 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<2x125x!tt.ptr<i64>> 
    %27 = tt.addptr %26, %23 : tensor<2x125x!tt.ptr<i64>>, tensor<2x125xi32> 
    tt.store %27, %outRHS : tensor<2x125x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>
// CHECK-SAME: %arg2: memref<?xi64>

// CHECK: %extracted_slice_0 = tensor.extract_slice %0[0, 0, 1] [2, 125, 1] [1, 1, 2] : tensor<2x125x2xi64> to tensor<2x125xi64>

// -----
// dtype uint32

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> , %arg2: !tt.ptr<i32> )  {
    %cst = arith.constant dense<8> : tensor<8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_2 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
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
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x2xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x2xi32> -> tensor<8x8x2xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x2xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x2x!tt.ptr<i32>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x2x!tt.ptr<i32>>, tensor<8x8x2xi32> 
    %19 = tt.load %18 : tensor<8x8x2x!tt.ptr<i32>> 
    %outLHS, %outRHS = tt.split %19 : tensor<8x8x2xi32> -> tensor<8x8xi32> 
    %20 = arith.muli %2, %cst : tensor<8x1xi32> 
    %21 = tt.broadcast %20 : tensor<8x1xi32> -> tensor<8x8xi32> 
    %22 = tt.broadcast %6 : tensor<1x8xi32> -> tensor<8x8xi32> 
    %23 = arith.addi %21, %22 : tensor<8x8xi32> 
    %24 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>> 
    %25 = tt.addptr %24, %23 : tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32> 
    tt.store %25, %outLHS : tensor<8x8x!tt.ptr<i32>> 
    %26 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>> 
    %27 = tt.addptr %26, %23 : tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32> 
    tt.store %27, %outRHS : tensor<8x8x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK-SAME: %arg2: memref<?xi32>

// CHECK: %extracted_slice_0 = tensor.extract_slice %0[0, 0, 1] [8, 8, 1] [1, 1, 2] : tensor<8x8x2xi32> to tensor<8x8xi32> 

// -----
// dtype uint16

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> , %arg2: !tt.ptr<i16> )  {
    %cst = arith.constant dense<2> : tensor<2x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<1x2x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %3 = arith.muli %2, %cst_1 : tensor<2x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<2x1x1xi32> 
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32> 
    %7 = arith.muli %6, %cst_0 : tensor<1x2x1xi32> 
    %8 = tt.broadcast %4 : tensor<2x1x1xi32> -> tensor<2x2x1xi32> 
    %9 = tt.broadcast %7 : tensor<1x2x1xi32> -> tensor<2x2x1xi32> 
    %10 = arith.addi %8, %9 : tensor<2x2x1xi32> 
    %11 = tt.expand_dims %5 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %12 = tt.broadcast %10 : tensor<2x2x1xi32> -> tensor<2x2x2xi32> 
    %13 = tt.broadcast %11 : tensor<1x1x2xi32> -> tensor<2x2x2xi32> 
    %14 = arith.addi %12, %13 : tensor<2x2x2xi32> 
    %15 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x2x2x!tt.ptr<i16>> 
    %16 = tt.addptr %15, %14 : tensor<2x2x2x!tt.ptr<i16>>, tensor<2x2x2xi32> 
    %17 = tt.load %16 : tensor<2x2x2x!tt.ptr<i16>> 
    %outLHS, %outRHS = tt.split %17 : tensor<2x2x2xi16> -> tensor<2x2xi16> 
    %18 = arith.muli %1, %cst : tensor<2x1xi32> 
    %19 = tt.broadcast %18 : tensor<2x1xi32> -> tensor<2x2xi32> 
    %20 = tt.broadcast %5 : tensor<1x2xi32> -> tensor<2x2xi32> 
    %21 = arith.addi %19, %20 : tensor<2x2xi32> 
    %22 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x2x!tt.ptr<i16>> 
    %23 = tt.addptr %22, %21 : tensor<2x2x!tt.ptr<i16>>, tensor<2x2xi32> 
    tt.store %23, %outLHS : tensor<2x2x!tt.ptr<i16>> 
    %24 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<2x2x!tt.ptr<i16>> 
    %25 = tt.addptr %24, %21 : tensor<2x2x!tt.ptr<i16>>, tensor<2x2xi32> 
    tt.store %25, %outRHS : tensor<2x2x!tt.ptr<i16>> 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK-SAME: %arg2: memref<?xi16>

// CHECK: %extracted_slice_0 = tensor.extract_slice %0[0, 0, 1] [2, 2, 1] [1, 1, 2] : tensor<2x2x2xi16> to tensor<2x2xi16>

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
// CHECK: %[[REV:.*]] =  memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [2, 8, 4], strides: [32, 4, 1] : memref<?xf8E4M3FN> to memref<2x8x4xf8E4M3FN, strided<[32, 4, 1]>>


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
// CHECK: %[[REV:.*]] =  memref.reinterpret_cast %[[X:.*]] to offset: [0], sizes: [2, 8, 4], strides: [32, 4, 1] : memref<?xf8E5M2> to memref<2x8x4xf8E5M2, strided<[32, 4, 1]>>