// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s

// dtype  bfloat16
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<bf16> , %arg2: !tt.ptr<bf16> )  {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %cst_2 = arith.constant dense<8> : tensor<8x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %2 = arith.muli %1, %cst_2 : tensor<8x1xi32> 
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.broadcast %2 : tensor<8x1xi32> -> tensor<8x8xi32> 
    %5 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<8x8xi32> 
    %6 = arith.addi %4, %5 : tensor<8x8xi32> 
    %7 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x8x!tt.ptr<bf16>> 
    %8 = tt.addptr %7, %6 : tensor<8x8x!tt.ptr<bf16>>, tensor<8x8xi32> 
    %9 = tt.load %8 : tensor<8x8x!tt.ptr<bf16>> 
    %10 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<8x8x!tt.ptr<bf16>> 
    %11 = tt.addptr %10, %6 : tensor<8x8x!tt.ptr<bf16>>, tensor<8x8xi32> 
    %12 = tt.load %11 : tensor<8x8x!tt.ptr<bf16>> 
    %13 = tt.join %9, %12 : tensor<8x8xbf16> -> tensor<8x8x2xbf16> 
    %14 = tt.expand_dims %1 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %15 = arith.muli %14, %cst_1 : tensor<8x1x1xi32> 
    %16 = arith.muli %15, %cst_0 : tensor<8x1x1xi32> 
    %17 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %18 = arith.muli %17, %cst : tensor<1x8x1xi32> 
    %19 = tt.broadcast %16 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %20 = tt.broadcast %18 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %21 = arith.addi %19, %20 : tensor<8x8x1xi32> 
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %25 = tt.broadcast %21 : tensor<8x8x1xi32> -> tensor<8x8x2xi32> 
    %26 = tt.broadcast %24 : tensor<1x1x2xi32> -> tensor<8x8x2xi32> 
    %27 = arith.addi %25, %26 : tensor<8x8x2xi32> 
    %28 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x8x2x!tt.ptr<bf16>> 
    %29 = tt.addptr %28, %27 : tensor<8x8x2x!tt.ptr<bf16>>, tensor<8x8x2xi32> 
    tt.store %29, %13 : tensor<8x8x2x!tt.ptr<bf16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xbf16>
// CHECK-SAME: %arg1: memref<?xbf16>
// CHECK:  %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [8, 8, 1] [1, 1, 2] : tensor<8x8xbf16> into tensor<8x8x2xbf16>

// -----
// uint8
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8> , %arg2: !tt.ptr<i8>)  {
    %cst = arith.constant dense<2> : tensor<1x256x1xi32> 
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32> 
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32> 
    %2 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x256x!tt.ptr<i8>> 
    %3 = tt.addptr %2, %1 : tensor<1x256x!tt.ptr<i8>>, tensor<1x256xi32> 
    %4 = tt.load %3 : tensor<1x256x!tt.ptr<i8>> 
    %5 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<1x256x!tt.ptr<i8>> 
    %6 = tt.addptr %5, %1 : tensor<1x256x!tt.ptr<i8>>, tensor<1x256xi32> 
    %7 = tt.load %6 : tensor<1x256x!tt.ptr<i8>> 
    %8 = tt.join %4, %7 : tensor<1x256xi8> -> tensor<1x256x2xi8> 
    %9 = tt.expand_dims %1 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32> 
    %10 = arith.muli %9, %cst : tensor<1x256x1xi32> 
    %11 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %14 = tt.broadcast %10 : tensor<1x256x1xi32> -> tensor<1x256x2xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x2xi32> -> tensor<1x256x2xi32> 
    %16 = arith.addi %14, %15 : tensor<1x256x2xi32> 
    %17 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x256x2x!tt.ptr<i8>> 
    %18 = tt.addptr %17, %16 : tensor<1x256x2x!tt.ptr<i8>>, tensor<1x256x2xi32> 
    tt.store %18, %8 : tensor<1x256x2x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [1, 256, 1] [1, 1, 2] : tensor<1x256xi8> into tensor<1x256x2xi8>

// -----
// uint16
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> , %arg2: !tt.ptr<i16> )  {
    %cst = arith.constant dense<2> : tensor<1x2x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<2x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %2 = arith.muli %1, %cst_1 : tensor<2x1xi32> 
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %4 = tt.broadcast %2 : tensor<2x1xi32> -> tensor<2x2xi32> 
    %5 = tt.broadcast %3 : tensor<1x2xi32> -> tensor<2x2xi32> 
    %6 = arith.addi %4, %5 : tensor<2x2xi32> 
    %7 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x2x!tt.ptr<i16>> 
    %8 = tt.addptr %7, %6 : tensor<2x2x!tt.ptr<i16>>, tensor<2x2xi32> 
    %9 = tt.load %8 : tensor<2x2x!tt.ptr<i16>> 
    %10 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<2x2x!tt.ptr<i16>> 
    %11 = tt.addptr %10, %6 : tensor<2x2x!tt.ptr<i16>>, tensor<2x2xi32> 
    %12 = tt.load %11 : tensor<2x2x!tt.ptr<i16>> 
    %13 = tt.join %9, %12 : tensor<2x2xi16> -> tensor<2x2x2xi16> 
    %14 = tt.expand_dims %1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %15 = arith.muli %14, %cst_0 : tensor<2x1x1xi32> 
    %16 = arith.muli %15, %cst_0 : tensor<2x1x1xi32> 
    %17 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32> 
    %18 = arith.muli %17, %cst : tensor<1x2x1xi32> 
    %19 = tt.broadcast %16 : tensor<2x1x1xi32> -> tensor<2x2x1xi32> 
    %20 = tt.broadcast %18 : tensor<1x2x1xi32> -> tensor<2x2x1xi32> 
    %21 = arith.addi %19, %20 : tensor<2x2x1xi32> 
    %22 = tt.expand_dims %3 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %23 = tt.broadcast %21 : tensor<2x2x1xi32> -> tensor<2x2x2xi32> 
    %24 = tt.broadcast %22 : tensor<1x1x2xi32> -> tensor<2x2x2xi32> 
    %25 = arith.addi %23, %24 : tensor<2x2x2xi32> 
    %26 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x2x2x!tt.ptr<i16>> 
    %27 = tt.addptr %26, %25 : tensor<2x2x2x!tt.ptr<i16>>, tensor<2x2x2xi32> 
    tt.store %27, %13 : tensor<2x2x2x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [2, 2, 1] [1, 1, 2] : tensor<2x2xi16> into tensor<2x2x2xi16>

// -----
// uint32
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)  {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %cst_2 = arith.constant dense<8> : tensor<8x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %2 = arith.muli %1, %cst_2 : tensor<8x1xi32> 
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %4 = tt.broadcast %2 : tensor<8x1xi32> -> tensor<8x8xi32> 
    %5 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<8x8xi32> 
    %6 = arith.addi %4, %5 : tensor<8x8xi32> 
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>> 
    %8 = tt.addptr %7, %6 : tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32> 
    %9 = tt.load %8 : tensor<8x8x!tt.ptr<i32>> 
    %10 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>> 
    %11 = tt.addptr %10, %6 : tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32> 
    %12 = tt.load %11 : tensor<8x8x!tt.ptr<i32>> 
    %13 = tt.join %9, %12 : tensor<8x8xi32> -> tensor<8x8x2xi32> 
    %14 = tt.expand_dims %1 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %15 = arith.muli %14, %cst_1 : tensor<8x1x1xi32> 
    %16 = arith.muli %15, %cst_0 : tensor<8x1x1xi32> 
    %17 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %18 = arith.muli %17, %cst : tensor<1x8x1xi32> 
    %19 = tt.broadcast %16 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %20 = tt.broadcast %18 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %21 = arith.addi %19, %20 : tensor<8x8x1xi32> 
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %25 = tt.broadcast %21 : tensor<8x8x1xi32> -> tensor<8x8x2xi32> 
    %26 = tt.broadcast %24 : tensor<1x1x2xi32> -> tensor<8x8x2xi32> 
    %27 = arith.addi %25, %26 : tensor<8x8x2xi32> 
    %28 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x2x!tt.ptr<i32>> 
    %29 = tt.addptr %28, %27 : tensor<8x8x2x!tt.ptr<i32>>, tensor<8x8x2xi32> 
    tt.store %29, %13 : tensor<8x8x2x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [8, 8, 1] [1, 1, 2] : tensor<8x8xi32> into tensor<8x8x2xi32>

// -----
// uint64

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> , %arg1: !tt.ptr<i64> , %arg2: !tt.ptr<i64> )  {
    %cst = arith.constant dense<2> : tensor<1x125x1xi32> 
    %cst_0 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<125> : tensor<2x1x1xi32> 
    %cst_2 = arith.constant dense<125> : tensor<2x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 125 : i32, start = 0 : i32} : tensor<125xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %3 = arith.muli %2, %cst_2 : tensor<2x1xi32> 
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<125xi32> -> tensor<1x125xi32> 
    %5 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x125xi32> 
    %6 = tt.broadcast %4 : tensor<1x125xi32> -> tensor<2x125xi32> 
    %7 = arith.addi %5, %6 : tensor<2x125xi32> 
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x125x!tt.ptr<i64>> 
    %9 = tt.addptr %8, %7 : tensor<2x125x!tt.ptr<i64>>, tensor<2x125xi32> 
    %10 = tt.load %9 : tensor<2x125x!tt.ptr<i64>> 
    %11 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<2x125x!tt.ptr<i64>> 
    %12 = tt.addptr %11, %7 : tensor<2x125x!tt.ptr<i64>>, tensor<2x125xi32> 
    %13 = tt.load %12 : tensor<2x125x!tt.ptr<i64>> 
    %14 = tt.join %10, %13 : tensor<2x125xi64> -> tensor<2x125x2xi64> 
    %15 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %16 = arith.muli %15, %cst_1 : tensor<2x1x1xi32> 
    %17 = arith.muli %16, %cst_0 : tensor<2x1x1xi32> 
    %18 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x125xi32> -> tensor<1x125x1xi32> 
    %19 = arith.muli %18, %cst : tensor<1x125x1xi32> 
    %20 = tt.broadcast %17 : tensor<2x1x1xi32> -> tensor<2x125x1xi32> 
    %21 = tt.broadcast %19 : tensor<1x125x1xi32> -> tensor<2x125x1xi32> 
    %22 = arith.addi %20, %21 : tensor<2x125x1xi32> 
    %23 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %25 = tt.broadcast %22 : tensor<2x125x1xi32> -> tensor<2x125x2xi32> 
    %26 = tt.broadcast %24 : tensor<1x1x2xi32> -> tensor<2x125x2xi32> 
    %27 = arith.addi %25, %26 : tensor<2x125x2xi32> 
    %28 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x125x2x!tt.ptr<i64>> 
    %29 = tt.addptr %28, %27 : tensor<2x125x2x!tt.ptr<i64>>, tensor<2x125x2xi32> 
    tt.store %29, %14 : tensor<2x125x2x!tt.ptr<i64>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [2, 125, 1] [1, 1, 2] : tensor<2x125xi64> into tensor<2x125x2xi64>

// -----
// bool
module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i1> , %arg1: !tt.ptr<i1> , %arg2: !tt.ptr<i1> )  {
    %c0_i32 = arith.constant 0 : i32 
    %0 = tt.addptr %arg1, %c0_i32 : !tt.ptr<i1>, i32 
    %1 = tt.bitcast %0 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %2 = tt.splat %1 : !tt.ptr<i8> -> tensor<1x1x!tt.ptr<i8>> 
    %3 = tt.load %2 : tensor<1x1x!tt.ptr<i8>> 
    %4 = tt.addptr %arg2, %c0_i32 : !tt.ptr<i1>, i32 
    %5 = tt.bitcast %4 : !tt.ptr<i1> -> !tt.ptr<i8> 
    %6 = tt.splat %5 : !tt.ptr<i8> -> tensor<1x1x!tt.ptr<i8>> 
    %7 = tt.load %6 : tensor<1x1x!tt.ptr<i8>> 
    %8 = tt.join %3, %7 : tensor<1x1xi8> -> tensor<1x1x2xi8> 
    %9 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32> 
    %12 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1x1x2x!tt.ptr<i1>> 
    %13 = tt.addptr %12, %11 : tensor<1x1x2x!tt.ptr<i1>>, tensor<1x1x2xi32> 
    %14 = tt.bitcast %13 : tensor<1x1x2x!tt.ptr<i1>> -> tensor<1x1x2x!tt.ptr<i8>> 
    tt.store %14, %8 : tensor<1x1x2x!tt.ptr<i8>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_npu_
// CHECK-SAME: %arg0: memref<?xi8>
// CHECK-SAME: %arg1: memref<?xi8>
// CHECK: %[[REV:.*]]  = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 1] [1, 1, 1] [1, 1, 2] : tensor<1x1xi8> into tensor<1x1x2xi8>

// -----
// f8E4M3FN
module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %cst_2 = arith.constant dense<8> : tensor<8x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %2 = arith.muli %1, %cst_2 : tensor<8x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %4 = tt.broadcast %2 : tensor<8x1xi32> -> tensor<8x8xi32>
    %5 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<8x8xi32>
    %6 = arith.addi %4, %5 : tensor<8x8xi32>
    %7 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<8x8x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %6 : tensor<8x8x!tt.ptr<f8E4M3FN>>, tensor<8x8xi32>
    %9 = tt.load %8 : tensor<8x8x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<8x8x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %6 : tensor<8x8x!tt.ptr<f8E4M3FN>>, tensor<8x8xi32>
    %12 = tt.load %11 : tensor<8x8x!tt.ptr<f8E4M3FN>>
    %13 = tt.join %9, %12 : tensor<8x8xf8E4M3FN> -> tensor<8x8x2xf8E4M3FN>
    %14 = tt.expand_dims %1 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %15 = arith.muli %14, %cst_1 : tensor<8x1x1xi32>
    %16 = arith.muli %15, %cst_0 : tensor<8x1x1xi32>
    %17 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %18 = arith.muli %17, %cst : tensor<1x8x1xi32>
    %19 = tt.broadcast %16 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %20 = tt.broadcast %18 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %21 = arith.addi %19, %20 : tensor<8x8x1xi32>
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
    %25 = tt.broadcast %21 : tensor<8x8x1xi32> -> tensor<8x8x2xi32>
    %26 = tt.broadcast %24 : tensor<1x1x2xi32> -> tensor<8x8x2xi32>
    %27 = arith.addi %25, %26 : tensor<8x8x2xi32>
    %28 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x8x2x!tt.ptr<f8E4M3FN>>
    %29 = tt.addptr %28, %27 : tensor<8x8x2x!tt.ptr<f8E4M3FN>>, tensor<8x8x2xi32>
    tt.store %29, %13 : tensor<8x8x2x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 0] [8, 8, 1] [1, 1, 2] : tensor<8x8xf8E4M3FN> into tensor<8x8x2xf8E4M3FN>

// -----
// f8E5M2

module {
  tt.func public @fn_npu_dtype(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x256x1xi32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %2 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<1x256x!tt.ptr<f8E5M2>>
    %3 = tt.addptr %2, %1 : tensor<1x256x!tt.ptr<f8E5M2>>, tensor<1x256xi32>
    %4 = tt.load %3 : tensor<1x256x!tt.ptr<f8E5M2>>
    %5 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<1x256x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %1 : tensor<1x256x!tt.ptr<f8E5M2>>, tensor<1x256xi32>
    %7 = tt.load %6 : tensor<1x256x!tt.ptr<f8E5M2>>
    %8 = tt.join %4, %7 : tensor<1x256xf8E5M2> -> tensor<1x256x2xf8E5M2>
    %9 = tt.expand_dims %1 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32>
    %10 = arith.muli %9, %cst : tensor<1x256x1xi32>
    %11 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
    %14 = tt.broadcast %10 : tensor<1x256x1xi32> -> tensor<1x256x2xi32>
    %15 = tt.broadcast %13 : tensor<1x1x2xi32> -> tensor<1x256x2xi32>
    %16 = arith.addi %14, %15 : tensor<1x256x2xi32>
    %17 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<1x256x2x!tt.ptr<f8E5M2>>
    %18 = tt.addptr %17, %16 : tensor<1x256x2x!tt.ptr<f8E5M2>>, tensor<1x256x2xi32>
    tt.store %18, %8 : tensor<1x256x2x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_dtype
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] = tensor.insert_slice %[[X:.*]] into %[[Y:.*]][0, 0, 0] [1, 256, 1] [1, 1, 2] : tensor<1x256xf8E5M2> into tensor<1x256x2xf8E5M2>