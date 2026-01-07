// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s
// uint16

module {
  tt.func public @fn_broadcast(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> )  {
    %cst = arith.constant dense<4> : tensor<2x1x1xi32> 
    %cst_0 = arith.constant dense<8> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<1x4x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<1x4x1xi32> 
    %6 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %8 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<1x4x8xi32> 
    %9 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<1x4x8xi32> 
    %10 = arith.addi %8, %9 : tensor<1x4x8xi32> 
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32> 
    %14 = arith.muli %13, %cst : tensor<2x1x1xi32> 
    %15 = tt.broadcast %14 : tensor<2x1x1xi32> -> tensor<2x4x1xi32> 
    %16 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<2x4x1xi32> 
    %17 = arith.addi %15, %16 : tensor<2x4x1xi32> 
    %18 = tt.broadcast %17 : tensor<2x4x1xi32> -> tensor<2x4x8xi32> 
    %19 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<2x4x8xi32> 
    %20 = arith.addi %18, %19 : tensor<2x4x8xi32> 
    %21 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x4x8x!tt.ptr<i16>> 
    %22 = tt.addptr %21, %10 : tensor<1x4x8x!tt.ptr<i16>>, tensor<1x4x8xi32> 
    %23 = tt.load %22 : tensor<1x4x8x!tt.ptr<i16>> 
    %24 = tt.broadcast %23 : tensor<1x4x8xi16> -> tensor<2x4x8xi16> 
    %25 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x4x8x!tt.ptr<i16>> 
    %26 = tt.addptr %25, %20 : tensor<2x4x8x!tt.ptr<i16>>, tensor<2x4x8xi32> 
    tt.store %26, %24 : tensor<2x4x8x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK: %[[REV:.*]] = linalg.broadcast ins(%[[X:.*]] : tensor<4x8xi16>) outs(%[[Y:.*]] : tensor<2x4x8xi16>) dimensions = [0]

// -----
// uint32

module {
  tt.func public @fn_broadcast(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> )  {
    %cst = arith.constant dense<4> : tensor<2x1x1xi32> 
    %cst_0 = arith.constant dense<8> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<1x4x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32> 
    %5 = arith.muli %4, %cst_1 : tensor<1x4x1xi32> 
    %6 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32> 
    %8 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<1x4x8xi32> 
    %9 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<1x4x8xi32> 
    %10 = arith.addi %8, %9 : tensor<1x4x8xi32> 
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> 
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32> 
    %14 = arith.muli %13, %cst : tensor<2x1x1xi32> 
    %15 = tt.broadcast %14 : tensor<2x1x1xi32> -> tensor<2x4x1xi32> 
    %16 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<2x4x1xi32> 
    %17 = arith.addi %15, %16 : tensor<2x4x1xi32> 
    %18 = tt.broadcast %17 : tensor<2x4x1xi32> -> tensor<2x4x8xi32> 
    %19 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<2x4x8xi32> 
    %20 = arith.addi %18, %19 : tensor<2x4x8xi32> 
    %21 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x4x8x!tt.ptr<i32>> 
    %22 = tt.addptr %21, %10 : tensor<1x4x8x!tt.ptr<i32>>, tensor<1x4x8xi32> 
    %23 = tt.load %22 : tensor<1x4x8x!tt.ptr<i32>> 
    %24 = tt.broadcast %23 : tensor<1x4x8xi32> -> tensor<2x4x8xi32> 
    %25 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x4x8x!tt.ptr<i32>> 
    %26 = tt.addptr %25, %20 : tensor<2x4x8x!tt.ptr<i32>>, tensor<2x4x8xi32> 
    tt.store %26, %24 : tensor<2x4x8x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK: %[[REV:.*]] = linalg.broadcast ins(%[[X:.*]] : tensor<4x8xi32>) outs(%[[Y:.*]] : tensor<2x4x8xi32>) dimensions = [0]

// -----
// f8E4M3FN

module {
  tt.func public @fn_broadcast(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_0 = arith.constant dense<8> : tensor<2x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<1x4x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<1x4x1xi32>
    %6 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %8 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<1x4x8xi32>
    %9 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<1x4x8xi32>
    %10 = arith.addi %8, %9 : tensor<1x4x8xi32>
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
    %14 = arith.muli %13, %cst : tensor<2x1x1xi32>
    %15 = tt.broadcast %14 : tensor<2x1x1xi32> -> tensor<2x4x1xi32>
    %16 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<2x4x1xi32>
    %17 = arith.addi %15, %16 : tensor<2x4x1xi32>
    %18 = tt.broadcast %17 : tensor<2x4x1xi32> -> tensor<2x4x8xi32>
    %19 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<2x4x8xi32>
    %20 = arith.addi %18, %19 : tensor<2x4x8xi32>
    %21 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<1x4x8x!tt.ptr<f8E4M3FN>>
    %22 = tt.addptr %21, %10 : tensor<1x4x8x!tt.ptr<f8E4M3FN>>, tensor<1x4x8xi32>
    %23 = tt.load %22 : tensor<1x4x8x!tt.ptr<f8E4M3FN>>
    %24 = tt.broadcast %23 : tensor<1x4x8xf8E4M3FN> -> tensor<2x4x8xf8E4M3FN>
    %25 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<2x4x8x!tt.ptr<f8E4M3FN>>
    %26 = tt.addptr %25, %20 : tensor<2x4x8x!tt.ptr<f8E4M3FN>>, tensor<2x4x8xi32>
    tt.store %26, %24 : tensor<2x4x8x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] =  linalg.broadcast ins(%[[X:.*]] : tensor<4x8xf8E4M3FN>) outs(%[[Y:.*]] : tensor<2x4x8xf8E4M3FN>) dimensions = [0]

// -----
// f8E5M2

module {
  tt.func public @fn_broadcast(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<2x1x1xi32>
    %cst_0 = arith.constant dense<8> : tensor<2x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<1x4x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<1x4x1xi32>
    %6 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %8 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<1x4x8xi32>
    %9 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<1x4x8xi32>
    %10 = arith.addi %8, %9 : tensor<1x4x8xi32>
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
    %14 = arith.muli %13, %cst : tensor<2x1x1xi32>
    %15 = tt.broadcast %14 : tensor<2x1x1xi32> -> tensor<2x4x1xi32>
    %16 = tt.broadcast %5 : tensor<1x4x1xi32> -> tensor<2x4x1xi32>
    %17 = arith.addi %15, %16 : tensor<2x4x1xi32>
    %18 = tt.broadcast %17 : tensor<2x4x1xi32> -> tensor<2x4x8xi32>
    %19 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<2x4x8xi32>
    %20 = arith.addi %18, %19 : tensor<2x4x8xi32>
    %21 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<1x4x8x!tt.ptr<f8E5M2>>
    %22 = tt.addptr %21, %10 : tensor<1x4x8x!tt.ptr<f8E5M2>>, tensor<1x4x8xi32>
    %23 = tt.load %22 : tensor<1x4x8x!tt.ptr<f8E5M2>>
    %24 = tt.broadcast %23 : tensor<1x4x8xf8E5M2> -> tensor<2x4x8xf8E5M2>
    %25 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<2x4x8x!tt.ptr<f8E5M2>>
    %26 = tt.addptr %25, %20 : tensor<2x4x8x!tt.ptr<f8E5M2>>, tensor<2x4x8xi32>
    tt.store %26, %24 : tensor<2x4x8x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] =  linalg.broadcast ins(%[[X:.*]] : tensor<4x8xf8E5M2>) outs(%[[Y:.*]] : tensor<2x4x8xf8E5M2>) dimensions = [0]