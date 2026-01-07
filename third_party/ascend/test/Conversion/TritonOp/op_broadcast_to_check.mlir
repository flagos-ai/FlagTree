// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s

// uint16

module {
  tt.func public @fn_broadcast_to(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i16> , %arg2: i32 ) {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %1 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>> 
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i16>>, tensor<16xi32> 
    %3 = tt.load %2 : tensor<16x!tt.ptr<i16>> 
    %4 = tt.reshape %3 : tensor<16xi16> -> tensor<2x1x8xi16> 
    %5 = tt.broadcast %4 : tensor<2x1x8xi16> -> tensor<2x4x8xi16> 
    %6 = tt.reshape %5 : tensor<2x4x8xi16> -> tensor<64xi16> 
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %8 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<64x!tt.ptr<i16>> 
    %9 = tt.addptr %8, %7 : tensor<64x!tt.ptr<i16>>, tensor<64xi32> 
    tt.store %9, %6 : tensor<64x!tt.ptr<i16>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_broadcast_to
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK:  %[[REV:.*]] = linalg.broadcast ins(%[[X:.*]] : tensor<2x8xi16>) outs(%[[Y:.*]] : tensor<2x4x8xi16>) dimensions = [1]

// -----
// uint32

module {
  tt.func public @fn_broadcast_to(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32> , %arg2: i32 )  {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> 
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> 
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>> 
    %4 = tt.reshape %3 : tensor<16xi32> -> tensor<2x1x8xi32> 
    %5 = tt.broadcast %4 : tensor<2x1x8xi32> -> tensor<2x4x8xi32> 
    %6 = tt.reshape %5 : tensor<2x4x8xi32> -> tensor<64xi32> 
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %8 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>> 
    %9 = tt.addptr %8, %7 : tensor<64x!tt.ptr<i32>>, tensor<64xi32> 
    tt.store %9, %6 : tensor<64x!tt.ptr<i32>> 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @fn_broadcast_to
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// CHECK:  %[[REV:.*]] = linalg.broadcast ins(%[[X:.*]] : tensor<2x8xi32>) outs(%[[Y:.*]] : tensor<2x4x8xi32>) dimensions = [1]

// -----

module {
  tt.func public @fn_broadcast_to(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} , %arg2: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<16x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f8E4M3FN>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f8E4M3FN>>
    %4 = tt.reshape %3 : tensor<16xf8E4M3FN> -> tensor<2x1x8xf8E4M3FN>
    %5 = tt.broadcast %4 : tensor<2x1x8xf8E4M3FN> -> tensor<2x4x8xf8E4M3FN>
    %6 = tt.reshape %5 : tensor<2x4x8xf8E4M3FN> -> tensor<64xf8E4M3FN>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %7 : tensor<64x!tt.ptr<f8E4M3FN>>, tensor<64xi32>
    tt.store %9, %6 : tensor<64x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xf8E4M3FN>
// CHECK-SAME: %arg1: memref<?xf8E4M3FN>
// CHECK: %[[REV:.*]] =  linalg.broadcast ins(%[[X:.*]] : tensor<2x8xf8E4M3FN>) outs(%[[Y:.*]] : tensor<2x4x8xf8E4M3FN>) dimensions = [1]

// -----

module {
  tt.func public @fn_broadcast_to(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} , %arg2: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<16x!tt.ptr<f8E5M2>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f8E5M2>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f8E5M2>>
    %4 = tt.reshape %3 : tensor<16xf8E5M2> -> tensor<2x1x8xf8E5M2>
    %5 = tt.broadcast %4 : tensor<2x1x8xf8E5M2> -> tensor<2x4x8xf8E5M2>
    %6 = tt.reshape %5 : tensor<2x4x8xf8E5M2> -> tensor<64xf8E5M2>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<64x!tt.ptr<f8E5M2>>
    %9 = tt.addptr %8, %7 : tensor<64x!tt.ptr<f8E5M2>>, tensor<64xi32>
    tt.store %9, %6 : tensor<64x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast
// CHECK-SAME: %arg0: memref<?xf8E5M2>
// CHECK-SAME: %arg1: memref<?xf8E5M2>
// CHECK: %[[REV:.*]] =  linalg.broadcast ins(%[[X:.*]] : tensor<2x8xf8E5M2>) outs(%[[Y:.*]] : tensor<2x4x8xf8E5M2>) dimensions = [1]