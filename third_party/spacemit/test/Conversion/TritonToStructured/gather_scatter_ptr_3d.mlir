// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// Make sure tts.make_gather_scatter_tptr is generated with correct gather_dim stride and offset.

// CHECK-LABEL:   tt.func public @row_gather3(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_6:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<32x!tt.ptr<i32>>
// CHECK:           %[[VAL_7:.*]] = "tts.load"(%[[VAL_6]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<i32>>) -> tensor<32xi32>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_11:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [32, 32, 32] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_7]], strides: {{\[}}%[[VAL_8]], %[[VAL_10]], %[[VAL_9]]], offsets: [0, 0, 0] : tensor<32xi32> <f32> to !tt.ptr<tensor<32x32x32xf32>>
// CHECK:           %[[VAL_12:.*]] = "tts.load"(%[[VAL_11]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<32x32x32xf32>>) -> tensor<32x32x32xf32>
// CHECK:           %[[VAL_13:.*]] = tts.make_gather_scatter_tptr %[[VAL_2]] to sizes: [32, 32, 32] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_7]], strides: {{\[}}%[[VAL_8]], %[[VAL_10]], %[[VAL_9]]], offsets: [0, 0, 0] : tensor<32xi32> <f32> to !tt.ptr<tensor<32x32x32xf32>>
// CHECK:           "tts.store"(%[[VAL_13]], %[[VAL_12]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<32x32x32xf32>>, tensor<32x32x32xf32>) -> ()
// CHECK:           tt.return

module attributes {} {
  tt.func public @row_gather3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<i32>>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<32x1xi32> -> tensor<32x1x1xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<32x1x1xi32>
    %7 = arith.muli %5, %6 : tensor<32x1x1xi32>
    %8 = tt.expand_dims %3 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %9 = tt.expand_dims %8 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %10 = tt.splat %arg4 : i32 -> tensor<1x32x1xi32>
    %11 = arith.muli %9, %10 : tensor<1x32x1xi32>
    %12 = tt.broadcast %7 : tensor<32x1x1xi32> -> tensor<32x32x1xi32>
    %13 = tt.broadcast %11 : tensor<1x32x1xi32> -> tensor<32x32x1xi32>
    %14 = arith.addi %12, %13 : tensor<32x32x1xi32>
    %15 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32>
    %17 = tt.splat %arg5 : i32 -> tensor<1x1x32xi32>
    %18 = arith.muli %16, %17 : tensor<1x1x32xi32>
    %19 = tt.broadcast %14 : tensor<32x32x1xi32> -> tensor<32x32x32xi32>
    %20 = tt.broadcast %18 : tensor<1x1x32xi32> -> tensor<32x32x32xi32>
    %21 = arith.addi %19, %20 : tensor<32x32x32xi32>
    %22 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x32x!tt.ptr<f32>>
    %23 = tt.addptr %22, %21 : tensor<32x32x32x!tt.ptr<f32>>, tensor<32x32x32xi32>
    %24 = tt.load %23 : tensor<32x32x32x!tt.ptr<f32>>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x32x!tt.ptr<f32>>
    %26 = tt.addptr %25, %21 : tensor<32x32x32x!tt.ptr<f32>>, tensor<32x32x32xi32>
    tt.store %26, %24 : tensor<32x32x32x!tt.ptr<f32>>
    tt.return
  }
}
