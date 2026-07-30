// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure i64 offset works.
// CHECK-LABEL:   tt.func public @index_select_row_kernel(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                            %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                            %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i64> to tensor<4x!tt.ptr<i64>>
// CHECK:           %[[VAL_5:.*]] = "tts.load"(%[[VAL_4]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<i64>>) -> tensor<4xi64>
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_7:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_5]], strides: {{\[}}%[[VAL_6]], 1], offsets: [0, 0] : tensor<4xi64> <f32> to !tt.ptr<tensor<4x16xf32>>
// CHECK:           %[[VAL_8:.*]] = "tts.load"(%[[VAL_7]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x16xf32>>) -> tensor<4x16xf32>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [4, 16], strides: {{\[}}%[[VAL_9]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<4x16x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[VAL_10]], %[[VAL_8]]) <{static_mask_dims = array<i64>}> : (tensor<4x16x!tt.ptr<f32>>, tensor<4x16xf32>) -> ()
// CHECK:           tt.return

module {
  tt.func public @index_select_row_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg3: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %3 = tt.load %2 : tensor<4x!tt.ptr<i64>>
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi64> -> tensor<4x1xi64>
    %6 = arith.extsi %arg3 : i32 to i64
    %7 = tt.splat %6 : i64 -> tensor<4x1xi64>
    %8 = arith.muli %5, %7 : tensor<4x1xi64>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi64>
    %11 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.broadcast %10 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %13 = tt.broadcast %11 : tensor<1x16xi32> -> tensor<4x16xi32>
    %14 = tt.addptr %12, %13 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %15 = tt.load %14 : tensor<4x16x!tt.ptr<f32>>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %16, %17 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %22 = tt.addptr %21, %13 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    tt.store %22, %15 : tensor<4x16x!tt.ptr<f32>>
    tt.return
  }
}
