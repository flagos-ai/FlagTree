// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure tts.make_gather_scatter_tptr was generated correctly.

// CHECK-LABEL:   tt.func public @index_select_row_with_double_mod_kernel2(
// CHECK-SAME:                                                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                             %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                             %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                             %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                             %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                             %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<1> : tensor<4x1xi32>
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
// CHECK:           %[[VAL_9:.*]] = "tts.load"(%[[VAL_8]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<i32>>) -> tensor<4xi32>
// CHECK:           %[[VAL_10:.*]] = tt.expand_dims %[[VAL_7]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK:           %[[VAL_11:.*]] = tt.splat %[[VAL_5]] : i32 -> tensor<4x1xi32>
// CHECK:           %[[VAL_12:.*]] = arith.remsi %[[VAL_10]], %[[VAL_11]] : tensor<4x1xi32>
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_6]] : tensor<4x1xi32>
// CHECK:           %[[VAL_14:.*]] = tt.reshape %[[VAL_13]] allow_reorder : tensor<4x1xi32> -> tensor<4xi32>
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_9]], %[[VAL_14]] : tensor<4xi32>
// CHECK:           %[[VAL_16:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_17:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_15]], strides: {{\[}}%[[VAL_16]], 1], offsets: [0, 0] : tensor<4xi32> <f32> to !tt.ptr<tensor<4x16xf32>>
// CHECK:           %[[VAL_18:.*]] = "tts.load"(%[[VAL_17]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x16xf32>>) -> tensor<4x16xf32>
// CHECK:           %[[VAL_19:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_20:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [4, 16], strides: {{\[}}%[[VAL_19]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<4x16x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[VAL_20]], %[[VAL_18]]) <{static_mask_dims = array<i64>}> : (tensor<4x16x!tt.ptr<f32>>, tensor<4x16xf32>) -> ()
// CHECK:           tt.return

module {
  tt.func public @index_select_row_with_double_mod_kernel2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<4x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %3 = tt.load %2 : tensor<4x!tt.ptr<i32>>
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %7 = tt.splat %arg5 : i32 -> tensor<4x1xi32>
    %8 = arith.remsi %6, %7 : tensor<4x1xi32>
    %9 = arith.addi %8, %cst : tensor<4x1xi32>
    %10 = arith.addi %5, %9 : tensor<4x1xi32>
    %11 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %12 = arith.muli %10, %11 : tensor<4x1xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %14 = tt.addptr %13, %12 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %15 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %16 = tt.broadcast %14 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %17 = tt.broadcast %15 : tensor<1x16xi32> -> tensor<4x16xi32>
    %18 = tt.addptr %16, %17 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %19 = tt.load %18 : tensor<4x16x!tt.ptr<f32>>
    %20 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %21 = arith.muli %6, %20 : tensor<4x1xi32>
    %22 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %23 = tt.addptr %22, %21 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %24 = tt.broadcast %23 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %25 = tt.addptr %24, %17 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    tt.store %25, %19 : tensor<4x16x!tt.ptr<f32>>
    tt.return
  }
}
