// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

// Make sure tts.make_gather_scatter_tptr is generated with correct offset, stride and gather_dim.

// CHECK-LABEL:   tt.func public @index_select_row_with_double_mod_kernel(
// CHECK-SAME:                                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                            %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                            %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                            %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                            %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                            %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
// CHECK:           %[[VAL_9:.*]] = "tts.load"(%[[VAL_8]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<i32>>) -> tensor<4xi32>
// CHECK:           %[[VAL_10:.*]] = tt.expand_dims %[[VAL_9]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK:           %[[VAL_11:.*]] = tt.expand_dims %[[VAL_7]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_5]] : i32 -> tensor<4x1xi32>
// CHECK:           %[[VAL_13:.*]] = arith.remsi %[[VAL_11]], %[[VAL_12]] : tensor<4x1xi32>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] : tensor<4x1xi32>
// CHECK:           %[[VAL_15:.*]] = tt.splat %[[VAL_6]] : i32 -> tensor<4x1xi32>
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_14]], %[[VAL_15]] : tensor<4x1xi32>
// CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_18:.*]] = tensor.collapse_shape %[[VAL_16]] {{\[\[}}0, 1]] : tensor<4x1xi32> into tensor<4xi32>
// CHECK:           %[[VAL_19:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_18]], strides: {{\[}}%[[VAL_17]], 1], offsets: [0, 0] : tensor<4xi32> <f32> to !tt.ptr<tensor<4x16xf32>>
// CHECK:           %[[VAL_20:.*]] = "tts.load"(%[[VAL_19]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x16xf32>>) -> tensor<4x16xf32>
// CHECK:           %[[VAL_21:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_22:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [4, 16], strides: {{\[}}%[[VAL_21]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<4x16x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[VAL_22]], %[[VAL_20]]) <{static_mask_dims = array<i64>}> : (tensor<4x16x!tt.ptr<f32>>, tensor<4x16xf32>) -> ()
// CHECK:           tt.return

module {
  tt.func public @index_select_row_with_double_mod_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg5: i32 loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0), %arg6: i32 loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0)) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> loc(#loc1)
    %1 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>> loc(#loc2)
    %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32> loc(#loc2)
    %3 = tt.load %2 : tensor<4x!tt.ptr<i32>> loc(#loc3)
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc4)
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32> loc(#loc5)
    %6 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32> loc(#loc6)
    %7 = tt.splat %arg5 : i32 -> tensor<4x1xi32> loc(#loc7)
    %8 = arith.remsi %6, %7 : tensor<4x1xi32> loc(#loc7)
    %9 = arith.addi %5, %8 : tensor<4x1xi32> loc(#loc8)
    %10 = tt.splat %arg6 : i32 -> tensor<4x1xi32> loc(#loc9)
    %11 = arith.remsi %9, %10 : tensor<4x1xi32> loc(#loc9)
    %12 = tt.splat %arg3 : i32 -> tensor<4x1xi32> loc(#loc10)
    %13 = arith.muli %11, %12 : tensor<4x1xi32> loc(#loc10)
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>> loc(#loc11)
    %15 = tt.addptr %14, %13 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32> loc(#loc11)
    %16 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> loc(#loc12)
    %17 = tt.broadcast %15 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>> loc(#loc13)
    %18 = tt.broadcast %16 : tensor<1x16xi32> -> tensor<4x16xi32> loc(#loc13)
    %19 = tt.addptr %17, %18 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32> loc(#loc13)
    %20 = tt.load %19 : tensor<4x16x!tt.ptr<f32>> loc(#loc14)
    %21 = tt.splat %arg4 : i32 -> tensor<4x1xi32> loc(#loc15)
    %22 = arith.muli %6, %21 : tensor<4x1xi32> loc(#loc15)
    %23 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>> loc(#loc16)
    %24 = tt.addptr %23, %22 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32> loc(#loc16)
    %25 = tt.broadcast %24 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>> loc(#loc17)
    %26 = tt.addptr %25, %18 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32> loc(#loc17)
    tt.store %26, %20 : tensor<4x16x!tt.ptr<f32>> loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":109:31)
#loc2 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":110:36)
#loc3 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":110:26)
#loc4 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":112:31)
#loc5 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:33)
#loc6 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:56)
#loc7 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:66)
#loc8 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:44)
#loc9 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:80)
#loc10 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:95)
#loc11 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:20)
#loc12 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:118)
#loc13 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":114:106)
#loc14 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":116:19)
#loc15 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":120:33)
#loc16 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":120:10)
#loc17 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":121:10)
#loc18 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":122:8)
#loc19 = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":118:4)

#loc = loc("/workspace/triton_shared/triton_shared/python/examples/test_index_select.py":95:0)