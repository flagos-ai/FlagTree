// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure extract_slice is generated correctly.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @row_gather3(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [32], strides: [1] : memref<*xi32> to memref<32xi32, strided<[1]>>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<32xi32>
// CHECK:           memref.copy %[[VAL_15]], %[[VAL_16]] : memref<32xi32, strided<[1]>> to memref<32xi32>
// CHECK:           %[[VAL_17:.*]] = bufferization.to_tensor %[[VAL_16]] restrict writable : memref<32xi32> to tensor<32xi32>
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:           %[[VAL_20:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_19]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK:           %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_17]], %[[VAL_20]] : tensor<32xi32>, tensor<32xi32>) outs(%[[VAL_17]] : tensor<32xi32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: i32):
// CHECK:             %[[VAL_25:.*]] = arith.muli %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:             linalg.yield %[[VAL_25]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           %[[VAL_26:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_27:.*]] = memref.alloc() : memref<32x32x32xf32>
// CHECK:           scf.for %[[VAL_28:.*]] = %[[VAL_14]] to %[[VAL_13]] step %[[VAL_12]] {
// CHECK:             %[[VAL_29:.*]] = tensor.extract %[[VAL_21]]{{\[}}%[[VAL_28]]] : tensor<32xi32>
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : i32 to index
// CHECK:             %[[VAL_31:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_30]]], sizes: [32, 1, 32], strides: {{\[}}%[[VAL_18]], 1, %[[VAL_26]]] : memref<*xf32> to memref<32x1x32xf32, strided<[?, 1, ?], offset: ?>>
// CHECK:             %[[VAL_32:.*]] = memref.subview %[[VAL_27]][0, %[[VAL_28]], 0] [32, 1, 32] [1, 1, 1] : memref<32x32x32xf32> to memref<32x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_31]], %[[VAL_32]] : memref<32x1x32xf32, strided<[?, 1, ?], offset: ?>> to memref<32x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = bufferization.to_tensor %[[VAL_27]] restrict writable : memref<32x32x32xf32> to tensor<32x32x32xf32>
// CHECK:           scf.for %[[VAL_34:.*]] = %[[VAL_14]] to %[[VAL_13]] step %[[VAL_12]] {
// CHECK:             %[[VAL_35:.*]] = tensor.extract %[[VAL_21]]{{\[}}%[[VAL_34]]] : tensor<32xi32>
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : i32 to index
// CHECK:             %[[VAL_37:.*]] = tensor.extract_slice %[[VAL_33]][0, %[[VAL_34]], 0] [32, 1, 32] [1, 1, 1] : tensor<32x32x32xf32> to tensor<32x1x32xf32>
// CHECK:             %[[VAL_38:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_36]]], sizes: [32, 1, 32], strides: {{\[}}%[[VAL_18]], 1, %[[VAL_26]]] : memref<*xf32> to memref<32x1x32xf32, strided<[?, 1, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_37]] in writable %[[VAL_38]] : (tensor<32x1x32xf32>, memref<32x1x32xf32, strided<[?, 1, ?], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return

module {
  tt.func public @row_gather3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %0 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<32x!tt.ptr<i32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<i32>>) -> tensor<32xi32>
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = tt.splat %arg4 : i32 -> tensor<32xi32>
    %4 = arith.muli %1, %3 : tensor<32xi32>
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = tts.make_gather_scatter_tptr %arg0 to sizes: [32, 32, 32] gather_scatter_dim: 1 gather_scatter_offset: %4, strides: [%2, 1, %5], offsets: [0, 0, 0] : tensor<32xi32> <f32> to !tt.ptr<tensor<32x32x32xf32>>
    %7 = "tts.load"(%6) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<32x32x32xf32>>) -> tensor<32x32x32xf32>
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = tt.splat %arg4 : i32 -> tensor<32xi32>
    %10 = arith.muli %1, %9 : tensor<32xi32>
    %11 = arith.index_cast %arg5 : i32 to index
    %12 = tts.make_gather_scatter_tptr %arg2 to sizes: [32, 32, 32] gather_scatter_dim: 1 gather_scatter_offset: %10, strides: [%8, 1, %11], offsets: [0, 0, 0] : tensor<32xi32> <f32> to !tt.ptr<tensor<32x32x32xf32>>
    "tts.store"(%12, %7) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<32x32x32xf32>>, tensor<32x32x32xf32>) -> ()
    tt.return
  }
}
