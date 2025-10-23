// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
// RUN: triton-adapter-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s --check-prefix NAMED
module {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: f32 {tt.divisibility = 16 : i32}, %arg2: f32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : f32 -> tensor<32xf32>
    %5 = tt.splat %arg2 : f32 -> tensor<32xf32>
    %6 = tt.clampf %3, %4, %5, propagateNan = none : tensor<32xf32>
    %7 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %8, %6 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[VAL_0:.*]]: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: f32 {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: f32 {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
// CHECK-SAME:      %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32, %[[ARG_10:.*]]: i32, %[[ARG_11:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:     %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[VAR_0:.*]] to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
// CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<32xf32>
// CHECK:     memref.copy %[[REINTERPRET_CAST]], %[[ALLOC:.*]] : memref<32xf32, strided<[1]>> to memref<32xf32>
// CHECK:     %[[VAR_10:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<32xf32>
// CHECK:     %[[VAR_11:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:     %[[VAR_12:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAR_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:     %[[VAR_14:.*]] = linalg.fill ins(%[[VAL_2]] : f32) outs(%[[VAR_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:     %[[VAR_15:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAR_10]], %[[VAR_14]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAR_10]] : tensor<32xf32>) {
// CHECK:     ^bb0(%[[VAR_16:.*]]: f32, %[[VAR_17:.*]]: f32, %[[VAR_18:.*]]: f32):
// CHECK:       %[[VAR_19:.*]] = arith.minnumf %[[VAR_16]], %[[VAR_17]] : f32
// CHECK:       linalg.yield %[[VAR_19]] : f32
// CHECK:     } -> tensor<32xf32>
// CHECK:     %[[VAR_20:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAR_12]], %[[VAR_15]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAR_12]] : tensor<32xf32>) {
// CHECK:     ^bb0(%[[VAR_21:.*]]: f32, %[[VAR_22:.*]]: f32, %[[VAR_23:.*]]: f32):
// CHECK:       %[[VAR_19]] = arith.maxnumf %[[VAR_21]], %[[VAR_22]] : f32
// CHECK:       linalg.yield %[[VAR_19]] : f32
// CHECK:     } -> tensor<32xf32>
// CHECK:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
// CHECK:     bufferization.materialize_in_destination %[[VAR_20]] in writable %[[REINTERPRET_CAST_0]] : (tensor<32xf32>, memref<32xf32, strided<[1]>>) -> ()
// CHECK:     return
// CHECK:   }

// NAMED-LABEL:   func.func @kernel
// NAMED-NOT:       linalg.generic
