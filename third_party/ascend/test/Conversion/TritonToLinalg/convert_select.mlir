// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) -> () {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<256> : tensor<512xi64>
    %cst_1 = arith.constant dense<512> : tensor<512xi32>
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %1 = arith.cmpi slt, %0, %cst_1 : tensor<512xi32>
    %2 = arith.extsi %0 : tensor<512xi32> to tensor<512xi64>
    %3 = arith.cmpi slt, %2, %cst_0 : tensor<512xi64>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %6 = tt.load %5, %1, %cst evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %8 = tt.addptr %7, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %9 = tt.load %8, %1, %cst evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    %10 = arith.select %3, %6, %9 : tensor<512xi1>, tensor<512xf32>
    %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %12 = tt.addptr %11, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    tt.store %12, %10, %1 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    tt.return
    }
}

// CHECK-LABEL:  func.func @kernel(
// CHECK-SAME:  %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[ARG_2:.*]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, %[[ARG_3:.*]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, %[[ARG_4:.*]]: memref<?xf32> {tt.tensor_kind = 1 : i32},
// CHECK-SAME:  %[[ARG_5:.*]]: i32, %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32, %[[ARG_10:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG:  [[c256_i64:%.+]] = arith.constant 256 : i64
// CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<512xi64>
// CHECK:  %[[VAL_1:.*]] = linalg.fill ins([[c256_i64]] : i64) outs(%[[VAL_0]] : tensor<512xi64>) -> tensor<512xi64>
// CHECK:  %[[VAL_2:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:  %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_2]] : tensor<512xi32>) {
// CHECK:  ^bb0(%out: i32):
// CHECK:    %[[VAL_10:.*]] = linalg.index 0 : index
// CHECK:    %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : index to i32
// CHECK:    linalg.yield %[[VAL_11]] : i32
// CHECK:  } -> tensor<512xi32>
// CHECK:  %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[VAL_3]] : tensor<512xi32>) outs(%[[VAL_0]] : tensor<512xi64>) {
// CHECK:  ^bb0(%in: i32, %out: i64):
// CHECK:    %[[VAL_10:.*]] = arith.extsi %in : i32 to i64
// CHECK:    linalg.yield %[[VAL_10]] : i64
// CHECK:  } -> tensor<512xi64>
// CHECK:  %[[VAL_5:.*]] = tensor.empty() : tensor<512xi1>
// CHECK:  %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_4]], %[[VAL_1]] : tensor<512xi64>, tensor<512xi64>) outs(%[[VAL_5]] : tensor<512xi1>) {
// CHECK:  ^bb0(%in: i64, %in_3: i64, %out: i1):
// CHECK:    %[[VAL_10:.*]] = arith.cmpi slt, %in, %in_3 : i64
// CHECK:    linalg.yield %[[VAL_10]] : i1
// CHECK:  } -> tensor<512xi1>
// CHECK:  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:  %alloc = memref.alloc() : memref<512xf32>
// CHECK:  memref.copy %reinterpret_cast, %alloc : memref<512xf32, strided<[1]>> to memref<512xf32>
// CHECK:  %[[VAL_7:.*]] = bufferization.to_tensor %alloc restrict writable : memref<512xf32>
// CHECK:  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:  %alloc_1 = memref.alloc() : memref<512xf32>
// CHECK:  memref.copy %reinterpret_cast_0, %alloc_1 : memref<512xf32, strided<[1]>> to memref<512xf32>
// CHECK:  %[[VAL_8:.*]] = bufferization.to_tensor %alloc_1 restrict writable : memref<512xf32>
// CHECK:  %[[VAL_9:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : tensor<512xi1>, tensor<512xf32>, tensor<512xf32>) outs(%[[VAL_7]] : tensor<512xf32>) {
// CHECK:  ^bb0(%in: i1, %in_3: f32, %in_4: f32, %out: f32):
// CHECK:    %[[VAL_10:.*]] = arith.select %in, %in_3, %in_4 : f32
// CHECK:    linalg.yield %[[VAL_10]] : f32
// CHECK:  } -> tensor<512xf32>
// CHECK:  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:  bufferization.materialize_in_destination %[[VAL_9]] in writable %reinterpret_cast_2 : (tensor<512xf32>, memref<512xf32, strided<[1]>>) -> ()
// CHECK:  return
// CHECK:  }