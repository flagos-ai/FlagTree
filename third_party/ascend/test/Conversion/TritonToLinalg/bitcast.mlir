// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[VAL_0:.*]]: memref<?xi32> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xf32> {tt.tensor_kind = 1 : i32},
// CHECK-SAME:      %[[ARG_4:.*]]: i32, %[[ARG_5:.*]]: i32, %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:   [[RC_:%.+]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<?xi32> to memref<1024xi32, strided<[1]>>
// CHECK:   [[RC_0_:%.+]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<?xf32> to memref<1024xf32, strided<[1]>>
// CHECK:   [[ALLOC_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:   memref.copy [[RC_]], [[ALLOC_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:   [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<1024xi32>
// CHECK:   [[VAR_1_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:   [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xi32>) outs([[VAR_1_]] : tensor<1024xf32>) {
// CHECK:   ^bb0(%in: i32, %out: f32):
// CHECK:     [[VAR_5_:%.+]] = arith.bitcast %in : i32 to f32
// CHECK:     linalg.yield [[VAR_5_]] : f32
// CHECK:   } -> tensor<1024xf32>
// CHECK:   bufferization.materialize_in_destination [[VAR_2_]] in writable [[RC_0_]]
// CHECK:     return
// CHECK:   }


