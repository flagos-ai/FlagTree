// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @add_kernel_01234(
// CHECK-SAME:   %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, [[PARAM_0_:%.+]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, [[PARAM_1_:%.+]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, [[PARAM_2_:%.+]]: memref<?xf32> {tt.tensor_kind = 1 : i32}, 
// CHECK-SAME:   [[PARAM_3_:%.+]]: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_1024_1_:%.+]] = arith.constant 1024 : i32
// CHECK:           [[VAR_0_:%.+]] = arith.muli %arg9, [[CST_1024_1_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1024], strides: [1]{{.*}} : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.addi [[VAR_1_]], [[CST_1024_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_5_0_:%.+]] = arith.maxsi [[VAR_1_]], [[VAR_4_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_3_]], [[VAR_5_0_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1]{{.*}} : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_]]_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1024], strides: [1]{{.*}} : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_3_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0] [[[VAR_6_]]] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_4_:%.+]] = memref.subview [[RES_1_]][0] [[[VAR_6_]]] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_3_]], [[VAR_subview_4_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_14_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xf32>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_14_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_7_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[in_1]], [[in_2]] : f32
// CHECK:             linalg.yield [[VAR_22_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1024], strides: [1]{{.*}} : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_15_]][0] [[[VAR_6_]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK-DAG:       [[VAR_subview_6_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_]][0] [[[VAR_6_]]] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_6_]]
// CHECK:           return
// CHECK:         }
