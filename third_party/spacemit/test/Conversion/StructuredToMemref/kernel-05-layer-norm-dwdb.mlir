// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @_layer_norm_bwd_dwdb_0123456(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32, %arg5: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %cst : f32 -> tensor<256x256xf32>
    %6 = tt.splat %arg4 : i32 -> tensor<256x1xi32>
    %7 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %8 = tt.splat %arg5 : i32 -> tensor<1x256xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1x256xi32>
    %10 = tt.broadcast %9 : tensor<1x256xi1> -> tensor<256x256xi1>
    %11 = tt.splat %arg5 : i32 -> tensor<256x1xi32>
    %12 = tt.broadcast %7 : tensor<1x256xi32> -> tensor<256x256xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %15:2 = scf.for %arg6 = %c0_i32 to %arg4 step %c256_i32 iter_args(%arg7 = %5, %arg8 = %5) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
      %24 = tt.splat %arg6 : i32 -> tensor<256xi32>
      %25 = arith.addi %24, %2 : tensor<256xi32>
      %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
      %27 = arith.cmpi slt, %26, %6 : tensor<256x1xi32>
      %28 = tt.broadcast %27 : tensor<256x1xi1> -> tensor<256x256xi1>
      %29 = arith.andi %28, %10 : tensor<256x256xi1>
      %30 = arith.muli %26, %11 : tensor<256x1xi32>
      %31 = tt.broadcast %30 : tensor<256x1xi32> -> tensor<256x256xi32>
      %32 = arith.addi %31, %12 : tensor<256x256xi32>
      %33 = tt.addptr %13, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %34 = tt.load %33, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %35 = arith.addf %arg7, %34 : tensor<256x256xf32>
      %36 = tt.addptr %14, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %37 = tt.load %36, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %38 = arith.addf %arg8, %37 : tensor<256x256xf32>
      scf.yield %35, %38 : tensor<256x256xf32>, tensor<256x256xf32>
    }
    %16 = "tt.reduce"(%15#0) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %17 = "tt.reduce"(%15#1) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %18 = tt.splat %arg5 : i32 -> tensor<256xi32>
    %19 = arith.cmpi slt, %4, %18 : tensor<256xi32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %21 = tt.addptr %20, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %21, %16, %19 : tensor<256x!tt.ptr<f32>>
    %22 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %23 = tt.addptr %22, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %23, %17, %19 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @_layer_norm_bwd_dwdb_0123456
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<256x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_9_]], [[CST_256_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg12_:%.+]] = [[CST_0_]] to [[PARAM_4_]] step [[CST_256_]] iter_args([[VAR_arg13_:%.+]] = [[VAR_1_]], [[VAR_arg14_:%.+]] = [[VAR_1_]]) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.index_cast [[VAR_arg12_]] : i32 to index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_14_:%.+]] = arith.muli [[VAR_12_]], [[VAR_13_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.addi [[VAR_14_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_15_]]{{.}}, sizes: [256, 256], strides: {{.}}[[VAR_13_]], 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_12_]], [[CST_256_1_]] : index
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:             [[VAR_18_:%.+]] = arith.minsi [[VAR_16_]], [[VAR_17_]] : index
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[VAR_18_]], [[VAR_12_]] : index
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.subi [[VAR_19_]], [[VAR_12_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.addi [[VAR_3_]], [[CST_256_1_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.minsi [[VAR_21_]], [[VAR_13_]] : index
// CHECK:             [[VAR_23_:%.+]] = arith.maxsi [[VAR_22_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.subi [[VAR_23_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.minsi [[VAR_20_]], [[CST_256_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.minsi [[VAR_24_]], [[CST_256_1_]] : index
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256x256xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_256_1_]] : index
// CHECK:             [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_26_]], [[CST_256_1_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.ori [[VAR_27_]], [[VAR_28_]] : i1
// CHECK:             scf.if [[VAR_29_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_]] : memref<256x256xf32>)
// CHECK:             }
// CHECK-DAG:         [[VAR_subview_5_:%.+]] = memref.subview [[VAR_reinterpret_cast_4_]][0, 0] {{.}}[[VAR_25_]], [[VAR_26_]]{{.}} [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_:%.+]] = memref.subview [[RES_]][0, 0] {{.}}[[VAR_25_]], [[VAR_26_]]{{.}} [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             memref.copy [[VAR_subview_5_]], [[VAR_subview_6_]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             [[VAR_30_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256x256xf32>
// CHECK:             [[VAR_31_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg13_]], [[VAR_30_]] : tensor<256x256xf32>, tensor<256x256xf32>) outs([[VAR_arg13_]] : tensor<256x256xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:               [[VAR_34_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : f32
// CHECK:               linalg.yield [[VAR_34_]] : f32
// CHECK:             } -> tensor<256x256xf32>
// CHECK-DAG:         [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_15_]]{{.}}, sizes: [256, 256], strides: {{.}}[[VAR_13_]], 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<256x256xf32>
// CHECK:             scf.if [[VAR_29_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_1_]] : memref<256x256xf32>)
// CHECK:             }
// CHECK-DAG:         [[VAR_subview_9_:%.+]] = memref.subview [[VAR_reinterpret_cast_7_]][0, 0] {{.}}[[VAR_25_]], [[VAR_26_]]{{.}} [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_10_:%.+]] = memref.subview [[RES_1_]][0, 0] {{.}}[[VAR_25_]], [[VAR_26_]]{{.}} [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             memref.copy [[VAR_subview_9_]], [[VAR_subview_10_]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             [[VAR_32_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<256x256xf32>
// CHECK:             [[VAR_33_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg14_]], [[VAR_32_]] : tensor<256x256xf32>, tensor<256x256xf32>) outs([[VAR_arg14_]] : tensor<256x256xf32>) {
// CHECK:             ^bb0([[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32, [[IN_5_:%.+]]: f32):
// CHECK:               [[VAR_34_1_:%.+]] = arith.addf [[IN_3_]], [[IN_4_]] : f32
// CHECK:               linalg.yield [[VAR_34_1_]] : f32
// CHECK:             } -> tensor<256x256xf32>
// CHECK:             scf.yield [[VAR_31_]], [[VAR_33_]] : tensor<256x256xf32>, tensor<256x256xf32>
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_5_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_4_]]#0 : tensor<256x256xf32>) outs([[VAR_6_]] : tensor<256xf32>) dimensions = [0]
// CHECK:             ([[IN_3_:.+]]: f32, [[init_:.+]]: f32) {
// CHECK:               [[VAR_12_1_:%.+]] = arith.addf [[IN_3_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_12_1_]] : f32
// CHECK:             }
// CHECK:           [[VAR_reduced_0_:%.+]] = linalg.reduce ins([[VAR_4_]]#1 : tensor<256x256xf32>) outs([[VAR_6_]] : tensor<256xf32>) dimensions = [0]
// CHECK:             ([[IN_3_:.+]]: f32, [[init_:.+]]: f32) {
// CHECK:               [[VAR_12_2_:%.+]] = arith.addf [[IN_3_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_12_2_]] : f32
// CHECK:             }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.addi [[VAR_3_]], [[CST_256_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_9_:%.+]] = arith.minsi [[VAR_7_]], [[VAR_8_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.maxsi [[VAR_9_]], [[VAR_3_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.subi [[VAR_10_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_reduced_]][0] {{.}}[[VAR_11_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_11_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_]] : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_3_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_extracted_slice_2_:%.+]] = tensor.extract_slice [[VAR_reduced_0_]][0] {{.}}[[VAR_11_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK:           [[VAR_subview_3_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0] {{.}}[[VAR_11_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_2_]] in writable [[VAR_subview_3_]] : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
