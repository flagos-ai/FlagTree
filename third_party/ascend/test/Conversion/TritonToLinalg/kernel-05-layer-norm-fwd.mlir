// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @_layer_norm_fwd_fused_0123456789(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32, %arg7: i32, %arg8: f32) {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %cst_0 : f32 -> tensor<256xf32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %7 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %5 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %6 : tensor<256xi32>
      %35 = tt.addptr %7, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.addf %arg10, %36 : tensor<256xf32>
      scf.yield %37 : tensor<256xf32>
    }
    %9 = "tt.reduce"(%8) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %10 = arith.sitofp %arg7 : i32 to f32
    %11 = arith.divf %9, %10 : f32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %13 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %14 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.splat %11 : f32 -> tensor<256xf32>
    %16 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %12 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %13 : tensor<256xi32>
      %35 = tt.addptr %14, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.subf %36, %15 : tensor<256xf32>
      %38 = arith.select %34, %37, %4 : tensor<256xi1>, tensor<256xf32>
      %39 = arith.mulf %38, %38 : tensor<256xf32>
      %40 = arith.addf %arg10, %39 : tensor<256xf32>
      scf.yield %40 : tensor<256xf32>
    }
    %17 = "tt.reduce"(%16) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %18 = arith.divf %17, %10 : f32
    %19 = arith.addf %18, %arg8 : f32
    %20 = math.sqrt %19 : f32
    %21 = arith.divf %cst, %20 : f32
    %22 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %22, %11 : !tt.ptr<f32>
    %23 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %23, %21 : !tt.ptr<f32>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %25 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %27 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %28 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %29 = tt.splat %11 : f32 -> tensor<256xf32>
    %30 = tt.splat %21 : f32 -> tensor<256xf32>
    %31 = tt.splat %2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %24 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %25 : tensor<256xi32>
      %35 = tt.addptr %26, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34 : tensor<256x!tt.ptr<f32>>
      %37 = tt.addptr %27, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %38 = tt.load %37, %34 : tensor<256x!tt.ptr<f32>>
      %39 = tt.addptr %28, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %40 = tt.load %39, %34, %4 : tensor<256x!tt.ptr<f32>>
      %41 = arith.subf %40, %29 : tensor<256xf32>
      %42 = arith.mulf %41, %30 : tensor<256xf32>
      %43 = arith.mulf %42, %36 : tensor<256xf32>
      %44 = arith.addf %43, %38 : tensor<256xf32>
      %45 = tt.addptr %31, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %45, %44, %34 : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-DAG:   #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @_layer_norm_fwd_fused_0123456789
// CHECK-SAME:   (%[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, [[PARAM_0_:%.+]]: memref<?xf32> {tt.tensor_kind = 0 : i32},  [[PARAM_1_:%.+]]: memref<?xf32> {tt.tensor_kind = 1 : i32}, [[PARAM_2_:%.+]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, [[PARAM_3_:%.+]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, [[PARAM_4_:%.+]]: memref<?xf32> {tt.tensor_kind = 1 : i32}, [[PARAM_5_:%.+]]: memref<?xf32> {tt.tensor_kind = 1 : i32},
// CHECK-SAME:  [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: f32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG:   [[c256:%.+]] = arith.constant 256 : index
// CHECK-DAG:   [[c0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:   [[c256_i32:%.+]] = arith.constant 256 : i32
// CHECK-DAG:   [[cst:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   [[c0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[cst_0:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:   %[[VAL_0:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:    %[[VAL_1:.*]] = linalg.fill ins([[cst_0]] : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_2:.*]] = tensor.empty() : tensor<256xf32>
// CHECK:    %[[VAL_3:.*]] = linalg.fill ins([[cst]] : f32) outs(%[[VAL_2]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:    %[[VAL_4:.*]] = arith.muli [[PARAM_12_]], [[PARAM_6_]] : i32
// CHECK:    %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:    %[[VAL_6:.*]] = scf.for %arg17 = [[c0_i32]] to [[PARAM_7_]] step [[c256_i32]] iter_args(%arg18 = %[[VAL_3]]) -> (tensor<256xf32>)  : i32 {
// CHECK:    %[[VAL_33:.*]] = arith.index_cast %arg17 : i32 to index
// CHECK:    %[[VAL_34:.*]] = arith.addi %[[VAL_5]], %[[VAL_33]] : index
// CHECK:    %reinterpret_cast_9 = memref.reinterpret_cast [[PARAM_0_]] to offset: [%[[VAL_34]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:    %alloc = memref.alloc() : memref<256xf32>
// CHECK:    %[[VAL_35:.*]] = arith.addi %[[VAL_33]], [[c256]] : index
// CHECK:    %[[VAL_36:.*]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:    %[[VAL_37:.*]] = arith.maxsi %[[VAL_33]], %[[VAL_36]] : index
// CHECK:    %[[VAL_38:.*]] = arith.minsi %[[VAL_35]], %[[VAL_37]] : index
// CHECK:    %[[VAL_39:.*]] = arith.subi %[[VAL_38]], %[[VAL_33]] : index
// CHECK:    %[[VAL_40:.*]] = arith.cmpi slt, %[[VAL_39]], [[c256]] : index
// CHECK:    scf.if %[[VAL_40]] {
// CHECK:      linalg.fill ins([[cst]] : f32) outs(%alloc : memref<256xf32>)
// CHECK:    }
// CHECK:    %subview = memref.subview %reinterpret_cast_9[0] [%[[VAL_39]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:    %subview_10 = memref.subview %alloc[0] [%[[VAL_39]]] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:    memref.copy %subview, %subview_10 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:    %[[VAL_41:.*]] = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
// CHECK:    %[[VAL_42:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg18, %[[VAL_41]] : tensor<256xf32>, tensor<256xf32>) outs(%arg18 : tensor<256xf32>) {
// CHECK:    ^bb0(%in: f32, %in_11: f32, %out: f32):
// CHECK:      %[[VAL_43:.*]] = arith.addf %in, %in_11 : f32
// CHECK:      linalg.yield %[[VAL_43]] : f32
// CHECK:    } -> tensor<256xf32>
// CHECK:    scf.yield %[[VAL_42]] : tensor<256xf32>
// CHECK:    }
// CHECK:    %[[VAL_7:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins([[cst]] : f32) outs(%[[VAL_7]] : tensor<f32>) -> tensor<f32>
// CHECK:    %reduced = linalg.reduce ins(%[[VAL_6]] : tensor<256xf32>) outs(%[[VAL_8]] : tensor<f32>) dimensions = [0] 
// CHECK:      (%in: f32, %init: f32) {
// CHECK:        %[[VAL_33:.*]] = arith.addf %in, %init : f32
// CHECK:        linalg.yield %[[VAL_33]] : f32
// CHECK:      }
// CHECK:    %extracted = tensor.extract %reduced[] : tensor<f32>
// CHECK:    %[[VAL_9:.*]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK:    %[[VAL_10:.*]] = linalg.fill ins(%extracted : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_9]] : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_12:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_10]], %[[VAL_11]] : tensor<1xf32>, tensor<1xf32>) outs(%[[VAL_10]] : tensor<1xf32>) {
// CHECK:    ^bb0(%in: f32, %in_9: f32, %out: f32):
// CHECK:      %[[VAL_33:.*]] = arith.divf %in, %in_9 : f32
// CHECK:      linalg.yield %[[VAL_33]] : f32
// CHECK:    } -> tensor<1xf32>
// CHECK:    %extracted_1 = tensor.extract %[[VAL_12]][[[c0]]] : tensor<1xf32>
// CHECK:    %[[VAL_13:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:    %[[VAL_14:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_13]] : tensor<256xi32>) {
// CHECK:    ^bb0(%out: i32):
// CHECK:      %[[VAL_33:.*]] = linalg.index 0 : index
// CHECK:      %[[VAL_34:.*]] = arith.index_cast %[[VAL_33]] : index to i32
// CHECK:      linalg.yield %[[VAL_34]] : i32
// CHECK:    } -> tensor<256xi32>
// CHECK:    %[[VAL_15:.*]] = linalg.fill ins([[PARAM_7_]] : i32) outs(%[[VAL_13]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK:    %[[VAL_16:.*]] = linalg.fill ins(%extracted_1 : f32) outs(%[[VAL_2]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:    %[[VAL_17:.*]] = scf.for %arg17 = [[c0_i32]] to [[PARAM_7_]] step [[c256_i32]] iter_args(%arg18 = %[[VAL_3]]) -> (tensor<256xf32>)  : i32 {
// CHECK:    %[[VAL_33:.*]] = linalg.fill ins(%arg17 : i32) outs(%[[VAL_13]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK:    %[[VAL_34:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_33]], %[[VAL_14]] : tensor<256xi32>, tensor<256xi32>) outs(%[[VAL_33]] : tensor<256xi32>) {
// CHECK:      ^bb0(%in: i32, %in_11: i32, %out: i32):
// CHECK:        %[[VAL_50:.*]] = arith.addi %in, %in_11 : i32
// CHECK:        linalg.yield %[[VAL_50:.*]] : i32
// CHECK:      } -> tensor<256xi32>
// CHECK:      %[[VAL_35:.*]] = tensor.empty() : tensor<256xi1>
// CHECK:      %[[VAL_36:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_34]], %[[VAL_15]] : tensor<256xi32>, tensor<256xi32>) outs(%[[VAL_35]] : tensor<256xi1>) {
// CHECK:      ^bb0(%in: i32, %in_11: i32, %out: i1):
// CHECK:        %[[VAL_50:.*]] = arith.cmpi slt, %in, %in_11 : i32
// CHECK:        linalg.yield %[[VAL_50:.*]] : i1
// CHECK:      } -> tensor<256xi1>
// CHECK:      %[[VAL_37:.*]] = arith.index_cast %arg17 : i32 to index
// CHECK:      %[[VAL_38:.*]] = arith.addi %[[VAL_5]], %[[VAL_37]] : index
// CHECK:      %reinterpret_cast_9 = memref.reinterpret_cast [[PARAM_0_]] to offset: [%[[VAL_38]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:      %alloc = memref.alloc() : memref<256xf32>
// CHECK:      %[[VAL_39:.*]] = arith.addi %[[VAL_37]], [[c256]] : index
// CHECK:      %[[VAL_40:.*]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:      %[[VAL_41:.*]] = arith.maxsi %[[VAL_37]], %[[VAL_40]] : index
// CHECK:      %[[VAL_42:.*]] = arith.minsi %[[VAL_39]], %[[VAL_41]] : index
// CHECK:      %[[VAL_43:.*]] = arith.subi %[[VAL_42]], %[[VAL_37]] : index
// CHECK:      %[[VAL_44:.*]] = arith.cmpi slt, %[[VAL_43]], [[c256]] : index
// CHECK:      scf.if %[[VAL_44]] {
// CHECK:        linalg.fill ins([[cst]] : f32) outs(%alloc : memref<256xf32>)
// CHECK:      }
// CHECK:      %subview = memref.subview %reinterpret_cast_9[0] [%[[VAL_43]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %subview_10 = memref.subview %alloc[0] [%[[VAL_43]]] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:      memref.copy %subview, %subview_10 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:      %[[VAL_45:.*]] = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
// CHECK:      %[[VAL_46:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_45]], %[[VAL_16]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAL_45]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_11: f32, %out: f32):
// CHECK:        %[[VAL_50:.*]] = arith.subf %in, %in_11 : f32
// CHECK:        linalg.yield %[[VAL_50:.*]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %[[VAL_47:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_36]], %[[VAL_46]], %[[VAL_3]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%[[VAL_46]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: i1, %in_11: f32, %in_12: f32, %out: f32):
// CHECK:        %[[VAL_50:.*]] = arith.select %in, %in_11, %in_12 : f32
// CHECK:        linalg.yield %[[VAL_50:.*]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %[[VAL_48:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_47]], %[[VAL_47]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAL_47]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_11: f32, %out: f32):
// CHECK:        %[[VAL_50:.*]] = arith.mulf %in, %in_11 : f32
// CHECK:        linalg.yield %[[VAL_50:.*]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %[[VAL_49:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg18, %[[VAL_48]] : tensor<256xf32>, tensor<256xf32>) outs(%arg18 : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_11: f32, %out: f32):
// CHECK:        %[[VAL_50:.*]] = arith.addf %in, %in_11 : f32
// CHECK:        linalg.yield %[[VAL_50:.*]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      scf.yield %[[VAL_49]] : tensor<256xf32>
// CHECK:    }
// CHECK:    %[[VAL_18:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:    %[[VAL_19:.*]] = linalg.fill ins([[cst]] : f32) outs(%[[VAL_18]] : tensor<f32>) -> tensor<f32>
// CHECK:    %reduced_2 = linalg.reduce ins(%[[VAL_17]] : tensor<256xf32>) outs(%[[VAL_19]] : tensor<f32>) dimensions = [0] 
// CHECK:      (%in: f32, %init: f32) {
// CHECK:        %[[VAL_33:.*]] = arith.addf %in, %init : f32
// CHECK:        linalg.yield %[[VAL_33]] : f32
// CHECK:      }
// CHECK:    %extracted_3 = tensor.extract %reduced_2[] : tensor<f32>
// CHECK:    %[[VAL_20:.*]] = linalg.fill ins(%extracted_3 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_20]], %[[VAL_11]] : tensor<1xf32>, tensor<1xf32>) outs(%[[VAL_20]] : tensor<1xf32>) {
// CHECK:    ^bb0(%in: f32, %in_9: f32, %out: f32):
// CHECK:      %[[VAL_33:.*]] = arith.divf %in, %in_9 : f32
// CHECK:      linalg.yield %[[VAL_33]] : f32
// CHECK:    } -> tensor<1xf32>
// CHECK:    %extracted_4 = tensor.extract %[[VAL_21]][[[c0]]] : tensor<1xf32>
// CHECK:    %[[VAL_22:.*]] = linalg.fill ins(%extracted_4 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_23:.*]] = linalg.fill ins([[PARAM_8_]] : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_22]], %[[VAL_23]] : tensor<1xf32>, tensor<1xf32>) outs(%[[VAL_22]] : tensor<1xf32>) {
// CHECK:    ^bb0(%in: f32, %in_9: f32, %out: f32):
// CHECK:      %[[VAL_33:.*]] = arith.addf %in, %in_9 : f32
// CHECK:      linalg.yield %[[VAL_33]] : f32
// CHECK:    } -> tensor<1xf32>
// CHECK:    %extracted_5 = tensor.extract %[[VAL_24]][[[c0]]] : tensor<1xf32>
// CHECK:    %[[VAL_25:.*]] = linalg.fill ins(%extracted_5 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_26:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[VAL_25]] : tensor<1xf32>) outs(%[[VAL_25]] : tensor<1xf32>) {
// CHECK:    ^bb0(%in: f32, %out: f32):
// CHECK:      %[[VAL_33:.*]] = math.sqrt %in : f32
// CHECK:      linalg.yield %[[VAL_33]] : f32
// CHECK:    } -> tensor<1xf32>
// CHECK:    %extracted_6 = tensor.extract %[[VAL_26]][[[c0]]] : tensor<1xf32>
// CHECK:    %[[VAL_27:.*]] = linalg.fill ins(%extracted_6 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %[[VAL_28:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_27]] : tensor<1xf32>, tensor<1xf32>) outs(%[[VAL_1]] : tensor<1xf32>) {
// CHECK:    ^bb0(%in: f32, %in_9: f32, %out: f32):
// CHECK:      %[[VAL_33:.*]] = arith.divf %in, %in_9 : f32
// CHECK:      linalg.yield %[[VAL_33]] : f32
// CHECK:    } -> tensor<1xf32>
// CHECK:    %extracted_7 = tensor.extract %[[VAL_28]][[[c0]]] : tensor<1xf32>
// CHECK:    %[[VAL_29:.*]] = arith.index_cast [[PARAM_12_]] : i32 to index
// CHECK:    %[[VAL_30:.*]] = linalg.fill ins(%extracted_1 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %reinterpret_cast = memref.reinterpret_cast [[PARAM_4_]] to offset: [%[[VAL_29]]], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_30]] in writable %reinterpret_cast : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:    %[[VAL_31:.*]] = linalg.fill ins(%extracted_7 : f32) outs(%[[VAL_0]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:    %reinterpret_cast_8 = memref.reinterpret_cast [[PARAM_5_]] to offset: [%[[VAL_29]]], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_31]] in writable %reinterpret_cast_8 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:    %[[VAL_32:.*]] = linalg.fill ins(%extracted_7 : f32) outs(%[[VAL_2]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:    scf.for %arg17 = [[c0_i32]] to [[PARAM_7_]] step [[c256_i32]]  : i32 {
// CHECK:      %[[VAL_33:.*]] = arith.index_cast %arg17 : i32 to index
// CHECK:      %reinterpret_cast_9 = memref.reinterpret_cast [[PARAM_2_]] to offset: [%[[VAL_33]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:      %alloc = memref.alloc() : memref<256xf32>
// CHECK:      %[[VAL_34:.*]] = arith.addi %[[VAL_33]], [[c256]] : index
// CHECK:      %[[VAL_35:.*]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:      %[[VAL_36:.*]] = arith.maxsi %[[VAL_33]], %[[VAL_35]] : index
// CHECK:      %[[VAL_37:.*]] = arith.minsi %[[VAL_34]], %[[VAL_36]] : index
// CHECK:      %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_33]] : index
// CHECK:      %subview = memref.subview %reinterpret_cast_9[0] [%[[VAL_38]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %subview_10 = memref.subview %alloc[0] [%[[VAL_38]]] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:      memref.copy %subview, %subview_10 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:      %[[VAL_39:.*]] = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
// CHECK:      %reinterpret_cast_11 = memref.reinterpret_cast [[PARAM_3_]] to offset: [%[[VAL_33]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:      %alloc_12 = memref.alloc() : memref<256xf32>
// CHECK:      %subview_13 = memref.subview %reinterpret_cast_11[0] [%[[VAL_38]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %subview_14 = memref.subview %alloc_12[0] [%[[VAL_38]]] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:      memref.copy %subview_13, %subview_14 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:      %[[VAL_40:.*]] = bufferization.to_tensor %alloc_12 restrict writable : memref<256xf32>
// CHECK:      %[[VAL_41:.*]] = arith.addi %[[VAL_5]], %[[VAL_33]] : index
// CHECK:      %reinterpret_cast_15 = memref.reinterpret_cast [[PARAM_0_]] to offset: [%[[VAL_41]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:      %alloc_16 = memref.alloc() : memref<256xf32>
// CHECK:      %[[VAL_42:.*]] = arith.cmpi slt, %[[VAL_38]], [[c256]] : index
// CHECK:      scf.if %[[VAL_42]] {
// CHECK:        linalg.fill ins([[cst]] : f32) outs(%alloc_16 : memref<256xf32>)
// CHECK:      }
// CHECK:      %subview_17 = memref.subview %reinterpret_cast_15[0] [%[[VAL_38]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %subview_18 = memref.subview %alloc_16[0] [%[[VAL_38]]] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:      memref.copy %subview_17, %subview_18 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:      %[[VAL_43:.*]] = bufferization.to_tensor %alloc_16 restrict writable : memref<256xf32>
// CHECK:      %[[VAL_44:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_43]], %[[VAL_16]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAL_43]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_21: f32, %out: f32):
// CHECK:        %[[VAL_48:.*]] = arith.subf %in, %in_21 : f32
// CHECK:        linalg.yield %[[VAL_48]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %[[VAR_45:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_44]], %[[VAL_32]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAL_44]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_21: f32, %out: f32):
// CHECK:        %[[VAL_48:.*]] = arith.mulf %in, %in_21 : f32
// CHECK:        linalg.yield %[[VAL_48]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:     %[[VAR_46:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAR_45]], %[[VAL_39]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAR_45]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_21: f32, %out: f32):
// CHECK:        %[[VAL_48:.*]] = arith.mulf %in, %in_21 : f32
// CHECK:        linalg.yield %[[VAL_48]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %[[VAL_47:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAR_46]], %[[VAL_40]] : tensor<256xf32>, tensor<256xf32>) outs(%[[VAR_46]] : tensor<256xf32>) {
// CHECK:      ^bb0(%in: f32, %in_21: f32, %out: f32):
// CHECK:        %[[VAL_48:.*]] = arith.addf %in, %in_21 : f32
// CHECK:        linalg.yield %[[VAL_48]] : f32
// CHECK:      } -> tensor<256xf32>
// CHECK:      %reinterpret_cast_19 = memref.reinterpret_cast [[PARAM_1_]] to offset: [%[[VAL_41]]], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK:      %extracted_slice = tensor.extract_slice %[[VAL_47]][0] [%[[VAL_38]]] [1] : tensor<256xf32> to tensor<?xf32>
// CHECK:      %subview_20 = memref.subview %reinterpret_cast_19[0] [%[[VAL_38]]] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:      bufferization.materialize_in_destination %extracted_slice in writable %subview_20 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
// CHECK:    }
// CHECK:    return
// CHECK:  }