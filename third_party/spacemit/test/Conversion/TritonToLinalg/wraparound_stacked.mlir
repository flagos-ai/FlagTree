// XFAIL: *
// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @wrap_stacked_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %3 = arith.remsi %1, %2 : tensor<4xi32>
    %4 = arith.addi %0, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg5 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg6 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg7 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %21, %cst_0 : tensor<1x4xi32>
    %28 = tt.broadcast %27 : tensor<1x4xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg5, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %26) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
      %32 = tt.load %arg9, %28, %cst : tensor<4x4x!tt.ptr<f32>>
      tt.store %arg10, %32 : tensor<4x4x!tt.ptr<f32>>
      %33 = tt.addptr %arg9, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %34 = tt.addptr %arg10, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %33, %34 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @wrap_stacked_masked_loop_01234567
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_9_dot_900000_:%.+]] = arith.constant -9.900000e+01 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.addi [[VAR_2_]], [[VAR_4_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.remsi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[VAR_6_]] : index
// CHECK:           [[VAR_9_:%.+]] = arith.subi [[VAR_8_]], [[VAR_5_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.divsi [[VAR_9_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: {{.}}[[VAR_10_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_1_]], [[VAR_3_]]{{.}} : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.subi [[CST_4_]], [[VAR_10_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_6_]]{{.}}, sizes: {{.}}[[VAR_11_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_1_]], [[VAR_3_]]{{.}} : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.muli [[PARAM_5_]], [[CST_4_1_]] : i32
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.muli [[VAR_18_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[CST_0_]]{{.}}, sizes: [4, 4], strides: {{.}}[[VAR_12_]], [[VAR_13_]]{{.}} : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]]:6 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]] step [[CST_1_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg16_:%.+]] = [[VAR_reinterpret_cast_]]_1, [[VAR_arg17_:%.+]] = [[VAR_17_]], [[VAR_arg18_:%.+]] = [[CST_0_]], [[VAR_arg19_:%.+]] = [[CST_0_]], [[VAR_arg20_:%.+]] = [[VAR_reinterpret_cast_]]_0) -> (memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>)  : i32 {
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<4x4xf32>
// CHECK:             linalg.fill ins([[CST_minus_9_dot_900000_]] : f32) outs([[RES_]] : memref<4x4xf32>)
// CHECK:             [[VAR_dim_:%.+]] = memref.dim [[VAR_arg15_]], [[CST_0_]] : memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK:             [[VAR_21_:%.+]] = arith.minsi [[VAR_dim_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.subi [[CST_4_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_subview_:%.+]] = memref.subview [[VAR_arg15_]][0, 0] {{.}}[[VAR_21_]], 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_subview_2_:%.+]] = memref.subview [[VAR_arg20_]][0, 0] {{.}}[[VAR_22_]], 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_subview_3_:%.+]] = memref.subview [[RES_]][0, 0] {{.}}[[VAR_21_]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
// CHECK-DAG:         [[VAR_subview_4_:%.+]] = memref.subview [[RES_]]{{.}}[[VAR_21_]], 0] {{.}}[[VAR_22_]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy [[VAR_subview_]], [[VAR_subview_]]_3 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
// CHECK:             memref.copy [[VAR_subview_2_]], [[VAR_subview_4_]] : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             [[VAR_23_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x4xf32>
// CHECK:             bufferization.materialize_in_destination [[VAR_23_]] in writable [[VAR_arg16_]]
// CHECK:             [[VAR_24_:%.+]] = arith.index_cast [[VAR_14_]] : i32 to index
// CHECK:             [[VAR_25_:%.+]] = arith.addi [[VAR_arg17_]], [[VAR_24_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.addi [[VAR_25_]], [[VAR_19_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.remsi [[VAR_26_]], [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.muli [[VAR_15_]], [[VAR_16_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.addi [[VAR_28_]], [[VAR_27_]] : index
// CHECK:             [[VAR_30_:%.+]] = arith.subi [[VAR_29_]], [[VAR_26_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.divsi [[VAR_30_]], [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_26_]]{{.}}, sizes: {{.}}[[VAR_31_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_16_]], [[VAR_18_]]{{.}} : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.subi [[CST_4_]], [[VAR_31_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_27_]]{{.}}, sizes: {{.}}[[VAR_32_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_16_]], [[VAR_18_]]{{.}} : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.index_cast [[VAR_14_]] : i32 to index
// CHECK:             [[VAR_34_:%.+]] = arith.addi [[VAR_arg18_]], [[VAR_33_]] : index
// CHECK:             [[VAR_35_:%.+]] = arith.addi [[VAR_34_]], [[VAR_arg19_]] : index
// CHECK:             [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_35_]]{{.}}, sizes: [4, 4], strides: {{.}}[[VAR_12_]], [[VAR_13_]]{{.}} : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
// CHECK:             scf.yield [[VAR_reinterpret_cast_5_]], [[VAR_reinterpret_cast_7_]], [[VAR_25_]], [[VAR_35_]], [[CST_0_]], [[VAR_reinterpret_cast_6_]] : memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
