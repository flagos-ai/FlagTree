// RUN: triton-shared-opt --triton-to-unstructured="offset-bit-width=64" %s | FileCheck %s

module {
  tt.func public @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.get_program_id x : i32
    %2 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.extsi %2 : tensor<64xi32> to tensor<64xi64>
    %5 = tt.splat %3 : i64 -> tensor<64xi64>
    %6 = arith.addi %4, %5 : tensor<64xi64>
    %7 = tt.splat %1 : i32 -> tensor<64xi32>
    %8 = arith.addi %2, %7 : tensor<64xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %10 = tt.addptr %9, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi64>
    %11 = tt.addptr %10, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %13:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %11, %arg4 = %0) -> (tensor<64x!tt.ptr<f32>>, tensor<64xi32>)  : i32 {
      %14 = tt.load %arg3 : tensor<64x!tt.ptr<f32>>
      %15 = tt.addptr %12, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %15, %14 : tensor<64x!tt.ptr<f32>>
      %16 = tt.addptr %arg3, %7 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %17 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %16, %17 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @gather_simple([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<64> : tensor<64xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<10> : tensor<64xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.get_program_id x : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.divsi [[VAR_0_]], [[VAR_cst_0_]] : tensor<64xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.extsi [[VAR_1_]] : i32 to i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.extsi [[VAR_2_]] : tensor<64xi32> to tensor<64xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = tt.splat [[VAR_3_]] : i64 -> tensor<64xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[VAR_5_]] : tensor<64xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = tt.splat [[VAR_1_]] : i32 -> tensor<64xi32>
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_2_]], [[VAR_7_]] : tensor<64xi32>
// CHECK:           [[VAR_9_:%.+]] = arith.extsi [[VAR_8_]] : tensor<64xi32> to tensor<64xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.addi [[VAR_6_]], [[VAR_9_]] : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_10_]], [[VAR_arg4_:%.+]] = [[VAR_0_]]) -> (tensor<64xi64>, tensor<64xi32>)  : i32 {
// CHECK-DAG:         [[VAR_12_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_arg3_]]{{.}} : (<f32>, tensor<64xi64>) -> tensor<64xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.extsi [[VAR_arg4_]] : tensor<64xi32> to tensor<64xi64>
// CHECK:             tts.scatter [[VAR_12_]] into [[PARAM_1_]]{{.}}[[VAR_13_]]{{.}} : tensor<64xf32> into (<f32>, tensor<64xi64>)
// CHECK:             [[VAR_14_:%.+]] = arith.extsi [[VAR_7_]] : tensor<64xi32> to tensor<64xi64>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addi [[VAR_arg3_]], [[VAR_14_]] : tensor<64xi64>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_cst_]] : tensor<64xi32>
// CHECK:             scf.yield [[VAR_15_]], [[VAR_16_]] : tensor<64xi64>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
