// RUN: triton-shared-opt --triton-to-unstructured --canonicalize %s | FileCheck %s

module {
  tt.func public @test_scalar_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    %2 = scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg2 = %1) -> (!tt.ptr<f32>)  : i32 {
      %3 = arith.muli %arg1, %c2_i32 : i32
      %4:2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %3, %arg5 = %arg2) -> (i32, !tt.ptr<f32>)  : i32 {
        %5 = arith.addi %arg4, %arg3 : i32
        %6 = arith.sitofp %5 : i32 to f32
        tt.store %arg5, %6 : !tt.ptr<f32>
        %7 = tt.addptr %arg5, %c1_i32 : !tt.ptr<f32>, i32
        scf.yield %5, %7 : i32, !tt.ptr<f32>
      }
      scf.yield %4#1 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @test_scalar_store([[PARAM_0_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] iter_args([[VAR_arg2_:%.+]] = [[VAR_0_]]) -> (i32)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.muli [[VAR_arg1_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg3_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg4_:%.+]] = [[VAR_2_]], [[VAR_arg5_:%.+]] = [[VAR_arg2_]]) -> (i32, i32)  : i32 {
// CHECK-DAG:           [[VAR_4_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_arg3_]] : i32
// CHECK:               [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i32 to f32
// CHECK:               tts.scatter [[VAR_5_]] into [[PARAM_0_]]{{.}}[[VAR_arg5_]]{{.}} : f32 into (<f32>, i32)
// CHECK:               [[VAR_6_:%.+]] = arith.addi [[VAR_arg5_]], [[CST_1_]] : i32
// CHECK:               scf.yield [[VAR_4_]], [[VAR_6_]] : i32, i32
// CHECK:             }
// CHECK:             scf.yield [[VAR_3_]]#1 : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
