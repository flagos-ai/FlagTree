// XFAIL: *
// Note: PtrAnalysis pass can create a tts.makeptr for below pattern:
//         %3 = arith.constant 0 : index
//         %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
//         %7 = tt.addptr %6, %3 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
//       But not if creating constant 0 and add it to a pointer is optimized away:
//         %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
//       A patch that rewrites tt.splat in such case will be sent separately.
// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @num_programs(%arg0: !tt.ptr<i32>) {
    %0 = tt.get_num_programs x : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.get_num_programs z : i32
    %3 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %4 = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>
    %5 = tt.make_range {end = 3 : i32, start = 2 : i32} : tensor<1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %7 = tt.addptr %6, %3 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %8 = tt.splat %0 : i32 -> tensor<1xi32>
    tt.store %7, %8 : tensor<1x!tt.ptr<i32>>
    %9 = tt.addptr %6, %4 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %10 = tt.splat %1 : i32 -> tensor<1xi32>
    tt.store %9, %10 : tensor<1x!tt.ptr<i32>>
    %11 = tt.addptr %6, %5 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %12 = tt.splat %2 : i32 -> tensor<1xi32>
    tt.store %11, %12 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @num_programs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<1xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[PARAM_1_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_1_]] in writable [[VAR_reinterpret_cast_]] : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_1_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = linalg.fill ins([[PARAM_2_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_2_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_3_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_3_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
