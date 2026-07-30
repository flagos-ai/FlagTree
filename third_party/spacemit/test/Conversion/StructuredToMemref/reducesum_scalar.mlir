// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128xbf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xbf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_1_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_0_]] : tensor<128xbf16>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: bf16, [[init_:%.+]]: f32) {
// CHECK:               [[VAR_3_:%.+]] = arith.extf [[in_]] : bf16 to f32
// CHECK:               [[VAR_4_:%.+]] = arith.addf [[VAR_3_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_4_]] : f32
// CHECK:             }
