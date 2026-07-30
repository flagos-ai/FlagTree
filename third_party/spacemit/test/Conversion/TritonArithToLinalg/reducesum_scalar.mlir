// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
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
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_8_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i32
// CHECK:             linalg.yield [[VAR_9_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<128x!tt.ptr<bf16>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<bf16>) outs([[VAR_2_]] : tensor<128x!tt.ptr<bf16>>) -> tensor<128x!tt.ptr<bf16>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>) outs([[VAR_3_]] : tensor<128x!tt.ptr<bf16>>) {
// CHECK:           ^bb0([[in_:%.*]]: !tt.ptr<bf16>, [[in_0_:%.*]]: i32, [[out_:%.*]]: !tt.ptr<bf16>):
// CHECK:             [[VAR_8_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield [[VAR_8_1_]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x!tt.ptr<bf16>>
// CHECK-DAG:       [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<128x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_6_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_6_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[LOAD_VAR_4_MEM_]] : tensor<128xbf16>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:%.*]]: bf16, [[init_:%.*]]: f32) {
// CHECK:               [[VAR_8_2_:%.+]] = arith.extf [[in_]] : bf16 to f32
// CHECK:               [[VAR_9_1_:%.+]] = arith.addf [[VAR_8_2_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_9_1_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK:           [[VAR_7_:%.+]] = arith.truncf [[VAR_extracted_]] : f32 to bf16
// CHECK:           tt.store [[PARAM_1_]], [[VAR_7_]] : !tt.ptr<bf16>
// CHECK:           return
// CHECK:         }
