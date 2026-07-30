// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse --triton-arith-to-linalg %s | FileCheck %s
// This is the IR for the following triton program:
//    @triton.jit
//    def test_1(in0, out0):
//        pid = tl.program_id(0)
//        id = tl.load(in0 + pid)
//        if id == -1:
//            return
//        offs = 1 + tl.arange(0, 4)
//        out_offs = tl.arange(0, 4)
//        tl.store(out0 + out_offs, offs)

module {
  tt.func public @test_1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant dense<1> : tensor<4xi32>
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    %2 = tt.load %1 : !tt.ptr<f32>
    %3 = arith.cmpf oeq, %2, %cst : f32
    cf.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %5 = arith.addi %4, %cst_0 : tensor<4xi32>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %7 = tt.addptr %6, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %8 = arith.sitofp %5 : tensor<4xi32> to tensor<4xf32>
    tt.store %7, %8 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_1_]] : i32) outs([[VAR_0_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.addptr [[PARAM_0_]], [[PARAM_5_]] : !tt.ptr<f32>, i32
// CHECK:           [[LOAD_VAR_2_MEM_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<f32>
// CHECK:           [[VAR_4_:%.+]] = arith.cmpf oeq, [[LOAD_VAR_2_MEM_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK:           cf.cond_br [[VAR_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           return
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_5_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_11_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]] : index to i32
// CHECK:             linalg.yield [[VAR_12_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_1_]] : tensor<4xi32>, tensor<4xi32>) outs([[VAR_6_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: i32):
// CHECK:             [[VAR_11_1_:%.+]] = arith.addi [[IN_1_]], [[IN_2_]] : i32
// CHECK:             linalg.yield [[VAR_11_1_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<4xi32>) outs([[VAR_9_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_11_2_:%.+]] = arith.sitofp [[IN_4_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_11_2_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           "tts.store"([[VAR_8_]], [[VAR_10_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           return
// CHECK:         }
