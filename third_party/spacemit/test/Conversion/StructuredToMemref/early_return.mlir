// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
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
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_1_]] : i32) outs([[VAR_0_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = affine.load [[VAR_reinterpret_cast_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           [[VAR_4_:%.+]] = arith.cmpf oeq, [[LOAD_VAR_reinterpret_cast_MEM_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK:           cf.cond_br [[VAR_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           return
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_9_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : index to i32
// CHECK:             linalg.yield [[VAR_10_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_5_]], [[VAR_1_]] : tensor<4xi32>, tensor<4xi32>) outs([[VAR_5_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: i32):
// CHECK:             [[VAR_9_1_:%.+]] = arith.addi [[IN_1_]], [[IN_2_]] : i32
// CHECK:             linalg.yield [[VAR_9_1_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]] : tensor<4xi32>) outs([[VAR_7_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_9_2_:%.+]] = arith.sitofp [[IN_4_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_9_2_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
