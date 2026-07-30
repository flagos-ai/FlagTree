// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

// @triton.jit
// def test_cumsum_op(
//     input_ptr, output_ptr, n_columns
// ):
//     row = tl.program_id(axis=0)
//     row_start = row * n_columns
//     columns = tl.arange(0, 4096)
//     offsets = row_start + columns
//     data = tl.load(input_ptr + offsets)
//     result = tl.cumsum(data, axis=0)
//     tl.store(output_ptr + offsets, result)
//
// ret = triton.compiler.compile(
//     test_cumsum_op,
//     signature=" *fp32,*i32,i32",
//     print_triton_ir_only=True,
// )
// print(ret.asm["ttir"])

module {
  tt.func public @test_cumsum_op_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8 = "tt.scan"(%7) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %12 = arith.addf %arg3, %arg4 : f32
      tt.scan.return %12 : f32
    }) : (tensor<4096xf32>) -> tensor<4096xf32>
    %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>>
    %10 = tt.addptr %9, %4 : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>
    %11 = arith.fptosi %8 : tensor<4096xf32> to tensor<4096xi32>
    tt.store %10, %11 : tensor<4096x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_cumsum_op_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_17_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_18_:%.+]] = arith.index_cast [[VAR_17_]] : index to i32
// CHECK:             linalg.yield [[VAR_18_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_3_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.fill ins([[VAR_0_]] : i32) outs([[VAR_3_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_2_]] : tensor<4096xi32>, tensor<4096xi32>) outs([[VAR_4_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_]]_0: i32, [[out_]]: i32):
// CHECK:             [[VAR_17_1_:%.+]] = arith.addi [[in_]], [[in_]]_0 : i32
// CHECK:             linalg.yield [[VAR_17_1_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_6_]] : tensor<4096x!tt.ptr<f32>>) -> tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_5_]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs([[VAR_7_]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_17_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_17_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_8_MEM_:%.+]] = tt.load [[VAR_8_]] : tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_10_:%.+]] = tensor.empty() : tensor<4096xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins([[LOAD_VAR_8_MEM_]] : tensor<4096xf32>) outs([[VAR_10_]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tensor.empty() : tensor<4096x!tt.ptr<i32>>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<i32>) outs([[VAR_12_]] : tensor<4096x!tt.ptr<i32>>) -> tensor<4096x!tt.ptr<i32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_13_]], [[VAR_5_]] : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>) outs([[VAR_13_]] : tensor<4096x!tt.ptr<i32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<i32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<i32>):
// CHECK:             [[VAR_17_3_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i32>, i32
// CHECK:             linalg.yield [[VAR_17_3_]] : !tt.ptr<i32>
// CHECK:           } -> tensor<4096x!tt.ptr<i32>>
// CHECK:           [[VAR_15_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_11_]] : tensor<4096xf32>) outs([[VAR_15_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[in_]]: f32, [[out_]]: i32):
// CHECK:             [[VAR_17_4_:%.+]] = arith.fptosi [[in_]] : f32 to i32
// CHECK:             linalg.yield [[VAR_17_4_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           tt.store [[VAR_14_]], [[VAR_16_]] : tensor<4096x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }
