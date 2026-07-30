// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

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
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4096xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4096xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tensor.empty() : tensor<4096xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins([[VAR_2_]] : tensor<4096xf32>) outs([[VAR_3_]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [4096], strides: [1] : memref<*xi32> to memref<4096xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]] : tensor<4096xf32>) outs([[VAR_5_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: i32):
// CHECK:             [[VAR_7_:%.+]] = arith.fptosi [[IN_0_]] : f32 to i32
// CHECK:             linalg.yield [[VAR_7_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_6_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<4096xi32>, memref<4096xi32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
