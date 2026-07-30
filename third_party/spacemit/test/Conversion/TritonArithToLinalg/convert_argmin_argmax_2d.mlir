// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

// @triton.jit
// def test(
//     a_ptr, c_ptr, stride_am, stride_an
// ):
//     offs_am = tl.arange(0, 4)
//     offs_an = tl.arange(0, 4)
//     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
//     a = tl.load(a_ptrs)
//     m = tl.argmax(a, axis=1)
//     tl.store(c_ptr + tl.arange(0, 4), m)
//
// ret = triton.compiler.compile(
//     test,
//     signature=" *fp32,*fp32,i32,i32",
//     print_triton_ir_only=True,
// )

module {
  tt.func public @test_argmax(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4x1xi32>
    %3 = arith.muli %1, %2 : tensor<4x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x4xi32>
    %6 = arith.muli %4, %5 : tensor<1x4xi32>
    %7 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
    %8 = tt.broadcast %6 : tensor<1x4xi32> -> tensor<4x4xi32>
    %9 = arith.addi %7, %8 : tensor<4x4xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %12 = tt.load %11 : tensor<4x4x!tt.ptr<f32>>
    %13 = tt.broadcast %4 : tensor<1x4xi32> -> tensor<4x4xi32>
    %14:2 = "tt.reduce"(%12, %13) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
      %18 = arith.cmpf oeq, %arg4, %arg6 : f32
      %19 = arith.cmpi slt, %arg5, %arg7 : i32
      %20 = arith.andi %18, %19 : i1
      %21 = arith.cmpf ogt, %arg4, %arg6 : f32
      %22 = arith.ori %21, %20 : i1
      %23 = arith.select %22, %arg4, %arg6 : f32
      %24 = arith.select %22, %arg5, %arg7 : i32
      tt.reduce.return %23, %24 : f32, i32
    }) : (tensor<4x4xf32>, tensor<4x4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %16 = tt.addptr %15, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %17 = arith.sitofp %14#1 : tensor<4xi32> to tensor<4xf32>
    tt.store %16, %17 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}


// -----

// @triton.jit
// def test(
//     a_ptr, c_ptr, stride_am, stride_an
// ):
//     offs_am = tl.arange(0, 4)
//     offs_an = tl.arange(0, 4)
//     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
//     a = tl.load(a_ptrs)
//     m = tl.argmin(a, axis=1)
//     tl.store(c_ptr + tl.arange(0, 4), m)
//
// ret = triton.compiler.compile(
//     test,
//     signature=" *fp32,*fp32,i32,i32",
//     print_triton_ir_only=True,
// )

module {
  tt.func public @test_argmin(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4x1xi32>
    %3 = arith.muli %1, %2 : tensor<4x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x4xi32>
    %6 = arith.muli %4, %5 : tensor<1x4xi32>
    %7 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
    %8 = tt.broadcast %6 : tensor<1x4xi32> -> tensor<4x4xi32>
    %9 = arith.addi %7, %8 : tensor<4x4xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %12 = tt.load %11 : tensor<4x4x!tt.ptr<f32>>
    %13 = tt.broadcast %4 : tensor<1x4xi32> -> tensor<4x4xi32>
    %14:2 = "tt.reduce"(%12, %13) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
      %18 = arith.cmpf oeq, %arg4, %arg6 : f32
      %19 = arith.cmpi slt, %arg5, %arg7 : i32
      %20 = arith.andi %18, %19 : i1
      %21 = arith.cmpf olt, %arg4, %arg6 : f32
      %22 = arith.ori %21, %20 : i1
      %23 = arith.select %22, %arg4, %arg6 : f32
      %24 = arith.select %22, %arg5, %arg7 : i32
      tt.reduce.return %23, %24 : f32, i32
    }) : (tensor<4x4xf32>, tensor<4x4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %16 = tt.addptr %15, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %17 = arith.sitofp %14#1 : tensor<4xi32> to tensor<4xf32>
    tt.store %16, %17 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @test_argmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_28_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_29_:%.+]] = arith.index_cast [[VAR_28_]] : index to i32
// CHECK:             linalg.yield [[VAR_29_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<4x1xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_2_]] : i32) outs([[VAR_2_]] : tensor<4x1xi32>) -> tensor<4x1xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]], [[VAR_3_]] : tensor<4x1xi32>, tensor<4x1xi32>) outs([[VAR_expanded_]] : tensor<4x1xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_1_:%.+]] = arith.muli [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_1_]] : i32
// CHECK:           } -> tensor<4x1xi32>
// CHECK-DAG:       [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<1x4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_3_]] : i32) outs([[VAR_5_]] : tensor<1x4xi32>) -> tensor<1x4xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]], [[VAR_6_]] : tensor<1x4xi32>, tensor<1x4xi32>) outs([[VAR_expanded_0_]] : tensor<1x4xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_2_:%.+]] = arith.muli [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_2_]] : i32
// CHECK:           } -> tensor<1x4xi32>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_4_]] : tensor<4x1xi32>) outs([[VAR_8_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_10_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_7_]] : tensor<1x4xi32>) outs([[VAR_10_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_9_]], [[VAR_11_]] : tensor<4x4xi32>, tensor<4x4xi32>) outs([[VAR_9_]] : tensor<4x4xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_3_:%.+]] = arith.addi [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_3_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_13_:%.+]] = tensor.empty() : tensor<4x4x!tt.ptr<f32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_13_]] : tensor<4x4x!tt.ptr<f32>>) -> tensor<4x4x!tt.ptr<f32>>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_14_]], [[VAR_12_]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>) outs([[VAR_14_]] : tensor<4x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_28_4_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_28_4_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x4x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_15_MEM_:%.+]] = tt.load [[VAR_15_]] : tensor<4x4x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_17_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<1x4xi32>) outs([[VAR_17_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_19_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_19_]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_22_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_21_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[LOAD_VAR_15_MEM_]], [[VAR_18_]] : tensor<4x4xf32>, tensor<4x4xi32>) outs([[VAR_20_]], [[VAR_22_]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             ([[in_:%.*]]: f32, [[in_1_:%.*]]: i32, [[init_:%.*]]: f32, [[init_2_:%.*]]: i32) {
// CHECK-DAG:           [[VAR_28_5_:%.+]] = arith.cmpf oeq, [[in_]], [[init_]] : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.andi [[VAR_28_5_]], [[VAR_29_1_]] : i1
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[in_]], [[init_]] : f32
// CHECK:               [[VAR_32_:%.+]] = arith.ori [[VAR_31_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[in_]], [[init_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[in_1_]], [[init_2_]] : i32
// CHECK:               linalg.yield [[VAR_33_]], [[VAR_34_]] : f32, i32
// CHECK:             }
// CHECK:           [[VAR_23_:%.+]] = tensor.empty() : tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_24_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_23_]] : tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_25_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_24_]], [[VAR_1_]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>) outs([[VAR_24_]] : tensor<4x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_28_6_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_28_6_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_26_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_27_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_reduced_]]#1 : tensor<4xi32>) outs([[VAR_26_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: f32):
// CHECK:             [[VAR_28_7_:%.+]] = arith.sitofp [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_28_7_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.store [[VAR_25_]], [[VAR_27_]] : tensor<4x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @test_argmin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_28_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_29_:%.+]] = arith.index_cast [[VAR_28_]] : index to i32
// CHECK:             linalg.yield [[VAR_29_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<4x1xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_2_]] : i32) outs([[VAR_2_]] : tensor<4x1xi32>) -> tensor<4x1xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]], [[VAR_3_]] : tensor<4x1xi32>, tensor<4x1xi32>) outs([[VAR_expanded_]] : tensor<4x1xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_1_:%.+]] = arith.muli [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_1_]] : i32
// CHECK:           } -> tensor<4x1xi32>
// CHECK-DAG:       [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<1x4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_3_]] : i32) outs([[VAR_5_]] : tensor<1x4xi32>) -> tensor<1x4xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]], [[VAR_6_]] : tensor<1x4xi32>, tensor<1x4xi32>) outs([[VAR_expanded_0_]] : tensor<1x4xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_2_:%.+]] = arith.muli [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_2_]] : i32
// CHECK:           } -> tensor<1x4xi32>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_4_]] : tensor<4x1xi32>) outs([[VAR_8_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_10_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_7_]] : tensor<1x4xi32>) outs([[VAR_10_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_9_]], [[VAR_11_]] : tensor<4x4xi32>, tensor<4x4xi32>) outs([[VAR_9_]] : tensor<4x4xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_28_3_:%.+]] = arith.addi [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_28_3_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_13_:%.+]] = tensor.empty() : tensor<4x4x!tt.ptr<f32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_13_]] : tensor<4x4x!tt.ptr<f32>>) -> tensor<4x4x!tt.ptr<f32>>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_14_]], [[VAR_12_]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>) outs([[VAR_14_]] : tensor<4x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_28_4_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_28_4_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x4x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_15_MEM_:%.+]] = tt.load [[VAR_15_]] : tensor<4x4x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_17_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<1x4xi32>) outs([[VAR_17_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_19_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_19_]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_22_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_21_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[LOAD_VAR_15_MEM_]], [[VAR_18_]] : tensor<4x4xf32>, tensor<4x4xi32>) outs([[VAR_20_]], [[VAR_22_]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             ([[in_]]: f32, [[in_]]_1: i32, [[in_]]it: f32, [[in_]]it_2: i32) {
// CHECK-DAG:           [[VAR_28_5_:%.+]] = arith.cmpf oeq, [[in_]], [[in_]]it : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.andi [[VAR_28_5_]], [[VAR_29_1_]] : i1
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf olt, [[in_]], [[in_]]it : f32
// CHECK:               [[VAR_32_:%.+]] = arith.ori [[VAR_31_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[in_]], [[in_]]it : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[in_1_]], [[init_2_]] : i32
// CHECK:               linalg.yield [[VAR_33_]], [[VAR_34_]] : f32, i32
// CHECK:             }
// CHECK:           [[VAR_23_:%.+]] = tensor.empty() : tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_24_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_23_]] : tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_25_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_24_]], [[VAR_1_]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>) outs([[VAR_24_]] : tensor<4x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_28_6_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_28_6_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_26_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_27_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_reduced_]]#1 : tensor<4xi32>) outs([[VAR_26_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: f32):
// CHECK:             [[VAR_28_7_:%.+]] = arith.sitofp [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_28_7_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.store [[VAR_25_]], [[VAR_27_]] : tensor<4x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
