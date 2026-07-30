// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s

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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @test_argmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_13_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i32
// CHECK:             linalg.yield [[VAR_14_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [4, 4], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}} : memref<*xf32> to memref<4x4xf32, strided<[?, ?]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4x4xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4x4xf32, strided<[?, ?]>> to memref<4x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<1x4xi32>) outs([[VAR_5_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_7_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_7_]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_9_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[VAR_4_]], [[VAR_6_]] : tensor<4x4xf32>, tensor<4x4xi32>) outs([[VAR_8_]], [[VAR_10_]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             ([[in_:.+]]: f32, [[in_1_:.+]]: i32, [[init:.+]]: f32, [[init_2:.+]]: i32) {
// CHECK-DAG:           [[VAR_13_1_:%.+]] = arith.cmpf oeq, [[in_]], [[init]] : f32
// CHECK-DAG:           [[VAR_14_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.andi [[VAR_13_1_]], [[VAR_14_1_]] : i1
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.cmpf ogt, [[in_]], [[init]] : f32
// CHECK:               [[VAR_17_:%.+]] = arith.ori [[VAR_16_]], [[VAR_15_]] : i1
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[in_]], [[init]] : f32
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[in_1_]], [[init_2]] : i32
// CHECK:               linalg.yield [[VAR_18_]], [[VAR_19_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_11_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_reduced_]]#1 : tensor<4xi32>) outs([[VAR_11_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: f32):
// CHECK:             [[VAR_13_2_:%.+]] = arith.sitofp [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_13_2_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           bufferization.materialize_in_destination [[VAR_12_]] in writable [[VAR_reinterpret_cast_0_]]
// CHECK:           return
// CHECK:         }

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
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @test_argmin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_13_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i32
// CHECK:             linalg.yield [[VAR_14_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [4, 4], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}} : memref<*xf32> to memref<4x4xf32, strided<[?, ?]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4x4xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4x4xf32, strided<[?, ?]>> to memref<4x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<1x4xi32>) outs([[VAR_5_]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           [[VAR_7_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_7_]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_9_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[VAR_4_]], [[VAR_6_]] : tensor<4x4xf32>, tensor<4x4xi32>) outs([[VAR_8_]], [[VAR_10_]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             ([[in_:.+]]: f32, [[in_1_:.+]]: i32, [[init:.+]]: f32, [[init_2:.+]]: i32) {
// CHECK-DAG:           [[VAR_13_1_:%.+]] = arith.cmpf oeq, [[in_]], [[init]] : f32
// CHECK-DAG:           [[VAR_14_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.andi [[VAR_13_1_]], [[VAR_14_1_]] : i1
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.cmpf olt, [[in_]], [[init]] : f32
// CHECK:               [[VAR_17_:%.+]] = arith.ori [[VAR_16_]], [[VAR_15_]] : i1
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[in_]], [[init]] : f32
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[in_1_]], [[init_2]] : i32
// CHECK:               linalg.yield [[VAR_18_]], [[VAR_19_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_11_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_reduced_]]#1 : tensor<4xi32>) outs([[VAR_11_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: f32):
// CHECK:             [[VAR_13_2_:%.+]] = arith.sitofp [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_13_2_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           bufferization.materialize_in_destination [[VAR_12_]] in writable [[VAR_reinterpret_cast_0_]]
// CHECK:           return
// CHECK:         }
