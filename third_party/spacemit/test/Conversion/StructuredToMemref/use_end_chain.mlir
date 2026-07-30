// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
  %5 = tt.make_range {end = 1152 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 1
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,1]
  %7 = tt.broadcast %6 : tensor<1x128xi32> -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,6]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [1,6]
  // mixed use
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = arith.sitofp %14 : tensor<256x128xi32> to tensor<256x128xbf16>
  tt.store %18, %20 : tensor<256x128x!tt.ptr<bf16>>
  tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : i32
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : i32
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<256x128xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_6_]] : i32) outs([[VAR_0_]] : tensor<256x128xi32>) -> tensor<256x128xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_2_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_13_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i32
// CHECK:             [[VAL_24:%.+]] = arith.addi [[VAR_14_]], [[CST_512_]] : i32
// CHECK:             linalg.yield [[VAL_24]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_3_]] {{.}}[0, 1]{{.}} output_shape [256, 1] : tensor<256xi32> into tensor<256x1xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<256x1xi32>) outs([[VAR_0_]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32):
// CHECK:             linalg.yield [[IN_1_]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_5_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[IN_3_:%.+]]: i32):
// CHECK:             [[VAR_13_1_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_14_1_:%.+]] = arith.index_cast [[VAR_13_1_]] : index to i32
// CHECK:             [[VAL_25:%.+]] = arith.addi [[VAR_14_1_]], [[CST_1024_]] : i32
// CHECK:             linalg.yield [[VAL_25]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_6_]] {{.}}[0, 1]{{.}} output_shape [1, 128] : tensor<128xi32> into tensor<1x128xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<1x128xi32>) outs([[VAR_0_]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: i32):
// CHECK:             linalg.yield [[IN_4_]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_7_]], [[VAR_1_]] : tensor<256x128xi32>, tensor<256x128xi32>) outs([[VAR_7_]] : tensor<256x128xi32>) {
// CHECK:           ^bb0([[IN_6_:%.+]]: i32, [[IN_7_:%.+]]: i32, [[IN_8_:%.+]]: i32):
// CHECK:             [[VAR_13_2_:%.+]] = arith.muli [[IN_6_]], [[IN_7_]] : i32
// CHECK:             linalg.yield [[VAR_13_2_]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_4_]], [[VAR_8_]] : tensor<256x128xi32>, tensor<256x128xi32>) outs([[VAR_4_]] : tensor<256x128xi32>) {
// CHECK:           ^bb0([[IN_9_:%.+]]: i32, [[IN_10_:%.+]]: i32, [[IN_11_:%.+]]: i32):
// CHECK:             [[VAR_13_3_:%.+]] = arith.addi [[IN_9_]], [[IN_10_]] : i32
// CHECK:             linalg.yield [[VAR_13_3_]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [6656], sizes: [256, 128], strides: [1, 6] : memref<*xbf16> to memref<256x128xbf16, strided<[1, 6], offset: 6656>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<256x128xbf16, strided<[1, 6], offset: 6656>> to memref<256x128xbf16>
// CHECK:           [[VAR_10_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination [[VAR_10_]] in writable [[VAR_reinterpret_cast_]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, 6], offset: 6656>>) -> ()
// CHECK:           [[VAR_11_:%.+]] = tensor.empty() : tensor<256x128xbf16>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_9_]] : tensor<256x128xi32>) outs([[VAR_11_]] : tensor<256x128xbf16>) {
// CHECK:           ^bb0([[IN_12_:%.+]]: i32, [[IN_13_:%.+]]: bf16):
// CHECK:             [[VAR_13_4_:%.+]] = arith.sitofp [[IN_12_]] : i32 to bf16
// CHECK:             linalg.yield [[VAR_13_4_]] : bf16
// CHECK:           } -> tensor<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination [[VAR_12_]] in writable [[VAR_reinterpret_cast_]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, 6], offset: 6656>>) -> ()
// CHECK:           return
// CHECK:         }
