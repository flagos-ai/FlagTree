// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
    // Mimic a scenario where the raw pointer points to a buffer with dimension (1024, 1024)
    // in row-major, but the actual tensor size is (arg2, arg3).
    // We are trying to load a 128x256 sub-buffer starting at (2, 3).
    // The resulting memref:
    //  offset = 3074
    //  size[1] = 128
    //  size[0] = 256
    //  stride[0] = 1024
    //  stride[1] = 1
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    // horizontal index
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c2 = arith.constant 2 : i32
    %c2tensor = tt.splat %c2 : i32 -> tensor<128xi32>
    %offset2 = arith.addi %2, %c2tensor : tensor<128xi32>
    %3 = tt.expand_dims %offset2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x256xi32>
    // vertical index
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %c3 = arith.constant 3 : i32
    %c3tensor = tt.splat %c3 : i32 -> tensor<256xi32>
    %offset5 = arith.addi %5, %c3tensor : tensor<256xi32>
    %c1024 = arith.constant 1024 : i32
    %c1024tensor = tt.splat %c1024 : i32 -> tensor<256xi32>
    %scale5 = arith.muli %offset5, %c1024tensor : tensor<256xi32>
    %6 = tt.expand_dims %scale5 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %7 = tt.broadcast %6 : tensor<1x256xi32> -> tensor<128x256xi32>
    // combined index
    %index = arith.addi %4, %7 : tensor<128x256xi32>
    %ldptr = tt.addptr %0, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %stptr = tt.addptr %1, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    // other value for masked load
    %cnan = arith.constant 0xFF80 : bf16
    %nans = tt.splat %cnan : bf16 -> tensor<128x256xbf16>
    // horizontal mask
    %8 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %9 = arith.cmpi slt, %offset2, %8 : tensor<128xi32>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %11 = tt.broadcast %10 : tensor<128x1xi1> -> tensor<128x256xi1>
    // vertical mask
    %12 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %13 = arith.cmpi slt, %offset5, %12 : tensor<256xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %15 = tt.broadcast %14 : tensor<1x256xi1> -> tensor<128x256xi1>
    // combined mask
    %mask = arith.andi %11, %15 : tensor<128x256xi1>
    // dim0 = min(%arg2, 128), dim1 = min(%arg3, 256)
    // TODO: need reinterpret cast
    %buff = tt.load %ldptr, %mask, %nans : tensor<128x256x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_130_:%.+]] = arith.constant 130 : index
// CHECK-DAG:       [[CST_259_:%.+]] = arith.constant 259 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:       [[CST_3074_:%.+]] = arith.constant 3074 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_3074_]]{{.}}, sizes: [128, 256], strides: [1, 1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, 1024], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[CST_3074_]]{{.}}, sizes: [128, 256], strides: [1, 1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, 1024], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = arith.minsi [[VAR_0_]], [[CST_130_]] : index
// CHECK:           [[VAR_2_:%.+]] = arith.maxsi [[VAR_1_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.subi [[VAR_2_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[CST_259_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.maxsi [[VAR_5_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.subi [[VAR_6_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.minsi [[VAR_3_]], [[CST_128_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.minsi [[VAR_7_]], [[CST_256_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128x256xbf16>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_128_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_9_]], [[CST_256_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.ori [[VAR_10_]], [[VAR_11_]] : i1
// CHECK:           scf.if [[VAR_12_]] {
// CHECK:             linalg.fill ins([[CST_0_]] : bf16) outs([[RES_]] : memref<128x256xbf16>)
// CHECK:           }
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0, 0] {{.}}[[VAR_8_]], [[VAR_9_]]{{.}} [1, 1] : memref<128x256xbf16, strided<[1, 1024], offset: ?>> to memref<?x?xbf16, strided<[1, 1024], offset: ?>>
// CHECK-DAG:       [[VAR_subview_1_:%.+]] = memref.subview [[RES_]][0, 0] {{.}}[[VAR_8_]], [[VAR_9_]]{{.}} [1, 1] : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_1_]] : memref<?x?xbf16, strided<[1, 1024], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
// CHECK:           [[VAR_13_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x256xbf16>
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_13_]][0, 0] {{.}}[[VAR_8_]], [[VAR_9_]]{{.}} [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
// CHECK-DAG:       [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_0_]][0, 0] {{.}}[[VAR_8_]], [[VAR_9_]]{{.}} [1, 1] : memref<128x256xbf16, strided<[1, 1024], offset: ?>> to memref<?x?xbf16, strided<[1, 1024], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_2_]] : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1, 1024], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
