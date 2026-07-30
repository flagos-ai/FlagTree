// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %c7_i32 = arith.constant 7 : i32
    %splat_c7_i32 = tt.splat %c7_i32 : i32 -> tensor<128xi32>
    %splat_c7_bf16 = arith.sitofp %splat_c7_i32 : tensor<128xi32> to tensor<128xbf16>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %splat_c7_bf16 : tensor<128x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_7_dot_000000_:%.+]] = arith.constant 7.000000e+00 : bf16
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = arith.minsi [[VAR_0_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.maxsi [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128xbf16>
// CHECK:           [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_2_]], [[CST_128_]] : index
// CHECK:           scf.if [[VAR_3_]] {
// CHECK:             linalg.fill ins([[CST_7_dot_000000_]] : bf16) outs([[RES_]] : memref<128xbf16>)
// CHECK:           }
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK-DAG:       [[VAR_subview_1_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_1_]] : memref<?xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xbf16>
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_4_]][0] {{.}}[[VAR_2_]]{{.}} [1] : tensor<128xbf16> to tensor<?xbf16>
// CHECK-DAG:       [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_0_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_2_]] : (tensor<?xbf16>, memref<?xbf16, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
