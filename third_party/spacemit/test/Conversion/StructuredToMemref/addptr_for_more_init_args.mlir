// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1
    %3 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg1, sizes: 256, offsets: 1024, strides: 1
    %_arg2, %_ptr_ld, %_arg3, %_ptr_st, %_arg4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%arg2 = %c1, %ptr_ld = %2, %arg3 = %c2, %ptr_st = %4, %arg4 = %c3) -> (index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index) {
        // perform load
        %5 = tt.load %ptr_ld {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_st, %5 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %cast3 = arith.index_cast %c3 : index to i32
        %6 = tt.splat %cast3 : i32 -> tensor<256xi32>
        %ptr_ld_iter = tt.addptr %ptr_ld, %6 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i*3, strides: 1
        %arg2_iter = arith.addi %arg2, %c3 : index
        %arg3_iter = arith.addi %arg3, %c3 : index
        %arg4_iter = arith.addi %arg4, %c3 : index
        %7 = arith.addi %arg2_iter, %arg3_iter : index
        %8 = arith.addi %7, %arg4_iter : index
        %cast8 = arith.index_cast %8 : index to i32
        %9 = tt.splat %cast8 : i32 -> tensor<256xi32>
        %ptr_st_iter = tt.addptr %ptr_st, %9 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg1, sizes: 256, offsets: 1024 + loop-carry variable*i, strides: 1
        scf.yield %arg2_iter, %ptr_ld_iter, %arg3_iter, %ptr_st_iter, %arg4_iter : index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]]:5 = scf.for [[VAR_arg8_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg9_:%.+]] = [[CST_1_]], [[VAR_arg10_:%.+]] = [[CST_1024_]], [[VAR_arg11_:%.+]] = [[CST_2_]], [[VAR_arg12_:%.+]] = [[CST_1024_]], [[VAR_arg13_:%.+]] = [[CST_3_]]) -> (index, index, index, index, index) {
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg12_]]{{.}}, sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg10_]]{{.}}, sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
// CHECK:             [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256xbf16>
// CHECK:             bufferization.materialize_in_destination [[VAR_1_]] in writable [[VAR_reinterpret_cast_]] : (tensor<256xbf16>, memref<256xbf16, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.addi [[VAR_arg10_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.addi [[VAR_arg9_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_arg11_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg13_]], [[CST_3_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.addi [[VAR_3_]], [[VAR_4_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[VAR_5_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_7_]] : index
// CHECK:             scf.yield [[VAR_3_]], [[VAR_2_]], [[VAR_4_]], [[VAR_8_]], [[VAR_5_]] : index, index, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
