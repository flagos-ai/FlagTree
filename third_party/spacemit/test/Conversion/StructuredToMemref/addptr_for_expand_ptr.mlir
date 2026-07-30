// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1

    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1

    // gep operand is another gep' output, which is passed into the loop as varible, used after update
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
      %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>

      %8 = tt.broadcast %7 : tensor<256x1xi32> -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 0], strides: [1, 0]

      %9 = tt.make_range {end = 512 : i32, start = 256 : i32} : tensor<256xi32>
      %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

      %11 = tt.broadcast %10 : tensor<1x256xi32> -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [0, 1]

      %12 = arith.addi %8, %11 : tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [1, 1]

      %13 = tt.expand_dims %ptr {axis = 1 : i32} : tensor<256x!tt.ptr<bf16>> -> tensor<256x1x!tt.ptr<bf16>>
      %14 = tt.broadcast %13 : tensor<256x1x!tt.ptr<bf16>> -> tensor<256x256x!tt.ptr<bf16>>

      %15 = tt.addptr %14, %12 : tensor<256x256x!tt.ptr<bf16>>, tensor<256x256xi32>
      // source: arg0, sizes: [256, 256], offsets: [1024 + i, 256], strides: [2, 1]

      // perform load
      %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x256x!tt.ptr<bf16>>
      tt.store %15, %16 : tensor<256x256x!tt.ptr<bf16>>
      // pointer updates
      %17 = tt.splat %i_c3 : i32 -> tensor<256xi32>
      // sizes: 256, offsets: 3, strides: 0
      %ptr_iter = tt.addptr %ptr, %17 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
      // source: arg0, sizes: 256, offsets: 1024 + i, strides: 4
      scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = scf.for [[VAR_arg7_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg8_:%.+]] = [[CST_1024_]]) -> (index) {
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.addi [[VAR_arg8_]], [[CST_256_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [256, 256], strides: [2, 1] : memref<*xbf16> to memref<256x256xbf16, strided<[2, 1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256x256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<256x256xbf16, strided<[2, 1], offset: ?>> to memref<256x256xbf16>
// CHECK:             [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256x256xbf16>
// CHECK:             bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_]] : (tensor<256x256xbf16>, memref<256x256xbf16, strided<[2, 1], offset: ?>>) -> ()
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_arg8_]], [[CST_3_]] : index
// CHECK:             scf.yield [[VAR_3_]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }
