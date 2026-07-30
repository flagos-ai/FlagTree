// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

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
      // source: arg0, sizes: [256, 256], offsets: [1024 + i*3, 256], strides: [2, 1]

      // perform load
      %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x256x!tt.ptr<bf16>>
      tt.store %15, %16 : tensor<256x256x!tt.ptr<bf16>>
      // pointer updates
      %17 = tt.splat %i_c3 : i32 -> tensor<256xi32>
      // sizes: 256, offsets: 3, strides: 0
      %ptr_iter = tt.addptr %ptr, %17 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
      // source: arg0, sizes: 256, offsets: 1024 + i*3, strides: 4
      scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg2_:%.+]] = [[CST_1024_]]) -> (index) {
// CHECK-DAG:         [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256, 256], strides: {{.}}[[CST_2_]], 1], offsets: {{.}}[[VAR_arg2_]], 256], shape: [0, 0], order: [] : <bf16> to tensor<256x256x!tt.ptr<bf16>>
// CHECK:             [[VAR_2_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<256x256x!tt.ptr<bf16>>) -> tensor<256x256xbf16>
// CHECK:             "tts.store"([[VAR_1_]], [[VAR_2_]]) <{static_mask_dims = array<i64>}> : (tensor<256x256x!tt.ptr<bf16>>, tensor<256x256xbf16>) -> ()
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_3_]] : index
// CHECK:             scf.yield [[VAR_3_]] : index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
