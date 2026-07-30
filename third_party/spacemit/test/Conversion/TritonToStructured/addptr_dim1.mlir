// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

    %splat_arg0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %2 = tt.addptr %splat_arg0, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    // source = %arg0, offset = [0, 0], size = [1, 256], stride = [0, 1]

    // 1x256 pointer should have meaningful stride in outer dimension
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<1x256x!tt.ptr<bf16>>

    %4 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    // 1x256 pointer should have meaningful stride in outer dimension
    %5 = tt.addptr %2, %4 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    // source = %arg0, offset = [%arg1, 0], size = [1, 256], stride = [0, 1]

    tt.store %5, %3 : tensor<1x256x!tt.ptr<bf16>>

    %10 = arith.constant 0.0 : bf16
    %11 = tt.splat %10 : bf16 -> tensor<4x256xbf16>

    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %c256 = arith.constant 256 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %11, %ptr = %2) -> (tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>) {
        %bptr = tt.broadcast %ptr : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
        // source = %arg0, offset = [0, 0], size = [4, 256], stride = [0, 1]

        %20 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
        %i_i32 = arith.index_cast %i : index to i32
        %21 = arith.muli %c256, %i_i32 : i32
        %22 = tt.splat %21 : i32 -> tensor<4xi32>
        %23 = arith.muli %20, %22 : tensor<4xi32>
        %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
        %25 = tt.broadcast %24 : tensor<4x1xi32> -> tensor<4x256xi32>
        // offset = [0, 0], size = [4, 256], stride = [i*256, 1]

        // %bptr should have zero stride and %30 should have correct stride
        %30 = tt.addptr %bptr, %25 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source = %arg0, offset = [0, 0], size = [4, 256], stride = [i*256, 1]

        %31 = tt.load %30 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %32 = arith.addf %sum_iter, %31 : tensor<4x256xbf16>

        %40 = tt.splat %c256 : i32 -> tensor<1x256xi32>
        %41 = tt.addptr %ptr, %40 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
        // source = %arg0, offset = [i*256, 0], size = [4, 256], stride = [i*256, 1]

        scf.yield %32, %41 : tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>
    }

    %31 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    %splat_c256 = tt.splat %c256 : i32 -> tensor<4xi32>
    %32 = arith.muli %31, %splat_c256 : tensor<4xi32>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %34 = tt.broadcast %33 : tensor<4x1xi32> -> tensor<4x256xi32>
    %35 = tt.broadcast %2 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    tt.store %36, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<4x256xbf16>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16>>) -> tensor<1x256xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1, 256], strides: [0, 1], offsets: {{.}}[[VAR_2_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_3_]], [[VAR_1_]]) <{static_mask_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xbf16>) -> ()
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_cst_]], [[VAR_arg4_:%.+]] = [[CST_0_]]) -> (tensor<4x256xbf16>, index) {
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.index_cast [[VAR_arg2_]] : index to i32
// CHECK:             [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_256_1_]] : i32
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i32 to index
// CHECK:             [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: {{.}}[[VAR_8_]], [[CST_1_]]{{.}}, offsets: {{.}}[[VAR_arg4_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:             [[VAR_10_:%.+]] = "tts.load"([[VAR_9_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addf [[VAR_arg3_]], [[VAR_10_]] : tensor<4x256xbf16>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_arg4_]], [[CST_256_]] : index
// CHECK:             scf.yield [[VAR_11_]], [[VAR_12_]] : tensor<4x256xbf16>, index
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: {{.}}[[CST_256_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_5_]], [[VAR_4_]]#0) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
