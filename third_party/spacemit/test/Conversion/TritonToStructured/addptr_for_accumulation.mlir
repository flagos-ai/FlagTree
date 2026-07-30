// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32,
    %arg4 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %c5 = arith.constant 5 : i32
    %splat6 = tt.splat %c5 : i32 -> tensor<4x256xi32>
    // scalar = 5
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>> // Why is the input unknown
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %19 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>> // this will be replaced with a memref.copy
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %19, %ptr_iter = %12) -> (tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>) {
        %20 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %sum = arith.addf %sum_iter, %20 : tensor<4x256xbf16>
        // pointer updates
        %17 = tt.splat %i_c3 : i32 -> tensor<4x256xi32>
        // offset: [3, 0], size = [4, 256], stride [0, 0]
        %ptr = tt.addptr %ptr_iter, %17 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source: %arg1, offset = [%arg3+%i, 0], size = [4, 256], stride = [1, 5]
        scf.yield %sum, %ptr : tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>
    }
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_0_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg6_:%.+]] = [[VAR_2_]], [[VAR_arg7_:%.+]] = [[VAR_3_]]) -> (tensor<4x256xbf16>, index) {
// CHECK-DAG:         [[VAR_7_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4, 256], strides: {{.}}[[CST_1_]], [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_arg7_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:             [[VAR_8_:%.+]] = "tts.load"([[VAR_7_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addf [[VAR_arg6_]], [[VAR_8_]] : tensor<4x256xbf16>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_3_]] : index
// CHECK:             scf.yield [[VAR_9_]], [[VAR_10_]] : tensor<4x256xbf16>, index
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_6_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_5_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_6_]], [[VAR_4_]]#0) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
