// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : i32
  )
  {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32}:tensor<1024xi32>
    %2 = tt.splat %0 : i32 -> tensor<1024xi32>
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    //%3: splat(%0) + range(0, 1024)
    //%3: offset = %0, size = 1024, stride = 1
    // vector and scalar are both constant
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %c10 = arith.constant 10 : i32
    %5 = tt.splat %c10 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%c10)*range(2048, 4096);
    //%6: offset = %c10*2048, size = 1024, stride = %c10*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %c10*2048 + %0, size = 1024, stride = %c10*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0 offset = %c10*2048 + pid0, size = 1024, stride = %c10*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1, offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[CST_11_:%.+]] = arith.constant 11 : index
// CHECK-DAG:       [[CST_20480_:%.+]] = arith.constant 20480 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[CST_20480_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: {{.}}[[CST_11_]]{{.}}, offsets: {{.}}[[VAR_3_]]{{.}}, shape: [0], order: [] : <bf16> to tensor<1024x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_1_]]{{.}}, shape: [0], order: [] : <bf16> to tensor<1024x!tt.ptr<bf16>>
// CHECK:           [[VAR_6_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16>>) -> tensor<1024xbf16>
// CHECK:           "tts.store"([[VAR_5_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16>>, tensor<1024xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
