// RUN: triton-shared-opt --triton-to-structured --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @add_ptr_into_make_block_ptr(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c32768_i64 : i64
    %3 = tt.addptr %arg0, %2 : !tt.ptr<bf16>, i64
    %4 = tt.make_tensor_ptr %3, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %5 = tt.addptr %arg1, %2 : !tt.ptr<bf16>, i64
    %6 = tt.make_tensor_ptr %5, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %7 = tt.load %4 : !tt.ptr<tensor<512x64xbf16>>
    tt.store %6, %7 : !tt.ptr<tensor<512x64xbf16>>
    tt.return
  }
}

// CHECK:         tt.func public @add_ptr_into_make_block_ptr([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_32768_:%.+]] = arith.constant 32768 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.extsi [[VAR_0_]] : i32 to i64
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_32768_]] : i64
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [512, 64], strides: {{.}}[[CST_64_]], [[CST_1_]]{{.}}, offsets: {{.}}[[VAR_3_]], [[CST_0_]]{{.}}, shape: {{.}}[[CST_512_]], [[CST_64_]]{{.}}, order: [1, 0] : <bf16> to !tt.ptr<tensor<512x64xbf16>>
// CHECK-DAG:       [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [512, 64], strides: {{.}}[[CST_64_]], [[CST_1_]]{{.}}, offsets: {{.}}[[VAR_3_]], [[CST_0_]]{{.}}, shape: {{.}}[[CST_512_]], [[CST_64_]]{{.}}, order: [1, 0] : <bf16> to !tt.ptr<tensor<512x64xbf16>>
// CHECK:           [[VAR_6_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<512x64xbf16>>) -> tensor<512x64xbf16>
// CHECK:           "tts.store"([[VAR_5_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<512x64xbf16>>, tensor<512x64xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
