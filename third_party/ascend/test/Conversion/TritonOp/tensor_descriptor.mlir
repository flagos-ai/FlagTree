// RUN: triton-adapter-opt %s --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// dtype: uint16
module {
  tt.func public @triton_tensor_descriptor_function_2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i16>, <tensor<8x32xui16>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i16>, <tensor<8x32xui16>>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0[%3, %5] : !tt.tensordesc<tensor<8x32xui16>> -> tensor<8x32xi16>
    tt.descriptor_store %1[%3, %5], %6 : !tt.tensordesc<tensor<8x32xui16>>, tensor<8x32xi16>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_tensor_descriptor_function_2d
// CHECK:       [[MEMREF_0:%.*]] = memref.reinterpret_cast [[VAL_0:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi16> to memref<8x32xi16, strided<[128, 1], offset: ?>>
// CHECK:       [[MEMREF_1:%.*]] = memref.alloc() : memref<8x32xi16>
// CHECK:       memref.copy [[MEMREF_0:%.*]], [[MEMREF_1:%.*]] : memref<8x32xi16, strided<[128, 1], offset: ?>> to memref<8x32xi16>
// CHECK:       [[VAL_1:%.*]] = bufferization.to_tensor [[MEMREF_1:%.*]] restrict writable : memref<8x32xi16>
// CHECK:       [[MEMREF_3:%.*]] = memref.reinterpret_cast [[VAL_2:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi16> to memref<8x32xi16, strided<[128, 1], offset: ?>>
// CHECK:       bufferization.materialize_in_destination [[VAL_1:%.*]] in writable [[MEMREF_3:%.*]] : (tensor<8x32xi16>, memref<8x32xi16, strided<[128, 1], offset: ?>>) -> ()

// -----

// dtype: uint32
module {
  tt.func public @triton_tensor_descriptor_function_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i32>, <tensor<8x32xui32>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i32>, <tensor<8x32xui32>>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0[%3, %5] : !tt.tensordesc<tensor<8x32xui32>> -> tensor<8x32xi32>
    tt.descriptor_store %1[%3, %5], %6 : !tt.tensordesc<tensor<8x32xui32>>, tensor<8x32xi32>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_tensor_descriptor_function_2d
// CHECK:       [[MEMREF_0:%.*]] = memref.reinterpret_cast [[VAL_0:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi32> to memref<8x32xi32, strided<[128, 1], offset: ?>>
// CHECK:       [[MEMREF_1:%.*]] = memref.alloc() : memref<8x32xi32>
// CHECK:       memref.copy [[MEMREF_0:%.*]], [[MEMREF_1:%.*]] : memref<8x32xi32, strided<[128, 1], offset: ?>> to memref<8x32xi32>
// CHECK:       [[VAL_1:%.*]] = bufferization.to_tensor [[MEMREF_1:%.*]] restrict writable : memref<8x32xi32>
// CHECK:       [[MEMREF_3:%.*]] = memref.reinterpret_cast [[VAL_2:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi32> to memref<8x32xi32, strided<[128, 1], offset: ?>>
// CHECK:       bufferization.materialize_in_destination [[VAL_1:%.*]] in writable [[MEMREF_3:%.*]] : (tensor<8x32xi32>, memref<8x32xi32, strided<[128, 1], offset: ?>>) -> ()

// -----

// dtype: uint64
module {
  tt.func public @triton_tensor_descriptor_function_2d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i64>, <tensor<8x32xui64>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <i64>, <tensor<8x32xui64>>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0[%3, %5] : !tt.tensordesc<tensor<8x32xui64>> -> tensor<8x32xi64>
    tt.descriptor_store %1[%3, %5], %6 : !tt.tensordesc<tensor<8x32xui64>>, tensor<8x32xi64>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_tensor_descriptor_function_2d
// CHECK:       [[MEMREF_0:%.*]] = memref.reinterpret_cast [[VAL_0:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi64> to memref<8x32xi64, strided<[128, 1], offset: ?>>
// CHECK:       [[MEMREF_1:%.*]] = memref.alloc() : memref<8x32xi64>
// CHECK:       memref.copy [[MEMREF_0:%.*]], [[MEMREF_1:%.*]] : memref<8x32xi64, strided<[128, 1], offset: ?>> to memref<8x32xi64>
// CHECK:       [[VAL_1:%.*]] = bufferization.to_tensor [[MEMREF_1:%.*]] restrict writable : memref<8x32xi64>
// CHECK:       [[MEMREF_3:%.*]] = memref.reinterpret_cast [[VAL_2:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xi64> to memref<8x32xi64, strided<[128, 1], offset: ?>>
// CHECK:       bufferization.materialize_in_destination [[VAL_1:%.*]] in writable [[MEMREF_3:%.*]] : (tensor<8x32xi64>, memref<8x32xi64, strided<[128, 1], offset: ?>>) -> ()

// -----

// dtype: fp8E4M3FN
module {
  tt.func public @triton_tensor_descriptor_function_2d(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <f8E4M3FN>, <tensor<8x32xf8E4M3FN>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <f8E4M3FN>, <tensor<8x32xf8E4M3FN>>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0[%3, %5] : !tt.tensordesc<tensor<8x32xf8E4M3FN>> -> tensor<8x32xf8E4M3FN>
    tt.descriptor_store %1[%3, %5], %6 : !tt.tensordesc<tensor<8x32xf8E4M3FN>>, tensor<8x32xf8E4M3FN>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_tensor_descriptor_function_2d
// CHECK:       [[MEMREF_0:%.*]] = memref.reinterpret_cast [[VAL_0:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xf8E4M3FN> to memref<8x32xf8E4M3FN, strided<[128, 1], offset: ?>>
// CHECK:       [[MEMREF_1:%.*]] = memref.alloc() : memref<8x32xf8E4M3FN>
// CHECK:       memref.copy [[MEMREF_0:%.*]], [[MEMREF_1:%.*]] : memref<8x32xf8E4M3FN, strided<[128, 1], offset: ?>> to memref<8x32xf8E4M3FN>
// CHECK:       [[VAL_1:%.*]] = bufferization.to_tensor [[MEMREF_1:%.*]] restrict writable : memref<8x32xf8E4M3FN>
// CHECK:       [[MEMREF_3:%.*]] = memref.reinterpret_cast [[VAL_2:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xf8E4M3FN> to memref<8x32xf8E4M3FN, strided<[128, 1], offset: ?>>
// CHECK:       bufferization.materialize_in_destination [[VAL_1:%.*]] in writable [[MEMREF_3:%.*]] : (tensor<8x32xf8E4M3FN>, memref<8x32xf8E4M3FN, strided<[128, 1], offset: ?>>) -> ()

// -----

// dtype: fp8E5M2
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tensor_descriptor_function_2d(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <f8E5M2>, <tensor<8x32xf8E5M2>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <f8E5M2>, <tensor<8x32xf8E5M2>>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0[%3, %5] : !tt.tensordesc<tensor<8x32xf8E5M2>> -> tensor<8x32xf8E5M2>
    tt.descriptor_store %1[%3, %5], %6 : !tt.tensordesc<tensor<8x32xf8E5M2>>, tensor<8x32xf8E5M2>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_tensor_descriptor_function_2d
// CHECK:       [[MEMREF_0:%.*]] = memref.reinterpret_cast [[VAL_0:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xf8E5M2> to memref<8x32xf8E5M2, strided<[128, 1], offset: ?>>
// CHECK:       [[MEMREF_1:%.*]] = memref.alloc() : memref<8x32xf8E5M2>
// CHECK:       memref.copy [[MEMREF_0:%.*]], [[MEMREF_1:%.*]] : memref<8x32xf8E5M2, strided<[128, 1], offset: ?>> to memref<8x32xf8E5M2>
// CHECK:       [[VAL_1:%.*]] = bufferization.to_tensor [[MEMREF_1:%.*]] restrict writable : memref<8x32xf8E5M2>
// CHECK:       [[MEMREF_3:%.*]] = memref.reinterpret_cast [[VAL_2:%.*]] to offset: [[[VAL_1:%.*]]], sizes: [8, 32], strides: [128, 1] : memref<?xf8E5M2> to memref<8x32xf8E5M2, strided<[128, 1], offset: ?>>
// CHECK:       bufferization.materialize_in_destination [[VAL_1:%.*]] in writable [[MEMREF_3:%.*]] : (tensor<8x32xf8E5M2>, memref<8x32xf8E5M2, strided<[128, 1], offset: ?>>) -> ()
