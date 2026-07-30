// RUN: triton-shared-opt --structured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @triton_tem_fused_mm_0(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c8_i32 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %3 = arith.subi %c1_i32, %2 : i32
    %4 = arith.minsi %3, %c8_i32 : i32
    %5 = arith.remsi %0, %4 : i32
    %6 = arith.addi %2, %5 : i32
    %7 = arith.remsi %0, %c8_i32 : i32
    %8 = arith.divsi %7, %4 : i32
    %9 = arith.muli %6, %c128_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.index_cast %9 : i32 to index
    %12 = arith.muli %8, %c128_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.index_cast %12 : i32 to index
    %15 = arith.muli %11, %c8 : index
    %16 = tts.make_tptr %arg0 to sizes: [128, 32], strides: [%c8, 1], offsets: [%15, 0], shape: [16, 0], order: [] : <f32> to tensor<128x32x!tt.ptr<f32>>
    %17 = "tts.load"(%16, %cst) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 128, 8>}> : (tensor<128x32x!tt.ptr<f32>>, f32) -> tensor<128x32xf32>
    %18 = arith.muli %14, %c8 : index
    %19 = tts.make_tptr %arg1 to sizes: [32, 128], strides: [1, %c8], offsets: [0, %18], shape: [0, 128], order: [] : <f32> to tensor<32x128x!tt.ptr<f32>>
    %20 = "tts.load"(%19, %cst) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 8, 128>}> : (tensor<32x128x!tt.ptr<f32>>, f32) -> tensor<32x128xf32>
    %21 = tt.dot %17, %20, %cst_0 : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>
    %22 = arith.muli %10, %c16 : index
    %23 = tts.make_tptr %arg2 to sizes: [128, 128], strides: [%c16, 1], offsets: [%22, %13], shape: [0, 0], order: [] : <f32> to tensor<128x128x!tt.ptr<f32>>
    %24 = arith.index_cast %9 : i32 to index
    %25 = arith.addi %24, %c128 : index
    %26 = arith.minsi %25, %c2 : index
    %27 = arith.maxsi %26, %24 : index
    %28 = arith.subi %27, %24 : index
    %29 = arith.index_cast %12 : i32 to index
    %30 = arith.addi %29, %c128 : index
    %31 = arith.minsi %30, %c16 : index
    %32 = arith.maxsi %31, %29 : index
    %33 = arith.subi %32, %29 : index
    %34 = arith.minsi %28, %c128 : index
    %35 = arith.minsi %33, %c128 : index
    "tts.store"(%23, %21, %34, %35) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x128x!tt.ptr<f32>>, tensor<128x128xf32>, index, index) -> ()
    tt.return
  }
}

// CHECK-DAG: [[arg1:%.+]] = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG: [[arg0:%.+]] = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>

// CHECK: [[arg0_chunk1:%.+]] = memref.reinterpret_cast [[arg0]]
// CHECK: [[arg0_chunk2:%.+]] = memref.reinterpret_cast [[arg0]]
// CHECK: memref.subview [[arg0_chunk1]]
// CHECK: memref.subview [[arg0_chunk2]]

// CHECK: [[arg1_chunk1:%.+]] = memref.reinterpret_cast [[arg1]]
// CHECK: [[arg1_chunk2:%.+]] = memref.reinterpret_cast [[arg1]]
// CHECK: memref.subview [[arg1_chunk1]]
// CHECK: memref.subview [[arg1_chunk2]]
