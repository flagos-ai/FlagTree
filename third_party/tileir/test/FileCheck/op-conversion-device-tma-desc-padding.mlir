// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts)" | FileCheck %s

// Verify that ConvertMakeTensorDescOp forwards the triton padding_option to
// the cuda_tile StridedView type's padding_value (PAD_ZERO -> zero,
// PAD_NAN -> nan). The descriptor load consumes the strided view with the
// matching padding so the unrealized_conversion_cast inserted by the
// type-converter default (zero) is reconciled.

// PAD_ZERO (= 1) is the default for tt.make_tensor_descriptor.
module {
  tt.func public @device_tma_padding_zero(
      %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %N : i32 to i64
    %1 = tt.make_tensor_descriptor %in_ptr, [%M, %N], [%0, %c1_i64] {padding = 1 : i32} : <f32>, <tensor<32x32xf32>>
    %2 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf32>> -> tensor<32x32xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @device_tma_padding_zero
// CHECK: make_strided_view {{.*}} padding_value = zero
// CHECK: load_view_tko {{.*}} strided_view<tile=(32x32), traversal_strides=[1,1], padding_value = zero

// -----

// PAD_NAN (= 2) must reach the cuda_tile view as padding_value = nan.
module {
  tt.func public @device_tma_padding_nan(
      %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %N : i32 to i64
    %1 = tt.make_tensor_descriptor %in_ptr, [%M, %N], [%0, %c1_i64] {padding = 2 : i32} : <f32>, <tensor<32x32xf32>>
    %2 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf32>> -> tensor<32x32xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @device_tma_padding_nan
// CHECK: make_strided_view {{.*}} padding_value = nan
// CHECK: load_view_tko {{.*}} strided_view<tile=(32x32), traversal_strides=[1,1], padding_value = nan
