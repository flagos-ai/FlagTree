// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg --split-input-file %s 2>&1 | FileCheck %s --check-prefix=WARN

// Unit tests for ConverterUtils::getBoundarySizes() (PR #1838,
// "[TritonToLinalg](fix) boundary size Error for make_block_ptr").
//
// `tl.make_block_ptr` is lowered to `tt.make_tensor_ptr` before
// TritonToLinalg runs.  getBoundarySizes() reconstructs the per-axis
// in-bounds size by decomposing the flat block offset with the full-shape
// strides; three defects in that decomposition are covered here:
//
//  1. (primary) An axis that is *not* boundary-checked never reduced the
//     flat offset (offset % stride), so a checked trailing axis saw the
//     offset contributed by the leading axis, shrinking its boundary to 0
//     and silently dropping the loaded/stored data.  The fix strips every
//     non-zero-stride axis, checked or not.
//
//  2. (incidental) A zero-stride (broadcast) axis listed in boundary_check
//     reached divOpFoldResult(offsetShift, 0), which emits "cannot div 0!"
//     and returns an empty OpFoldResult, crashing the compiler.  The fix
//     skips such axes (emitting a warning) and keeps the current block size.
//
//  3. Logical axis order need not be physical major-to-minor order.  For a
//     permuted layout with strides [1, 800, 20], decomposing axes as [0, 1, 2]
//     treats the entire linear offset as axis 0.  The fix visits axes in
//     descending stride order [1, 2, 0].

// The skipped zero-stride axis of the second case is reported through a
// warning instead of the previous "cannot div 0!" error (the diagnostic
// precedes the module on stderr).
// WARN-NOT: error:
// WARN: warning: getBoundarySizes() cannot reconstruct boundary on checked zero-stride axis 0
// WARN-LABEL: func.func @boundary_size_zero_stride_axis

// Primary case: axis 0 (stride 64) is *not* checked, axis 1 is.  The block
// starts at (16, 36) of a (64, 60) tensor: the flat offset 16*64 + 36 = 1060
// must be reduced by axis 0 first, leaving 36 for axis 1 -> boundary
// 60 - 36 = 24.  Before the fix the un-reduced offset 1060 was used,
// clamping the axis-1 boundary to 0 ([16, 0]) and dropping the whole load.

// CHECK-LABEL: func.func @boundary_size_unchecked_axis
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_unchecked_axis(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c36_i32 = arith.constant 36 : i32
    %c1_i64 = arith.constant 1 : i64
    %c60_i64 = arith.constant 60 : i64
    %c64_i64 = arith.constant 64 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    tt.return
  }
}

// -----
// Incidental case: axis 0 is a broadcast axis (stride 0) and is
// boundary-checked.  The zero-stride axis keeps its full block size (16)
// while the axis-1 boundary is reconstructed as 60 - 36 = 24.

// CHECK-LABEL: func.func @boundary_size_zero_stride_axis
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
// CHECK: memref.copy
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_zero_stride_axis(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c36_i32 = arith.constant 36 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c60_i64 = arith.constant 60 : i64
    %c64_i64 = arith.constant 64 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c64_i64, %c60_i64], [%c0_i64, %c1_i64], [%c0_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c0_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    tt.return
  }
}

// -----
// Permuted layout: logical axes [0, 1, 2] have physical strides [1, 800, 20].
// The block starts at logical offset [7, 18, 32] in shape [20, 30, 40].  Its
// flat offset is 7 + 18*800 + 32*20 = 15047.  Decomposing in physical order
// [1, 2, 0] recovers [18, 32, 7], which maps back to logical boundary sizes
// [1, min(16, 30-18), min(64, 40-32)] = [1, 12, 8].

// CHECK-LABEL: func.func @boundary_size_permuted_strides
// CHECK: memref.subview {{.*}}[0, 0, 0] [1, 12, 8] [1, 1, 1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_permuted_strides(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c7_i32 = arith.constant 7 : i32
    %c18_i32 = arith.constant 18 : i32
    %c32_i32 = arith.constant 32 : i32
    %c20_i64 = arith.constant 20 : i64
    %c30_i64 = arith.constant 30 : i64
    %c40_i64 = arith.constant 40 : i64
    %c800_i64 = arith.constant 800 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c20_i64, %c30_i64, %c40_i64], [%c1_i64, %c800_i64, %c20_i64], [%c7_i32, %c18_i32, %c32_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x16x64xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 0, 1, 2>} : !tt.ptr<tensor<1x16x64xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c20_i64, %c30_i64, %c40_i64], [%c1200_i64, %c40_i64, %c1_i64], [%c7_i32, %c18_i32, %c32_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x16x64xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 0, 1, 2>} : !tt.ptr<tensor<1x16x64xf32>>
    tt.return
  }
}
