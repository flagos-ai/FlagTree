// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=False compile-mode=simd' '--triton-to-unstructure=compile-on-910-95=False compile-mode=simd' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured '--triton-to-linalg=compile-mode=simd' --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @triton_conv2d_3d_kernel(%input_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %weight_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_in_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %c_in_offsets_0 = tt.expand_dims %c_in_offsets {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c_in_offsets_1 = tt.expand_dims %c_in_offsets_0 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %h_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %h_offsets_2 = tt.expand_dims %h_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %h_offsets_3 = tt.expand_dims %h_offsets_2 {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %w_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %w_offsets_4 = tt.expand_dims %w_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %w_offsets_5 = tt.expand_dims %w_offsets_4 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %input_offs = arith.constant dense<256> : tensor<8x1x1xi32>
    %input_offs_6 = arith.constant dense<16> : tensor<1x16x1xi32>
    %input_offs_h = arith.muli %c_in_offsets_1, %input_offs : tensor<8x1x1xi32>
    %input_offs_hw = arith.muli %h_offsets_3, %input_offs_6 : tensor<1x16x1xi32>
    %input_offs_t0 = tt.broadcast %input_offs_h : tensor<8x1x1xi32> -> tensor<8x16x1xi32>
    %input_offs_t1 = tt.broadcast %input_offs_hw : tensor<1x16x1xi32> -> tensor<8x16x1xi32>
    %input_offs_t2 = arith.addi %input_offs_t0, %input_offs_t1 : tensor<8x16x1xi32>
    %input_offs_t3 = tt.broadcast %input_offs_t2 : tensor<8x16x1xi32> -> tensor<8x16x16xi32>
    %input_offs_t4 = tt.broadcast %w_offsets_5 : tensor<1x1x16xi32> -> tensor<8x16x16xi32>
    %input_offs_t5 = arith.addi %input_offs_t3, %input_offs_t4 : tensor<8x16x16xi32>
    %input_tensor = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<8x16x16x!tt.ptr<f16>>
    %input_tensor_7 = tt.addptr %input_tensor, %input_offs_t5 : tensor<8x16x16x!tt.ptr<f16>>, tensor<8x16x16xi32>
    %input_tile = tt.load %input_tensor_7 : tensor<8x16x16x!tt.ptr<f16>>
    %co_offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %co_offs_0 = tt.expand_dims %co_offs {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %co_offs_1 = tt.expand_dims %co_offs_0 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %co_offs_2 = tt.expand_dims %co_offs_1 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %ci_offs = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %ci_offs_0 = tt.expand_dims %ci_offs {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %ci_offs_1 = tt.expand_dims %ci_offs_0 {axis = 3 : i32} : tensor<1x8x1xi32> -> tensor<1x8x1x1xi32>
    %kh_offs = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kh_offs_0 = tt.expand_dims %kh_offs {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kh_offs_1 = tt.expand_dims %kh_offs_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %kh_offs_2 = tt.expand_dims %kh_offs_1 {axis = 3 : i32} : tensor<1x1x3xi32> -> tensor<1x1x3x1xi32>
    %kw_offs = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kw_offs_0 = tt.expand_dims %kw_offs {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kw_offs_1 = tt.expand_dims %kw_offs_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %kw_offs_2 = tt.expand_dims %kw_offs_1 {axis = 2 : i32} : tensor<1x1x3xi32> -> tensor<1x1x1x3xi32>
    %weight_offs_c = arith.constant dense<72> : tensor<4x1x1x1xi32>
    %weight_offs_ci = arith.constant dense<9> : tensor<1x8x1x1xi32>
    %weight_offs_kh = arith.constant dense<3> : tensor<1x1x3x1xi32>
    %weight_t0 = arith.muli %co_offs_2, %weight_offs_c : tensor<4x1x1x1xi32>
    %weight_t1 = arith.muli %ci_offs_1, %weight_offs_ci : tensor<1x8x1x1xi32>
    %weight_t2 = arith.muli %kh_offs_2, %weight_offs_kh : tensor<1x1x3x1xi32>
    %weight_t3 = tt.broadcast %weight_t0 : tensor<4x1x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t4 = tt.broadcast %weight_t1 : tensor<1x8x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t5 = arith.addi %weight_t3, %weight_t4 : tensor<4x8x1x1xi32>
    %weight_t6 = tt.broadcast %weight_t5 : tensor<4x8x1x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t7 = tt.broadcast %weight_t2 : tensor<1x1x3x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t8 = arith.addi %weight_t6, %weight_t7 : tensor<4x8x3x1xi32>
    %weight_t9 = tt.broadcast %weight_t8 : tensor<4x8x3x1xi32> -> tensor<4x8x3x3xi32>
    %weight_t10 = tt.broadcast %kw_offs_2 : tensor<1x1x1x3xi32> -> tensor<4x8x3x3xi32>
    %weight_offs = arith.addi %weight_t9, %weight_t10 : tensor<4x8x3x3xi32>
    %weight_tensor = tt.splat %weight_ptr : !tt.ptr<f16> -> tensor<4x8x3x3x!tt.ptr<f16>>
    %weight_tensor_ = tt.addptr %weight_tensor, %weight_offs : tensor<4x8x3x3x!tt.ptr<f16>>, tensor<4x8x3x3xi32>
    %weight_tile = tt.load %weight_tensor_ : tensor<4x8x3x3x!tt.ptr<f16>>
    %output = ascend.conv2d(%input_tile, %weight_tile) {dilation = array<i32: 1, 1>, groups = 1 : i64, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>} : (tensor<8x16x16xf16>, tensor<4x8x3x3xf16>) -> tensor<4x14x14xf16>
    %lo_h = tt.make_range {end = 14 : i32, start = 0 : i32} : tensor<14xi32>
    %lo_h_0 = tt.expand_dims %lo_h {axis = 0 : i32} : tensor<14xi32> -> tensor<1x14xi32>
    %lo_h_1 = tt.expand_dims %lo_h_0 {axis = 2 : i32} : tensor<1x14xi32> -> tensor<1x14x1xi32>
    %lo_w = tt.make_range {end = 14 : i32, start = 0 : i32} : tensor<14xi32>
    %lo_w_0 = tt.expand_dims %lo_w {axis = 0 : i32} : tensor<14xi32> -> tensor<1x14xi32>
    %lo_w_1 = tt.expand_dims %lo_w_0 {axis = 1 : i32} : tensor<1x14xi32> -> tensor<1x1x14xi32>
    %out_offs_c = arith.constant dense<196> : tensor<4x1x1xi32>
    %out_offs_h = arith.constant dense<14> : tensor<1x14x1xi32>
    %co_offs_3 = tt.expand_dims %co_offs {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %co_offs_4 = tt.expand_dims %co_offs_3 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %out_t0 = arith.muli %co_offs_4, %out_offs_c : tensor<4x1x1xi32>
    %out_t1 = arith.muli %lo_h_1, %out_offs_h : tensor<1x14x1xi32>
    %out_t2 = tt.broadcast %out_t0 : tensor<4x1x1xi32> -> tensor<4x14x1xi32>
    %out_t3 = tt.broadcast %out_t1 : tensor<1x14x1xi32> -> tensor<4x14x1xi32>
    %out_t4 = arith.addi %out_t2, %out_t3 : tensor<4x14x1xi32>
    %out_t5 = tt.broadcast %out_t4 : tensor<4x14x1xi32> -> tensor<4x14x14xi32>
    %out_t6 = tt.broadcast %lo_w_1 : tensor<1x1x14xi32> -> tensor<4x14x14xi32>
    %out_offs = arith.addi %out_t5, %out_t6 : tensor<4x14x14xi32>
    %0 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<4x14x14x!tt.ptr<f16>>
    %1 = tt.addptr %0, %out_offs : tensor<4x14x14x!tt.ptr<f16>>, tensor<4x14x14xi32>
    tt.store %1, %output : tensor<4x14x14x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_conv2d_3d_kernel(
// CHECK: %[[VAL_0:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [8, 16, 16], strides: [256, 16, 1]
// CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<8x16x16xf16>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [4, 8, 3, 3], strides: [72, 9, 3, 1]
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<4x8x3x3xf16>
// CHECK: memref.copy %[[VAL_2]], %[[VAL_3]]
// CHECK: hfusion.conv2d
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: groups = 1 : i32
// CHECK-SAME: padding = array<i64: 0, 0>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK: %[[VAL_6:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [4, 14, 14], strides: [196, 14, 1]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[VAL_6]]
