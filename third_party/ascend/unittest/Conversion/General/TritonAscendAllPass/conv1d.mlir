// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @triton_conv1d_2d_kernel(%input_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %weight_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %output_offsets = arith.constant dense<126> : tensor<16x1xi32>
    %weight_offsets = arith.constant dense<3> : tensor<1x16x1xi32>
    %weight_offsets_0 = arith.constant dense<48> : tensor<16x1x1xi32>
    %input_offsets = arith.constant dense<128> : tensor<16x1xi32>
    %c_in_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c_in_offsets_1 = tt.expand_dims %c_in_offsets {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %l_in_offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %l_in_offsets_2 = tt.expand_dims %l_in_offsets {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %input_offsets_3 = arith.muli %c_in_offsets_1, %input_offsets : tensor<16x1xi32>
    %input_offsets_4 = tt.broadcast %input_offsets_3 : tensor<16x1xi32> -> tensor<16x128xi32>
    %input_offsets_5 = tt.broadcast %l_in_offsets_2 : tensor<1x128xi32> -> tensor<16x128xi32>
    %input_offsets_6 = arith.addi %input_offsets_4, %input_offsets_5 : tensor<16x128xi32>
    %input_tensor = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>>
    %input_tensor_7 = tt.addptr %input_tensor, %input_offsets_6 : tensor<16x128x!tt.ptr<f16>>, tensor<16x128xi32>
    %input_tensor_8 = tt.load %input_tensor_7 : tensor<16x128x!tt.ptr<f16>>
    %c_out_offsets = tt.expand_dims %c_in_offsets_1 {axis = 2 : i32} : tensor<16x1xi32> -> tensor<16x1x1xi32>
    %c_in_group_offsets = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %c_in_group_offsets_9 = tt.expand_dims %c_in_group_offsets {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %k_offsets = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %k_offsets_10 = tt.expand_dims %k_offsets {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %k_offsets_11 = tt.expand_dims %k_offsets_10 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %weight_offsets_12 = arith.muli %c_out_offsets, %weight_offsets_0 : tensor<16x1x1xi32>
    %weight_offsets_13 = arith.muli %c_in_group_offsets_9, %weight_offsets : tensor<1x16x1xi32>
    %weight_offsets_14 = tt.broadcast %weight_offsets_12 : tensor<16x1x1xi32> -> tensor<16x16x1xi32>
    %weight_offsets_15 = tt.broadcast %weight_offsets_13 : tensor<1x16x1xi32> -> tensor<16x16x1xi32>
    %weight_offsets_16 = arith.addi %weight_offsets_14, %weight_offsets_15 : tensor<16x16x1xi32>
    %weight_offsets_17 = tt.broadcast %weight_offsets_16 : tensor<16x16x1xi32> -> tensor<16x16x3xi32>
    %weight_offsets_18 = tt.broadcast %k_offsets_11 : tensor<1x1x3xi32> -> tensor<16x16x3xi32>
    %weight_offsets_19 = arith.addi %weight_offsets_17, %weight_offsets_18 : tensor<16x16x3xi32>
    %weight_tensor = tt.splat %weight_ptr : !tt.ptr<f16> -> tensor<16x16x3x!tt.ptr<f16>>
    %weight_tensor_20 = tt.addptr %weight_tensor, %weight_offsets_19 : tensor<16x16x3x!tt.ptr<f16>>, tensor<16x16x3xi32>
    %weight_tensor_21 = tt.load %weight_tensor_20 : tensor<16x16x3x!tt.ptr<f16>>
    %bias_tensor = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %bias_tensor_22 = tt.addptr %bias_tensor, %c_in_offsets : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    %bias_tensor_23 = tt.load %bias_tensor_22 : tensor<16x!tt.ptr<f16>>
    %output = ascend.conv1d(%input_tensor_8, %weight_tensor_21, %bias_tensor_23) {dilation = 1 : i64, groups = 1 : i64, padding_size = 0 : i64, stride = 1 : i64} : (tensor<16x128xf16>, tensor<16x16x3xf16>, tensor<16xf16>) -> tensor<16x126xf16>
    %l_out_off = tt.make_range {end = 126 : i32, start = 0 : i32} : tensor<126xi32>
    %l_out_off_24 = tt.expand_dims %l_out_off {axis = 0 : i32} : tensor<126xi32> -> tensor<1x126xi32>
    %output_offsets_25 = arith.muli %c_in_offsets_1, %output_offsets : tensor<16x1xi32>
    %output_offsets_26 = tt.broadcast %output_offsets_25 : tensor<16x1xi32> -> tensor<16x126xi32>
    %output_offsets_27 = tt.broadcast %l_out_off_24 : tensor<1x126xi32> -> tensor<16x126xi32>
    %output_offsets_28 = arith.addi %output_offsets_26, %output_offsets_27 : tensor<16x126xi32>
    %0 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<16x126x!tt.ptr<f16>>
    %1 = tt.addptr %0, %output_offsets_28 : tensor<16x126x!tt.ptr<f16>>, tensor<16x126xi32>
    tt.store %1, %output : tensor<16x126x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_conv1d_2d_kernel(
// CHECK: %[[VAL_0:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [16, 128], strides: [128, 1]
// CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<16x128xf16>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [16, 16, 3], strides: [48, 3, 1]
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<16x16x3xf16>
// CHECK: memref.copy %[[VAL_2]], %[[VAL_3]]
// CHECK: %[[VAL_4:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [16], strides: [1]
// CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<16xf16>
// CHECK: memref.copy %[[VAL_4]], %[[VAL_5]]
// CHECK: %[[VAL_6:.*]] = hfusion.conv1d
// CHECK-SAME: {dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}
// CHECK: %[[VAL_7:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [16, 126], strides: [126, 1]
// CHECK: bufferization.materialize_in_destination %[[VAL_6]] in writable %[[VAL_7]]


// -----


module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @triton_conv1d_3d_kernel(%input_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %weight_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %output_offsets = arith.constant dense<128> : tensor<1x30x1xi32>
    %output_offsets_0 = arith.constant dense<3840> : tensor<2x1x1xi32>
    %weight_offsets = arith.constant dense<3> : tensor<1x32x1xi32>
    %weight_offsets_1 = arith.constant dense<96> : tensor<30x1x1xi32>
    %input_offsets = arith.constant dense<128> : tensor<1x32x1xi32>
    %input_offsets_2 = arith.constant dense<4096> : tensor<2x1x1xi32>
    %n_offsets = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %n_offsets_3 = tt.expand_dims %n_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %n_offsets_4 = tt.expand_dims %n_offsets_3 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %c_in_offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c_in_offsets_5 = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %c_in_offsets_6 = tt.expand_dims %c_in_offsets_5 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %l_in_offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %l_in_offsets_7 = tt.expand_dims %l_in_offsets {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %l_in_offsets_8 = tt.expand_dims %l_in_offsets_7 {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %input_offsets_9 = arith.muli %n_offsets_4, %input_offsets_2 : tensor<2x1x1xi32>
    %input_offsets_10 = arith.muli %c_in_offsets_6, %input_offsets : tensor<1x32x1xi32>
    %input_offsets_11 = tt.broadcast %input_offsets_9 : tensor<2x1x1xi32> -> tensor<2x32x1xi32>
    %input_offsets_12 = tt.broadcast %input_offsets_10 : tensor<1x32x1xi32> -> tensor<2x32x1xi32>
    %input_offsets_13 = arith.addi %input_offsets_11, %input_offsets_12 : tensor<2x32x1xi32>
    %input_offsets_14 = tt.broadcast %input_offsets_13 : tensor<2x32x1xi32> -> tensor<2x32x128xi32>
    %input_offsets_15 = tt.broadcast %l_in_offsets_8 : tensor<1x1x128xi32> -> tensor<2x32x128xi32>
    %input_offsets_16 = arith.addi %input_offsets_14, %input_offsets_15 : tensor<2x32x128xi32>
    %input_tensor = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<2x32x128x!tt.ptr<f16>>
    %input_tensor_17 = tt.addptr %input_tensor, %input_offsets_16 : tensor<2x32x128x!tt.ptr<f16>>, tensor<2x32x128xi32>
    %input_tensor_18 = tt.load %input_tensor_17 : tensor<2x32x128x!tt.ptr<f16>>
    %c_out_offsets = tt.make_range {end = 30 : i32, start = 0 : i32} : tensor<30xi32>
    %c_out_offsets_19 = tt.expand_dims %c_out_offsets {axis = 1 : i32} : tensor<30xi32> -> tensor<30x1xi32>
    %c_out_offsets_20 = tt.expand_dims %c_out_offsets_19 {axis = 2 : i32} : tensor<30x1xi32> -> tensor<30x1x1xi32>
    %k_offsets = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %k_offsets_21 = tt.expand_dims %k_offsets {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %k_offsets_22 = tt.expand_dims %k_offsets_21 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %weight_offsets_23 = arith.muli %c_out_offsets_20, %weight_offsets_1 : tensor<30x1x1xi32>
    %weight_offsets_24 = arith.muli %c_in_offsets_6, %weight_offsets : tensor<1x32x1xi32>
    %weight_offsets_25 = tt.broadcast %weight_offsets_23 : tensor<30x1x1xi32> -> tensor<30x32x1xi32>
    %weight_offsets_26 = tt.broadcast %weight_offsets_24 : tensor<1x32x1xi32> -> tensor<30x32x1xi32>
    %weight_offsets_27 = arith.addi %weight_offsets_25, %weight_offsets_26 : tensor<30x32x1xi32>
    %weight_offsets_28 = tt.broadcast %weight_offsets_27 : tensor<30x32x1xi32> -> tensor<30x32x3xi32>
    %weight_offsets_29 = tt.broadcast %k_offsets_22 : tensor<1x1x3xi32> -> tensor<30x32x3xi32>
    %weight_offsets_30 = arith.addi %weight_offsets_28, %weight_offsets_29 : tensor<30x32x3xi32>
    %weight_tensor = tt.splat %weight_ptr : !tt.ptr<f16> -> tensor<30x32x3x!tt.ptr<f16>>
    %weight_tensor_31 = tt.addptr %weight_tensor, %weight_offsets_30 : tensor<30x32x3x!tt.ptr<f16>>, tensor<30x32x3xi32>
    %weight_tensor_32 = tt.load %weight_tensor_31 : tensor<30x32x3x!tt.ptr<f16>>
    %bias_tensor = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<30x!tt.ptr<f16>>
    %bias_tensor_33 = tt.addptr %bias_tensor, %c_out_offsets : tensor<30x!tt.ptr<f16>>, tensor<30xi32>
    %bias_tensor_34 = tt.load %bias_tensor_33 : tensor<30x!tt.ptr<f16>>
    %output = ascend.conv1d(%input_tensor_18, %weight_tensor_32, %bias_tensor_34) {dilation = 1 : i64, groups = 1 : i64, padding_size = 1 : i64, stride = 1 : i64} : (tensor<2x32x128xf16>, tensor<30x32x3xf16>, tensor<30xf16>) -> tensor<2x30x128xf16>
    %c_out_offsets_35 = tt.expand_dims %c_out_offsets {axis = 0 : i32} : tensor<30xi32> -> tensor<1x30xi32>
    %c_out_offsets_36 = tt.expand_dims %c_out_offsets_35 {axis = 2 : i32} : tensor<1x30xi32> -> tensor<1x30x1xi32>
    %output_offsets_37 = arith.muli %n_offsets_4, %output_offsets_0 : tensor<2x1x1xi32>
    %output_offsets_38 = arith.muli %c_out_offsets_36, %output_offsets : tensor<1x30x1xi32>
    %output_offsets_39 = tt.broadcast %output_offsets_37 : tensor<2x1x1xi32> -> tensor<2x30x1xi32>
    %output_offsets_40 = tt.broadcast %output_offsets_38 : tensor<1x30x1xi32> -> tensor<2x30x1xi32>
    %output_offsets_41 = arith.addi %output_offsets_39, %output_offsets_40 : tensor<2x30x1xi32>
    %output_offsets_42 = tt.broadcast %output_offsets_41 : tensor<2x30x1xi32> -> tensor<2x30x128xi32>
    %output_offsets_43 = tt.broadcast %l_in_offsets_8 : tensor<1x1x128xi32> -> tensor<2x30x128xi32>
    %output_offsets_44 = arith.addi %output_offsets_42, %output_offsets_43 : tensor<2x30x128xi32>
    %0 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<2x30x128x!tt.ptr<f16>>
    %1 = tt.addptr %0, %output_offsets_44 : tensor<2x30x128x!tt.ptr<f16>>, tensor<2x30x128xi32>
    tt.store %1, %output : tensor<2x30x128x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_conv1d_3d_kernel(
// CHECK: %[[VAL_0:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [2, 32, 128], strides: [4096, 128, 1]
// CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<2x32x128xf16>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [30, 32, 3], strides: [96, 3, 1]
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<30x32x3xf16>
// CHECK: memref.copy %[[VAL_2]], %[[VAL_3]]
// CHECK: %[[VAL_4:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [30], strides: [1]
// CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<30xf16>
// CHECK: memref.copy %[[VAL_4]], %[[VAL_5]]
// CHECK: %[[VAL_6:.*]] = hfusion.conv1d
// CHECK-SAME: {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32}
// CHECK: %[[VAL_7:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [2, 30, 128], strides: [3840, 128, 1]
// CHECK: bufferization.materialize_in_destination %[[VAL_6]] in writable %[[VAL_7]]
