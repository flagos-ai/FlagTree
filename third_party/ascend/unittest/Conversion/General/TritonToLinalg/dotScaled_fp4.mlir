// RUN: triton-opt --pass-pipeline="builtin.module(triton-to-unstructure{compile-on-910-95=true compile-mode=simd},triton-to-linalg{compile-on-910-95=true enable-nd2nz-on-vector=false global-kernel=false named-ops=true compile-mode=simd})" --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_dot_scaled_test_fp4(%a_ptr: !tt.ptr<i8>, %a_scale_ptr: !tt.ptr<i8>, %b_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>, %c_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %b = arith.constant dense<16> : tensor<32x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<16x1xi32>
    %a = arith.constant dense<32> : tensor<16x1xi32>
    %m_offs = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %k_packed_offs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %scale_offs = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %a_1 = tt.expand_dims %m_offs {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_2 = arith.muli %a_1, %a : tensor<16x1xi32>
    %a_3 = tt.splat %a_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_4 = tt.addptr %a_3, %a_2 : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_5 = tt.expand_dims %k_packed_offs {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %a_6 = tt.broadcast %a_4 : tensor<16x1x!tt.ptr<i8>> -> tensor<16x32x!tt.ptr<i8>>
    %a_7 = tt.broadcast %a_5 : tensor<1x32xi32> -> tensor<16x32xi32>
    %a_8 = tt.addptr %a_6, %a_7 : tensor<16x32x!tt.ptr<i8>>, tensor<16x32xi32>
    %a_9 = tt.load %a_8 : tensor<16x32x!tt.ptr<i8>>
    %a_scale = arith.muli %a_1, %cst_0 : tensor<16x1xi32>
    %a_scale_10 = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_scale_11 = tt.addptr %a_scale_10, %a_scale : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_scale_12 = tt.expand_dims %scale_offs {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %a_scale_13 = tt.broadcast %a_scale_11 : tensor<16x1x!tt.ptr<i8>> -> tensor<16x2x!tt.ptr<i8>>
    %a_scale_14 = tt.broadcast %a_scale_12 : tensor<1x2xi32> -> tensor<16x2xi32>
    %a_scale_15 = tt.addptr %a_scale_13, %a_scale_14 : tensor<16x2x!tt.ptr<i8>>, tensor<16x2xi32>
    %a_scale_16 = tt.load %a_scale_15 : tensor<16x2x!tt.ptr<i8>>
    %b_17 = tt.expand_dims %k_packed_offs {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %b_18 = arith.muli %b_17, %b : tensor<32x1xi32>
    %b_19 = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>>
    %b_20 = tt.addptr %b_19, %b_18 : tensor<32x1x!tt.ptr<i8>>, tensor<32x1xi32>
    %b_21 = tt.expand_dims %m_offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %b_22 = tt.broadcast %b_20 : tensor<32x1x!tt.ptr<i8>> -> tensor<32x16x!tt.ptr<i8>>
    %b_23 = tt.broadcast %b_21 : tensor<1x16xi32> -> tensor<32x16xi32>
    %b_24 = tt.addptr %b_22, %b_23 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %b_25 = tt.load %b_24 : tensor<32x16x!tt.ptr<i8>>
    %b_scale = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %b_scale_26 = tt.addptr %b_scale, %a_scale : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %b_scale_27 = tt.broadcast %b_scale_26 : tensor<16x1x!tt.ptr<i8>> -> tensor<16x2x!tt.ptr<i8>>
    %b_scale_28 = tt.addptr %b_scale_27, %a_scale_14 : tensor<16x2x!tt.ptr<i8>>, tensor<16x2xi32>
    %b_scale_29 = tt.load %b_scale_28 : tensor<16x2x!tt.ptr<i8>>
    %acc_30 = tt.dot_scaled %a_9 scale %a_scale_16, %b_25 scale %b_scale_29, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x32xi8>, tensor<16x2xi8> * tensor<32x16xi8>, tensor<16x2xi8> -> tensor<16x16xf32>
    %0 = arith.muli %a_1, %cst : tensor<16x1xi32>
    %1 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %3 = tt.broadcast %2 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %4 = tt.broadcast %b_21 : tensor<1x16xi32> -> tensor<16x16xi32>
    %5 = tt.addptr %3, %4 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %5, %acc_30 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}


// CHECK-LABEL: func.func @_dot_scaled_test_fp4
// CHECK: hfusion.matmul_mx
// CHECK: outs([[OUT:.*]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: bufferization.materialize_in_destination
