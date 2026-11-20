// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -o - | FileCheck %s
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -canonicalize -cse -o - | FileCheck %s --check-prefix=POST
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -o - | FileCheck %s --check-prefix=CLEAN
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -canonicalize -cse -o - | FileCheck %s --check-prefix=POST-CLEAN
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -canonicalize -cse -o - | FileCheck %s --check-prefix=POST-EXPAND
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -canonicalize -cse -o - | FileCheck %s --check-prefix=POST-SPLAT
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -canonicalize -cse -o - | FileCheck %s --check-prefix=POST-MULI

// The rule owns one closed component: two masked loads, two stores, a binary
// fanout, dense-splat `other`/accumulator values, an ordered tt.assert, four
// loop-carried pointers, and two accumulators.  N is a runtime physical stride
// rather than a dynamic tensor dimension.  The active mask is intentionally
// retained so N=1 remains a legal runtime case.
// CHECK-LABEL: tt.func @masked_dynamic_stride_loop(
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation add"{{.*}} : tensor<3x2xi1>
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation add"{{.*}} : tensor<3x2xi1>
// CHECK-NOT: tensor<1x3xi64>
// CHECK-NOT: tensor<2x3xi64>
// CHECK: scf.for {{.*}} -> (tensor<3x2x!tt.ptr<f32>>, tensor<3x2x!tt.ptr<f32>>, tensor<3x2x!tt.ptr<f32>>, tensor<3x2x!tt.ptr<f32>>, tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: arith.addf {{.*}} : tensor<3x2xf32>
// CHECK: arith.mulf {{.*}} : tensor<3x2xf32>
// CHECK: tt.assert {{.*}}, "preserve loop assertion" : i1
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: "tt.reduce"({{.*}}) <{axis = 0 : i32}>
// CHECK: (tensor<3x2xf32>) -> tensor<2xf32>
// POST-LABEL: tt.func @masked_dynamic_stride_loop(
// POST: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// POST: tt.assert {{.*}}, "int32 overflow detected for operation add"{{.*}} : tensor<3x2xi1>
// POST: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// POST: tt.assert {{.*}}, "int32 overflow detected for operation add"{{.*}} : tensor<3x2xi1>
// POST-NOT: tensor<1x3xi64>
// POST-NOT: tensor<2x3xi64>
// POST: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>

// Keep the old i64 *and* i1 layout shapes forbidden over the whole function,
// rather than only in the interval after the checked assertions.  This makes
// a residual guard producer before the first assert observable to FileCheck.
// CLEAN-LABEL: tt.func @masked_dynamic_stride_loop(
// CLEAN-NOT: tensor<1x3xi64>
// CLEAN-NOT: tensor<2x3xi64>
// CLEAN-NOT: tensor<1x3xi1>
// CLEAN-NOT: tensor<2x3xi1>
// CLEAN-LABEL: tt.func @rank3_static_permutation_loop(
// POST-CLEAN-LABEL: tt.func @masked_dynamic_stride_loop(
// POST-CLEAN-NOT: tensor<1x3xi64>
// POST-CLEAN-NOT: tensor<2x3xi64>
// POST-CLEAN-NOT: tensor<1x3xi1>
// POST-CLEAN-NOT: tensor<2x3xi1>
// POST-CLEAN-LABEL: tt.func @rank3_static_permutation_loop(

// CSE/canonicalize is optional after graph-optimize, but it must not leave a
// second layout DAG in this minimal closed component.  Count each operation
// class in a dedicated FileCheck run because the three classes interleave.
// POST-EXPAND-LABEL: tt.func @masked_dynamic_stride_loop(
// POST-EXPAND-COUNT-2: tt.expand_dims
// POST-EXPAND-NOT: tt.expand_dims
// POST-EXPAND-LABEL: tt.func @rank3_static_permutation_loop(
// POST-SPLAT-LABEL: tt.func @masked_dynamic_stride_loop(
// POST-SPLAT-COUNT-9: tt.splat
// POST-SPLAT-NOT: tt.splat
// POST-SPLAT-LABEL: tt.func @rank3_static_permutation_loop(
// POST-MULI-LABEL: tt.func @masked_dynamic_stride_loop(
// POST-MULI-COUNT-2: arith.muli
// POST-MULI-NOT: arith.muli
// POST-MULI-LABEL: tt.func @rank3_static_permutation_loop(
tt.func @masked_dynamic_stride_loop(%src0: !tt.ptr<f32>, %src1: !tt.ptr<f32>, %dst0: !tt.ptr<f32>, %dst1: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %row = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %row_expand = tt.expand_dims %row {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %column_expand = tt.expand_dims %column {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %n_row = tt.splat %n : i32 -> tensor<1x3xi32>
  %row_i64 = arith.extsi %row_expand : tensor<1x3xi32> to tensor<1x3xi64>
  %n_i64 = arith.extsi %n : i32 to i64
  %n_i64_row = tt.splat %n_i64 : i64 -> tensor<1x3xi64>
  %row_product_i64 = arith.muli %row_i64, %n_i64_row : tensor<1x3xi64>
  %row_i64_max = arith.constant dense<2147483647> : tensor<1x3xi64>
  %row_i64_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
  %row_product_le = arith.cmpi sle, %row_product_i64, %row_i64_max : tensor<1x3xi64>
  %row_product_ge = arith.cmpi sge, %row_product_i64, %row_i64_min : tensor<1x3xi64>
  %row_product_ok = arith.andi %row_product_le, %row_product_ge : tensor<1x3xi1>
  tt.assert %row_product_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
  %row_scaled = arith.muli %row_expand, %n_row : tensor<1x3xi32>
  %row_full = tt.broadcast %row_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %column_full = tt.broadcast %column_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %row_full_i64 = arith.extsi %row_full : tensor<2x3xi32> to tensor<2x3xi64>
  %column_full_i64 = arith.extsi %column_full : tensor<2x3xi32> to tensor<2x3xi64>
  %offset_i64 = arith.addi %row_full_i64, %column_full_i64 : tensor<2x3xi64>
  %offset_i64_max = arith.constant dense<2147483647> : tensor<2x3xi64>
  %offset_i64_min = arith.constant dense<-2147483648> : tensor<2x3xi64>
  %offset_le = arith.cmpi sle, %offset_i64, %offset_i64_max : tensor<2x3xi64>
  %offset_ge = arith.cmpi sge, %offset_i64, %offset_i64_min : tensor<2x3xi64>
  %offset_ok = arith.andi %offset_le, %offset_ge : tensor<2x3xi1>
  tt.assert %offset_ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2x3xi1>
  %offset = arith.addi %column_full, %row_full : tensor<2x3xi32>
  %src0_base = tt.splat %src0 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src1_base = tt.splat %src1 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst0_base = tt.splat %dst0 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst1_base = tt.splat %dst1 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src0_ptr = tt.addptr %src0_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %src1_ptr = tt.addptr %src1_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst0_ptr = tt.addptr %dst0_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst1_ptr = tt.addptr %dst1_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  tt.assert %row_product_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
  tt.assert %offset_ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2x3xi1>
  %n_column = tt.splat %n : i32 -> tensor<2x1xi32>
  %column_mask = arith.cmpi slt, %column_expand, %n_column : tensor<2x1xi32>
  %mask = tt.broadcast %column_mask : tensor<2x1xi1> -> tensor<2x3xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
  // A shared producer is a legal fork DAG, not a recursive cycle.  The
  // external-value preflight must admit it and LayoutValueCloner must reuse
  // one natural-layout increment producer for both operands.
  %delta_base = tt.splat %n : i32 -> tensor<2x3xi32>
  %delta = arith.andi %delta_base, %delta_base : tensor<2x3xi32>
  %true = arith.constant true
  %loop:6 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%p0 = %src0_ptr, %p1 = %src1_ptr, %p2 = %dst0_ptr, %p3 = %dst1_ptr, %acc0 = %zero, %acc1 = %zero) -> (tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3xf32>, tensor<2x3xf32>) : i32 {
    %a = tt.load %p0, %mask, %zero : tensor<2x3x!tt.ptr<f32>>
    %b = tt.load %p1, %mask, %zero : tensor<2x3x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<2x3xf32>
    %product = arith.mulf %a, %b : tensor<2x3xf32>
    tt.assert %true, "preserve loop assertion" : i1
    %next_acc0 = arith.addf %acc0, %sum : tensor<2x3xf32>
    %next_acc1 = arith.addf %acc1, %product : tensor<2x3xf32>
    tt.store %p2, %sum, %mask : tensor<2x3x!tt.ptr<f32>>
    tt.store %p3, %product, %mask : tensor<2x3x!tt.ptr<f32>>
    %next_p0 = tt.addptr %p0, %delta : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %next_p1 = tt.addptr %p1, %delta : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %next_p2 = tt.addptr %p2, %delta : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %next_p3 = tt.addptr %p3, %delta : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    scf.yield %next_p0, %next_p1, %next_p2, %next_p3, %next_acc0, %next_acc1 : tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3x!tt.ptr<f32>>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  %reduced = "tt.reduce"(%loop#4) <{axis = 1 : i32}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %combined = arith.addf %lhs, %rhs : f32
    tt.reduce.return %combined : f32
  }) : (tensor<2x3xf32>) -> tensor<2xf32>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
  %out_ptr = tt.addptr %out_base, %column : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
  %out_n = tt.splat %n : i32 -> tensor<2xi32>
  %out_mask = arith.cmpi slt, %column, %out_n : tensor<2xi32>
  tt.store %out_ptr, %reduced, %out_mask : tensor<2x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @rank3_static_permutation_loop(
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1x1xi1>
// CHECK: scf.for {{.*}} -> (tensor<3x4x2x!tt.ptr<f32>>, tensor<3x4x2x!tt.ptr<f32>>)
// CHECK: tt.load {{.*}} : tensor<3x4x2x!tt.ptr<f32>>
// CHECK: tt.trans {{.*}} {order = array<i32: 0, 2, 1>} : tensor<3x4x2xf32> -> tensor<3x2x4xf32>
// CHECK: tt.trans {{.*}} {order = array<i32: 0, 2, 1>} : tensor<3x2x4xf32> -> tensor<3x4x2xf32>
// CHECK: tt.store {{.*}} : tensor<3x4x2x!tt.ptr<f32>>
tt.func @rank3_static_permutation_loop(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %axis0_e1 = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis0_e2 = tt.expand_dims %axis0_e1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %axis1_e0 = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %axis1_e2 = tt.expand_dims %axis1_e0 {axis = 2 : i32} : tensor<1x3xi32> -> tensor<1x3x1xi32>
  %axis2_e0 = tt.expand_dims %axis2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %axis2_e1 = tt.expand_dims %axis2_e0 {axis = 0 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %eight = arith.constant 8 : i32
  %s0 = tt.splat %one : i32 -> tensor<2x1x1xi32>
  %s1 = tt.splat %eight : i32 -> tensor<1x3x1xi32>
  %s2 = tt.splat %two : i32 -> tensor<1x1x4xi32>
  %t0 = arith.muli %axis0_e2, %s0 : tensor<2x1x1xi32>
  %axis1_i64 = arith.extsi %axis1_e2 : tensor<1x3x1xi32> to tensor<1x3x1xi64>
  %eight_i64 = arith.extsi %eight : i32 to i64
  %s1_i64 = tt.splat %eight_i64 : i64 -> tensor<1x3x1xi64>
  %t1_i64 = arith.muli %axis1_i64, %s1_i64 : tensor<1x3x1xi64>
  %t1_i64_max = arith.constant dense<2147483647> : tensor<1x3x1xi64>
  %t1_i64_min = arith.constant dense<-2147483648> : tensor<1x3x1xi64>
  %t1_le = arith.cmpi sle, %t1_i64, %t1_i64_max : tensor<1x3x1xi64>
  %t1_ge = arith.cmpi sge, %t1_i64, %t1_i64_min : tensor<1x3x1xi64>
  %t1_ok = arith.andi %t1_le, %t1_ge : tensor<1x3x1xi1>
  tt.assert %t1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3x1xi1>
  %t1 = arith.muli %axis1_e2, %s1 : tensor<1x3x1xi32>
  %t2 = arith.muli %axis2_e1, %s2 : tensor<1x1x4xi32>
  %b0 = tt.broadcast %t0 : tensor<2x1x1xi32> -> tensor<2x3x4xi32>
  %b1 = tt.broadcast %t1 : tensor<1x3x1xi32> -> tensor<2x3x4xi32>
  %b2 = tt.broadcast %t2 : tensor<1x1x4xi32> -> tensor<2x3x4xi32>
  %offset01 = arith.addi %b0, %b1 : tensor<2x3x4xi32>
  %offset = arith.addi %offset01, %b2 : tensor<2x3x4xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x4x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x4x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4xi32>
  %zero = arith.constant dense<0> : tensor<2x3x4xi32>
  %loop:2 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%p0 = %src_ptr, %p1 = %dst_ptr) -> (tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4x!tt.ptr<f32>>) : i32 {
    %value = tt.load %p0 : tensor<2x3x4x!tt.ptr<f32>>
    %transposed = tt.trans %value {order = array<i32: 2, 1, 0>} : tensor<2x3x4xf32> -> tensor<4x3x2xf32>
    %restored = tt.trans %transposed {order = array<i32: 2, 1, 0>} : tensor<4x3x2xf32> -> tensor<2x3x4xf32>
    %result = arith.negf %restored : tensor<2x3x4xf32>
    tt.store %p1, %result : tensor<2x3x4x!tt.ptr<f32>>
    %next_p0 = tt.addptr %p0, %zero : tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4xi32>
    %next_p1 = tt.addptr %p1, %zero : tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4xi32>
    scf.yield %next_p0, %next_p1 : tensor<2x3x4x!tt.ptr<f32>>, tensor<2x3x4x!tt.ptr<f32>>
  }
  tt.return
}

// A rank-3 dynamic physical layout uses the canonical mixed-radix strides
// 3*N, N, 1.  Only the innermost lane is runtime-bounded by N, but that is
// enough to prove the two adjacent capacity relations 3*N >= 3*N and
// N >= 1*N.  The rule must therefore handle an N-dimensional symbolic proof,
// not a rank-2 special case.
// CHECK-LABEL: tt.func @rank3_dynamic_stride_block(
// CHECK: tt.load {{.*}} : tensor<2x3x4x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<2x3x4x!tt.ptr<f32>>
tt.func @rank3_dynamic_stride_block(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
  %axis0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis0_e1 = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %axis0_e2 = tt.expand_dims %axis0_e1 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
  %axis1_e0 = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %axis1_e2 = tt.expand_dims %axis1_e0 {axis = 2 : i32} : tensor<1x3xi32> -> tensor<1x3x1xi32>
  %axis2_e0 = tt.expand_dims %axis2 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %axis2_e1 = tt.expand_dims %axis2_e0 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
  %n_axis1 = tt.splat %n : i32 -> tensor<1x3x1xi32>
  %axis1_scaled = arith.muli %axis1_e2, %n_axis1 : tensor<1x3x1xi32>
  %three = arith.constant 3 : i32
  %three_n = arith.muli %three, %n : i32
  %three_n_axis2 = tt.splat %three_n : i32 -> tensor<1x1x2xi32>
  %axis2_scaled = arith.muli %axis2_e1, %three_n_axis2 : tensor<1x1x2xi32>
  %axis0_full = tt.broadcast %axis0_e2 : tensor<4x1x1xi32> -> tensor<4x3x2xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3x1xi32> -> tensor<4x3x2xi32>
  %axis2_full = tt.broadcast %axis2_scaled : tensor<1x1x2xi32> -> tensor<4x3x2xi32>
  %offset01 = arith.addi %axis0_full, %axis1_full : tensor<4x3x2xi32>
  %offset = arith.addi %offset01, %axis2_full : tensor<4x3x2xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x3x2x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x3x2x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<4x3x2x!tt.ptr<f32>>, tensor<4x3x2xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<4x3x2x!tt.ptr<f32>>, tensor<4x3x2xi32>
  %n_axis0 = tt.splat %n : i32 -> tensor<4x1x1xi32>
  %axis0_mask = arith.cmpi slt, %axis0_e2, %n_axis0 : tensor<4x1x1xi32>
  %mask = tt.broadcast %axis0_mask : tensor<4x1x1xi1> -> tensor<4x3x2xi1>
  %value = tt.load %src_ptr, %mask : tensor<4x3x2x!tt.ptr<f32>>
  %result = arith.negf %value : tensor<4x3x2xf32>
  tt.store %dst_ptr, %result, %mask : tensor<4x3x2x!tt.ptr<f32>>
  tt.return
}

// The same closed component may live directly in a function block.  Its
// pointer/mask producer DAG is cloned locally, while the anchor remains one
// exact Block and does not require a synthetic scf.for port.
// CHECK-LABEL: tt.func @plain_block_component(
// CHECK: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<3x2x!tt.ptr<f32>>
tt.func @plain_block_component(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
  %result = arith.negf %value : tensor<2x3xf32>
  tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// A branch-local closure must also be eligible.  No result crosses the if
// port; when a layout value would reach scf.yield, the rule rejects rather
// than changing one branch's result type.
// CHECK-LABEL: tt.func @if_local_block_component(
// CHECK: scf.if {{.*}} {
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// CHECK: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<3x2x!tt.ptr<f32>>
tt.func @if_local_block_component(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %condition: i1) {
  scf.if %condition {
    %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %two = arith.constant 2 : i32
    %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
    %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
    %axis1_scaled_i64 = arith.extsi %axis1_scaled : tensor<1x3xi32> to tensor<1x3xi64>
    %axis1_max = arith.constant dense<2147483647> : tensor<1x3xi64>
    %axis1_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
    %axis1_le = arith.cmpi sle, %axis1_scaled_i64, %axis1_max : tensor<1x3xi64>
    %axis1_ge = arith.cmpi sge, %axis1_scaled_i64, %axis1_min : tensor<1x3xi64>
    %axis1_ok = arith.andi %axis1_le, %axis1_ge : tensor<1x3xi1>
    tt.assert %axis1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
    %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
    %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
    %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
    %result = arith.negf %value : tensor<2x3xf32>
    tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  } else {
  }
  tt.return
}

// The generic Block path also applies inside a while body when the entire
// pointer/guard/load/store closure is materialized in that one body Block.
// The scalar induction variable crosses the Region port, but no rank-2 layout
// value does.
// CHECK-LABEL: tt.func @while_local_block_component(
// CHECK: scf.while
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// CHECK: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<3x2x!tt.ptr<f32>>
tt.func @while_local_block_component(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %loop:1 = scf.while (%iter = %c0) : (i32) -> (i32) {
    %condition = arith.cmpi slt, %iter, %c1 : i32
    scf.condition(%condition) %iter : i32
  } do {
  ^bb0(%iter: i32):
    %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %two = arith.constant 2 : i32
    %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
    %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
    %axis1_scaled_i64 = arith.extsi %axis1_scaled : tensor<1x3xi32> to tensor<1x3xi64>
    %axis1_max = arith.constant dense<2147483647> : tensor<1x3xi64>
    %axis1_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
    %axis1_le = arith.cmpi sle, %axis1_scaled_i64, %axis1_max : tensor<1x3xi64>
    %axis1_ge = arith.cmpi sge, %axis1_scaled_i64, %axis1_min : tensor<1x3xi64>
    %axis1_ok = arith.andi %axis1_le, %axis1_ge : tensor<1x3xi1>
    tt.assert %axis1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
    %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
    %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
    %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
    %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
    %result = arith.negf %value : tensor<2x3xf32>
    tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
    %next = arith.addi %iter, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// The same Block rule must fail closed when its transformed value would leave
// through an if result.  This is intentionally not a cross-branch rewrite.
// CHECK-LABEL: tt.func @if_port_escape_is_rejected(
// CHECK-NOT: tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.load {{.*}} : tensor<2x3x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<2x3x!tt.ptr<f32>>
// CHECK: tt.return
tt.func @if_port_escape_is_rejected(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %condition: i1) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %if_result = scf.if %condition -> (tensor<2x3xf32>) {
    %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
    %result = arith.negf %value : tensor<2x3xf32>
    tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
    scf.yield %result : tensor<2x3xf32>
  } else {
    %zero = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
    scf.yield %zero : tensor<2x3xf32>
  }
  tt.store %dst_ptr, %if_result : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// A user-authored assertion is not an overflow port, even if it deliberately
// uses the same message.  The frontend-only marker distinguishes it from the
// preceding automatic assertion.  Because it observes a value in the old
// layout closure, the entire candidate must fail closed instead of silently
// transposing that user assertion.
// CHECK-LABEL: tt.func @user_assert_guard_escape_is_rejected(
// CHECK-COUNT-2: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<1x3xi1>
// CHECK-NOT: tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.load {{.*}} : tensor<2x3x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<2x3x!tt.ptr<f32>>
tt.func @user_assert_guard_escape_is_rejected(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis1_scaled_i64 = arith.extsi %axis1_scaled : tensor<1x3xi32> to tensor<1x3xi64>
  %axis1_max = arith.constant dense<2147483647> : tensor<1x3xi64>
  %axis1_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
  %axis1_le = arith.cmpi sle, %axis1_scaled_i64, %axis1_max : tensor<1x3xi64>
  %axis1_ge = arith.cmpi sge, %axis1_scaled_i64, %axis1_min : tensor<1x3xi64>
  %axis1_ok = arith.andi %axis1_le, %axis1_ge : tensor<1x3xi1>
  tt.assert %axis1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
  tt.assert %axis1_ok, "int32 overflow detected for operation mul" : tensor<1x3xi1>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
  %result = arith.negf %value : tensor<2x3xf32>
  tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// The automatic guard itself is eligible, but this pointer/guard closure
// originates outside the branch that owns the load/store endpoint.  Crossing
// a Region boundary would require a new port, so the block candidate is
// intentionally rejected.
// CHECK-LABEL: tt.func @cross_region_guard_is_rejected(
// CHECK: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<1x3xi1>
// CHECK: scf.if {{.*}} {
// CHECK-NOT: tensor<3x2x!tt.ptr<f32>>
// CHECK: tt.load {{.*}} : tensor<2x3x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<2x3x!tt.ptr<f32>>
tt.func @cross_region_guard_is_rejected(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %condition: i1) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis1_scaled_i64 = arith.extsi %axis1_scaled : tensor<1x3xi32> to tensor<1x3xi64>
  %axis1_max = arith.constant dense<2147483647> : tensor<1x3xi64>
  %axis1_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
  %axis1_le = arith.cmpi sle, %axis1_scaled_i64, %axis1_max : tensor<1x3xi64>
  %axis1_ge = arith.cmpi sge, %axis1_scaled_i64, %axis1_min : tensor<1x3xi64>
  %axis1_ok = arith.andi %axis1_le, %axis1_ge : tensor<1x3xi1>
  tt.assert %axis1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  scf.if %condition {
    %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
    %result = arith.negf %value : tensor<2x3xf32>
    tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  } else {
  }
  tt.return
}
