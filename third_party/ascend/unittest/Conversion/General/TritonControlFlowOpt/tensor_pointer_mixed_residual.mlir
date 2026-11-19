// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // A nonzero range start and an explicit extension may update the scalar
  // fields without computing (rows + columns) - columns. The loop still
  // transfers only its i64 uniform offset.
  tt.func public @mixed_extended_range(
      %base: !tt.ptr<f32>, %rows: tensor<2xi64>, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %splat, %row : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
    %broadcast = tt.broadcast %row_ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %range = tt.make_range {end = 39 : i32, start = 7 : i32} : tensor<32xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %extended = arith.extsi %expanded : tensor<1x32xi32> to tensor<1x32xi64>
    %columns = tt.broadcast %extended : tensor<1x32xi64> -> tensor<2x32xi64>
    %initial = tt.addptr %broadcast, %columns : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi64>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %initial) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }

  // Do not distribute sign-extension across i32 arithmetic: shift=INT_MAX
  // wraps some column lanes before tt.addptr sign-extends them to i64.
  tt.func public @mixed_narrow_add(
      %base: !tt.ptr<f32>, %rows: tensor<2xi64>, %shift: i32, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %splat, %row : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
    %broadcast = tt.broadcast %row_ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %columns = tt.broadcast %expanded : tensor<1x32xi32> -> tensor<2x32xi32>
    %shifts = tt.splat %shift : i32 -> tensor<2x32xi32>
    %offsets = arith.addi %columns, %shifts : tensor<2x32xi32>
    %result = tt.addptr %broadcast, %offsets : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %result) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }

  // The existing descriptor itself can be narrow. Its complete i32 offset
  // must be reconstructed before widening, even if the new range is safe.
  tt.func public @mixed_narrow_descriptor(
      %base: !tt.ptr<f32>, %rows: tensor<2xi32>, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %row_offsets = tt.broadcast %row : tensor<2x1xi32> -> tensor<2x32xi32>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %columns = tt.broadcast %expanded : tensor<1x32xi32> -> tensor<2x32xi32>
    %offsets = arith.addi %row_offsets, %columns : tensor<2x32xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x32x!tt.ptr<f32>>
    %initial = tt.addptr %splat, %offsets : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %wide_columns = arith.extsi %columns : tensor<2x32xi32> to tensor<2x32xi64>
    %result = tt.addptr %initial, %wide_columns : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi64>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %result) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }

  // A range on an opaque axis belongs in the residual, even though the range
  // itself has no opaque contribution and does not change the axis kinds.
  tt.func public @mixed_opaque_axis_update(
      %base: !tt.ptr<f32>, %rows: tensor<2xi64>, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %splat, %row : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
    %broadcast = tt.broadcast %row_ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %range = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %expanded = tt.expand_dims %range {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %row_delta = tt.broadcast %expanded : tensor<2x1xi32> -> tensor<2x32xi32>
    %result = tt.addptr %broadcast, %row_delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %result) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @mixed_extended_range
// CHECK: %[[ROWS:.*]] = tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CHECK-NOT: arith.subi
// CHECK: scf.for {{.*}} iter_args(%[[OFFSET:.*]] = %{{.*}}) -> (i64)
// CHECK: %[[NEXT:.*]] = arith.addi %[[OFFSET]], %{{.*}} : i64
// CHECK: scf.yield %[[NEXT]] : i64
// CHECK: arith.addi %{{.*}}, %[[ROWS]] : tensor<2x32xi64>
// CHECK: tt.return

// CHECK-LABEL: tt.func public @mixed_narrow_add
// CHECK: %[[NARROW:.*]] = arith.addi {{.*}} : tensor<2x32xi32>
// CHECK: arith.extsi %[[NARROW]] : tensor<2x32xi32> to tensor<2x32xi64>
// CHECK: arith.subi {{.*}} : tensor<2x32xi64>
// CHECK: tt.return

// CHECK-LABEL: tt.func public @mixed_narrow_descriptor
// CHECK: arith.addi {{.*}} : tensor<2x32xi32>
// CHECK: arith.extsi {{.*}} : tensor<2x32xi32> to tensor<2x32xi64>
// CHECK: arith.subi {{.*}} : tensor<2x32xi64>
// CHECK: tt.return

// CHECK-LABEL: tt.func public @mixed_opaque_axis_update
// CHECK: %[[ROWS:.*]] = tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CHECK: %[[DELTA:.*]] = arith.extsi {{.*}} : tensor<2x32xi32> to tensor<2x32xi64>
// CHECK: arith.addi %[[ROWS]], %[[DELTA]] : tensor<2x32xi64>
// CHECK: PointerDescriptorStructuredAxes = array<i32: 0, 1>
// CHECK: tt.return
