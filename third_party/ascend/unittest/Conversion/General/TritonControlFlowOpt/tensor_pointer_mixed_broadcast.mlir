// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // The original left-associated expression broadcasts pointers after adding
  // opaque row offsets. Preserve the structured column axis and i64 residual.
  // Only the uniform offset changes on the loop backedge.
  tt.func public @mixed_pointer_broadcast_loop(
      %base: !tt.ptr<f32>, %rows: tensor<2xi64>, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row_offsets = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %base_tensor, %row_offsets : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
    %broadcast = tt.broadcast %row_ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %columns = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %column_offsets = tt.broadcast %columns : tensor<1x32xi32> -> tensor<2x32xi32>
    %initial = tt.addptr %broadcast, %column_offsets : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %initial) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }

  // A unit column dimension can have a nonzero symbolic stride. Broadcasting
  // it repeats the same pointer: reset that stride, but keep the opaque rows.
  // Both branches must transfer their own broadcast residual through the if.
  tt.func public @mixed_pointer_broadcast_if(
      %base: !tt.ptr<f32>, %then_rows: tensor<2xi64>,
      %else_rows: tensor<2xi64>, %condition: i1)
      -> tensor<2x32x!tt.ptr<f32>> {
    %single = tt.make_range {end = 4 : i32, start = 3 : i32} : tensor<1xi32>
    %single_column = tt.expand_dims %single {axis = 0 : i32} : tensor<1xi32> -> tensor<1x1xi32>
    %column_offsets = tt.broadcast %single_column : tensor<1x1xi32> -> tensor<2x1xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %result = scf.if %condition -> (tensor<2x32x!tt.ptr<f32>>) {
      %rows = tt.expand_dims %then_rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
      %row_ptrs = tt.addptr %base_tensor, %rows : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
      %ptrs = tt.addptr %row_ptrs, %column_offsets : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
      %broadcast = tt.broadcast %ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
      scf.yield %broadcast : tensor<2x32x!tt.ptr<f32>>
    } else {
      %rows = tt.expand_dims %else_rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
      %row_ptrs = tt.addptr %base_tensor, %rows : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
      %ptrs = tt.addptr %row_ptrs, %column_offsets : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
      %broadcast = tt.broadcast %ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
      scf.yield %broadcast : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %result : tensor<2x32x!tt.ptr<f32>>
  }

  // Unknown tensor bases do not acquire a scalar-base descriptor or a claimed
  // structured axis merely because they are broadcast.
  tt.func public @unknown_pointer_broadcast(
      %source: tensor<2x1x!tt.ptr<f32>>, %upper: index)
      -> tensor<2x32x!tt.ptr<f32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %initial = tt.broadcast %source : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %initial) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    tt.return %final : tensor<2x32x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @mixed_pointer_broadcast_loop
// CHECK: tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CHECK-NOT: arith.subi
// CHECK: scf.for {{.*}} iter_args(%[[OFFSET:.*]] = %{{.*}}) -> (i64)
// CHECK: %[[NEXT:.*]] = arith.addi %[[OFFSET]], %{{.*}} : i64
// CHECK: scf.yield %[[NEXT]] : i64
// CHECK: PointerDescriptorStructuredAxes = array<i32: 0, 1>
// CHECK: tt.return

// CHECK-LABEL: tt.func public @mixed_pointer_broadcast_if
// CHECK: %[[SELECTED:.*]] = scf.if {{.*}} -> (tensor<2x32xi64>)
// CHECK: tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CHECK: scf.yield {{.*}} : tensor<2x32xi64>
// CHECK: } else {
// CHECK: tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CHECK: scf.yield {{.*}} : tensor<2x32xi64>
// CHECK: tt.addptr %{{.*}}, %[[SELECTED]] {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 0, 1>}
// CHECK: tt.return

// CHECK-LABEL: tt.func public @unknown_pointer_broadcast
// CHECK: tt.broadcast {{.*}} : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
// CHECK: scf.for {{.*}} -> (tensor<2x32xi32>)
// CHECK: PointerDescriptorStructuredAxes = array<i32: 0, 0>
// CHECK: tt.return
