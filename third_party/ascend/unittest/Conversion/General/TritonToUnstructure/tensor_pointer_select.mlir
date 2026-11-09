// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s

tt.func public @opaque_tensor_pointer_select(
    %lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0> : tensor<4xi32>
  %one = arith.constant dense<1> : tensor<4xi32>
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %lhs_tensor = tt.splat %lhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %rhs_tensor = tt.splat %rhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %lhs_ptrs = tt.addptr %lhs_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %rhs_ptrs = tt.addptr %rhs_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %bits = arith.andi %range, %one : tensor<4xi32>
  %condition = arith.cmpi eq, %bits, %zero : tensor<4xi32>
  %selected = arith.select %condition, %lhs_ptrs, %rhs_ptrs : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
  %advanced = tt.addptr %selected, %one : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %loaded = tt.load %advanced : tensor<4x!tt.ptr<f32>>
  tt.return %loaded : tensor<4xf32>
}

// CHECK-LABEL: tt.func public @opaque_tensor_pointer_select
// CHECK:       %[[SELECTED:.*]] = arith.select {{.*}} : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
// CHECK:       scf.for
// CHECK:       %[[LANE_PTR:.*]] = tensor.extract %[[SELECTED]]{{\[}}%{{.*}}] {DiscreteMemAccess} : tensor<4x!tt.ptr<f32>>
// CHECK:       %[[ACCESS_PTR:.*]] = tt.addptr %[[LANE_PTR]],
// CHECK-SAME:  : !tt.ptr<f32>, i64
// CHECK:       tt.load %[[ACCESS_PTR]] {DiscreteMemAccess} : !tt.ptr<f32>
// CHECK:       tt.return
