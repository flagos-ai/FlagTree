// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s
// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s --check-prefix=COUNT
//
// Nested scalar and tensor pointer carriers must retain their live offsets.
// Retired copies must not remain in the function and duplicate its stores.
//
// CHECK-LABEL: tt.func public @nested_pointer_retirement(
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[STEP:.*]] = arith.constant dense<1> : tensor<4xi64>
// CHECK: scf.for {{.*}} iter_args(%[[OUTER_SCALAR:.*]] = {{.*}}, %[[OUTER_VECTOR:.*]] = {{.*}}) -> (i64, tensor<4xi64>)
// CHECK: %[[INNER:.*]]:2 = scf.for {{.*}} iter_args(%[[SCALAR:.*]] = %[[OUTER_SCALAR]], %[[VECTOR:.*]] = %[[OUTER_VECTOR]]) -> (i64, tensor<4xi64>)
// CHECK: tt.addptr {{.*}}, %[[VECTOR]] : tensor<4x!tt.ptr<i32>>, tensor<4xi64>
// CHECK: %[[POINTER:.*]] = tt.addptr {{.*}}, %[[SCALAR]] : !tt.ptr<i32>, i64
// CHECK: tt.load %[[POINTER]] : !tt.ptr<i32>
// CHECK: tt.store
// CHECK-DAG: %[[NEXT_SCALAR:.*]] = arith.addi %[[SCALAR]], %[[ONE]] : i64
// CHECK-DAG: %[[NEXT_VECTOR:.*]] = arith.addi %[[VECTOR]], %[[STEP]] : tensor<4xi64>
// CHECK: scf.yield %[[NEXT_SCALAR]], %[[NEXT_VECTOR]] : i64, tensor<4xi64>
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1 : i64, tensor<4xi64>
// CHECK: tt.return
//
// COUNT-LABEL: tt.func public @nested_pointer_retirement(
// COUNT-COUNT-2: scf.for
// COUNT-NOT: scf.for
// COUNT: tt.store
// COUNT-NOT: scf.for
// COUNT-NOT: tt.store
// COUNT: tt.return

tt.func public @nested_pointer_retirement(%base: !tt.ptr<i32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %step = arith.constant dense<1> : tensor<4xi32>
  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %initial = tt.addptr %splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %outer:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %base, %v = %initial) -> (!tt.ptr<i32>, tensor<4x!tt.ptr<i32>>) : i32 {
    %inner:2 = scf.for %j = %c0 to %n step %c1 iter_args(%q = %p, %w = %v) -> (!tt.ptr<i32>, tensor<4x!tt.ptr<i32>>) : i32 {
      %y = tt.load %q : !tt.ptr<i32>
      %ys = tt.splat %y : i32 -> tensor<4xi32>
      tt.store %w, %ys : tensor<4x!tt.ptr<i32>>
      %nextq = tt.addptr %q, %c1 : !tt.ptr<i32>, i32
      %nextw = tt.addptr %w, %step : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      scf.yield %nextq, %nextw : !tt.ptr<i32>, tensor<4x!tt.ptr<i32>>
    }
    scf.yield %inner#0, %inner#1 : !tt.ptr<i32>, tensor<4x!tt.ptr<i32>>
  }
  tt.return
}
