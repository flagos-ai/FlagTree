// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=FALLBACK

module {
  tt.func public @scope_internal_opaque_tensor_pointer(%lhs_base: !tt.ptr<f32>, %rhs_base: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %lhs = tt.splat %lhs_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs = tt.splat %rhs_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %condition = tt.splat %cond : i1 -> tensor<4xi1>
    %result = scope.scope : () -> (tensor<4x!tt.ptr<f32>>) {
      %opaque = arith.select %condition, %lhs, %rhs : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
      scope.return %opaque : tensor<4x!tt.ptr<f32>>
    }
    tt.return %result : tensor<4x!tt.ptr<f32>>
  }
}

// FALLBACK-LABEL: tt.func public @scope_internal_opaque_tensor_pointer
// FALLBACK:       scope.scope
// FALLBACK-SAME:  tensor<4x!tt.ptr<f32>>
// FALLBACK:       arith.select
// FALLBACK:       tt.return
