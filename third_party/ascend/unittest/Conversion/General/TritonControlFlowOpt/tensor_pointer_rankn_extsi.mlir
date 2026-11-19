// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=CFO

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_pointer_rankn_extsi(
      %base: !tt.ptr<f32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<1> : tensor<2x4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32}
        : tensor<4xi32> -> tensor<1x4xi32>
    %extended = arith.extsi %expanded
        : tensor<1x4xi32> to tensor<1x4xi64>
    %offset = tt.broadcast %extended
        : tensor<1x4xi64> -> tensor<2x4xi64>
    %base_tensor = tt.splat %base : !tt.ptr<f32>
        -> tensor<2x4x!tt.ptr<f32>>
    %initial = tt.addptr %base_tensor, %offset
        : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi64>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%ptr = %initial) -> (tensor<2x4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %delta
          : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %next : tensor<2x4x!tt.ptr<f32>>
    }
    %value = tt.load %final : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

// CFO-LABEL: tt.func public @tensor_pointer_rankn_extsi
// CFO:       scf.for {{.*}}iter_args({{.*}}) -> (i64)
// CFO:       PointerDescriptorStructuredAxes = array<i32: 1, 1>
