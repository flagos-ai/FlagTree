// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --triton-to-linalg --split-input-file %s -verify-each | FileCheck %s --check-prefix=E2E --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast --implicit-check-not=PointerDescriptorBoundary --implicit-check-not=PointerDescriptorRebuild --implicit-check-not=PointerDescriptorOffsetForm --implicit-check-not=PointerDescriptorStructuredAxes

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @scope_block_ptr_load_store(%base: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %pointer = scope.scope : () -> (!tt.ptr<tensor<16xf32>>) {
      %initial = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
      %next = tt.advance %initial, [%c3_i32] : !tt.ptr<tensor<16xf32>>
      scope.return %next : !tt.ptr<tensor<16xf32>>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    %value = tt.load %pointer : !tt.ptr<tensor<16xf32>>
    %output_pointer = tt.make_tensor_ptr %output, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    tt.store %output_pointer, %value : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// E2E-LABEL: func.func @scope_block_ptr_load_store
// E2E:       scope.scope
// E2E:       memref.copy
// E2E:       bufferization.materialize_in_destination %{{.*}} in writable %{{.*}}
// E2E:       return

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @scope_tensor_ptr_masked_load_store(%base: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pointer = scope.scope : () -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %base_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scope.return %next : tensor<4x!tt.ptr<f32>>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    %mask = arith.constant dense<true> : tensor<4xi1>
    %other = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %value = tt.load %pointer, %mask, %other : tensor<4x!tt.ptr<f32>>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_pointer = tt.addptr %output_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_pointer, %value, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// E2E-LABEL: func.func @scope_tensor_ptr_masked_load_store
// E2E:       scope.scope
// E2E:       memref.copy
// E2E:       bufferization.materialize_in_destination %{{.*}} in writable %{{.*}}
// E2E:       return
