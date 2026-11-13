// RUN: triton-opt --triton-control-flow-opt --split-input-file %s -verify-each | FileCheck %s --check-prefix=CFO

module {
  tt.func public @scope_block_ptr_result(%base: !tt.ptr<f16>, %delta: i32) -> !tt.ptr<tensor<32xf16>> {
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i32 = arith.constant 0 : i32
    %result = scope.scope : () -> (!tt.ptr<tensor<32xf16>>) {
      %pointer = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
      %next = tt.advance %pointer, [%delta] : !tt.ptr<tensor<32xf16>>
      scope.return %next : !tt.ptr<tensor<32xf16>>
    } {hivm.disable_auto_sync = true, hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline, scope_test_attr = "kept"}
    tt.return %result : !tt.ptr<tensor<32xf16>>
  }
}

// CFO-LABEL: tt.func public @scope_block_ptr_result
// CFO:       %[[SCOPE:.*]]:4 = scope.scope : () -> (i64, i64, i64, i32) {
// CFO:         scope.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i64, i64, i64, i32
// CFO:       } {hivm.disable_auto_sync = true, hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline, scope_test_attr = "kept"}
// CFO:       tt.make_tensor_ptr
// CFO-SAME:  PointerDescriptorRebuild
// CFO-NOT:   PointerDescriptorBoundary

// -----

module {
  tt.func public @scope_block_ptr_mixed_results(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<16xf16>> {
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0_i32 = arith.constant 0 : i32
    %results:3 = scope.scope : () -> (i32, !tt.ptr<tensor<16xf16>>, tensor<4xi32>) {
      %ordinary = arith.constant 7 : i32
      %tensor = arith.constant dense<9> : tensor<4xi32>
      %pointer = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
      scope.return %ordinary, %pointer, %tensor : i32, !tt.ptr<tensor<16xf16>>, tensor<4xi32>
    }
    tt.return %results#1 : !tt.ptr<tensor<16xf16>>
  }
}

// CFO-LABEL: tt.func public @scope_block_ptr_mixed_results
// CFO:       %[[SCOPE:.*]]:6 = scope.scope : () -> (i32, i64, i64, i64, i32, tensor<4xi32>) {
// CFO:         scope.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i64, i64, i64, i32, tensor<4xi32>
// CFO:       }
// CFO:       %[[POINTER:.*]] = tt.make_tensor_ptr
// CFO-SAME:  PointerDescriptorRebuild
// CFO:       tt.return %[[POINTER]] : !tt.ptr<tensor<16xf16>>

// -----

module {
  tt.func public @scope_tensor_ptr_scalar_base(%base: !tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %result = scope.scope : () -> (tensor<4x!tt.ptr<f32>>) {
      %pointer = tt.addptr %base_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scope.return %pointer : tensor<4x!tt.ptr<f32>>
    }
    tt.return %result : tensor<4x!tt.ptr<f32>>
  }
}

// CFO-LABEL: tt.func public @scope_tensor_ptr_scalar_base
// CFO:       %[[SCOPE:.*]]:4 = scope.scope
// CFO-SAME:  -> (i64, i32, i32, tensor<4xi32>)
// CFO:         scope.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i64, i32, i32, tensor<4xi32>
// CFO:       }
// CFO:       %[[POINTER:.*]] = tt.addptr
// CFO-SAME:  PointerDescriptorOffsetForm = "strided_1d"
// CFO-SAME:  PointerDescriptorRebuild
// CFO-SAME:  PointerDescriptorStructuredAxes = array<i32: 1>
// CFO:       tt.return %[[POINTER]] : tensor<4x!tt.ptr<f32>>

// -----

module {
  tt.func public @scope_tensor_ptr_external_opaque_base(%lhs_base: !tt.ptr<f32>, %rhs_base: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %lhs = tt.splat %lhs_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs = tt.splat %rhs_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %condition = tt.splat %cond : i1 -> tensor<4xi1>
    %opaque = arith.select %condition, %lhs, %rhs : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
    %delta = arith.constant dense<3> : tensor<4xi32>
    %result = scope.scope : () -> (tensor<4x!tt.ptr<f32>>) {
      %pointer = tt.addptr %opaque, %delta : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scope.return %pointer : tensor<4x!tt.ptr<f32>>
    }
    tt.return %result : tensor<4x!tt.ptr<f32>>
  }
}

// CFO-LABEL: tt.func public @scope_tensor_ptr_external_opaque_base
// CFO:       %[[OPAQUE:.*]] = arith.select
// CFO:       %[[SCOPE:.*]]:3 = scope.scope
// CFO-SAME:  -> (i32, i32, tensor<4xi32>)
// CFO:         scope.return
// CFO:       }
// CFO:       %[[REBUILT:.*]] = tt.addptr %[[OPAQUE]],
// CFO-SAME:  PointerDescriptorRebuild
// CFO-NOT:   PointerDescriptorOffsetForm
// CFO:       tt.return %[[REBUILT]] : tensor<4x!tt.ptr<f32>>

// -----

module {
  tt.func public @scope_contains_pointer_for(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<16xf16>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %result = scope.scope : () -> (!tt.ptr<tensor<16xf16>>) {
      %initial = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
      %final = scf.for %iv = %c0 to %c2 step %c1 iter_args(%pointer = %initial) -> (!tt.ptr<tensor<16xf16>>) {
        %next = tt.advance %pointer, [%c1_i32] : !tt.ptr<tensor<16xf16>>
        scf.yield %next : !tt.ptr<tensor<16xf16>>
      }
      scope.return %final : !tt.ptr<tensor<16xf16>>
    } {scope_test_attr = "outer"}
    tt.return %result : !tt.ptr<tensor<16xf16>>
  }
}

// CFO-LABEL: tt.func public @scope_contains_pointer_for
// CFO:       scope.scope
// CFO:         scf.for
// CFO-SAME:    -> (i32)
// CFO:         } {PointerDescriptorBoundary = array<i32: 0>}
// CFO:       } {scope_test_attr = "outer"}

// -----

module {
  tt.func public @nested_scope_pointer_result(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<16xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %outer = scope.scope : () -> (!tt.ptr<tensor<16xf16>>) {
      %inner = scope.scope : () -> (!tt.ptr<tensor<16xf16>>) {
        %pointer = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
        scope.return %pointer : !tt.ptr<tensor<16xf16>>
      } {scope_test_attr = "inner"}
      %next = tt.advance %inner, [%c1_i32] : !tt.ptr<tensor<16xf16>>
      scope.return %next : !tt.ptr<tensor<16xf16>>
    } {scope_test_attr = "outer"}
    tt.return %outer : !tt.ptr<tensor<16xf16>>
  }
}

// CFO-LABEL: tt.func public @nested_scope_pointer_result
// CFO:       scope.scope : () -> (i64, i64, i64, i32) {
// CFO:         scope.scope : () -> (i64, i64, i64, i32) {
// CFO:         } {scope_test_attr = "inner"}
// CFO:       } {scope_test_attr = "outer"}

// -----

module {
  tt.func public @resultless_scope_contains_pointer_if(%base: !tt.ptr<f16>, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    scope.scope : () -> () {
      %initial = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
      %selected = scf.if %cond -> (!tt.ptr<tensor<16xf16>>) {
        %then = tt.advance %initial, [%c1_i32] : !tt.ptr<tensor<16xf16>>
        scf.yield %then : !tt.ptr<tensor<16xf16>>
      } else {
        %else = tt.advance %initial, [%c2_i32] : !tt.ptr<tensor<16xf16>>
        scf.yield %else : !tt.ptr<tensor<16xf16>>
      }
      %loaded = tt.load %selected : !tt.ptr<tensor<16xf16>>
      scope.return
    } {scope_test_attr = "resultless"}
    tt.return
  }
}

// CFO-LABEL: tt.func public @resultless_scope_contains_pointer_if
// CFO:       scope.scope : () -> () {
// CFO:         scf.if
// CFO-SAME:    -> (i32)
// CFO:         tt.load
// CFO:       } {scope_test_attr = "resultless"}

// -----

module {
  tt.func public @scope_attributes_preserved(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<16xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %result = scope.scope : () -> (!tt.ptr<tensor<16xf16>>) {
      %pointer = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
      scope.return %pointer : !tt.ptr<tensor<16xf16>>
    } {hivm.disable_auto_sync = true, hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline, scope_test_attr = "all-preserved"}
    tt.return %result : !tt.ptr<tensor<16xf16>>
  }
}

// CFO-LABEL: tt.func public @scope_attributes_preserved
// CFO:       scope.scope : () -> (i64, i64, i64, i32) {
// CFO:       } {hivm.disable_auto_sync = true, hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline, scope_test_attr = "all-preserved"}
