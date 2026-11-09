// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s --check-prefix=NO-ADVANCE

// NO-ADVANCE: module
// NO-ADVANCE-NOT: tt.advance

module {
  tt.func public @for_block_ptr_dynamic_step(%base: !tt.ptr<f16>, %ub: index) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %final = scf.for %iv = %c0 to %ub step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<32xf16>>) {
      %iv_i32 = arith.index_cast %iv : index to i32
      %next = tt.advance %ptr, [%iv_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %next : !tt.ptr<tensor<32xf16>>
    }
    tt.return %final : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_dynamic_step
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}) -> (i32) {
// CHECK-NOT:     arith.muli
// CHECK:         tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[OFF]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>
// CHECK:         %[[NEXT:.*]] = arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT]] : i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @for_block_ptr_invariant_delta_carried(%base: !tt.ptr<f16>, %ub: index, %stride: i32) -> tensor<32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty_i32 = tt.splat %c0_i32 : i32 -> tensor<32xi32>
    %empty_f16 = arith.sitofp %empty_i32 : tensor<32xi32> to tensor<32xf16>
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.for %iv = %c0 to %ub step %c1 iter_args(%ptr = %ptr0, %acc = %empty_f16) -> (!tt.ptr<tensor<32xf16>>, tensor<32xf16>) {
      %next = tt.advance %ptr, [%stride] : !tt.ptr<tensor<32xf16>>
      %loaded = tt.load %next : !tt.ptr<tensor<32xf16>>
      scf.yield %next, %loaded : !tt.ptr<tensor<32xf16>>, tensor<32xf16>
    }
    tt.return %res#1 : tensor<32xf16>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_invariant_delta_carried(
// CHECK-SAME:  %[[BASE:[^ ,]+]]: !tt.ptr<f16>
// CHECK:       %[[BASE_ADDRESS:.*]] = tt.ptr_to_int %[[BASE]]
// CHECK:       %[[FOR:[^:]+]]:2 = scf.for
// CHECK-SAME:  {{.*}} -> (i32, tensor<32xf16>) {
// CHECK-NOT:     arith.index_cast
// CHECK-NOT:     arith.muli
// CHECK:         %[[NEXT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NOT:     tt.int_to_ptr %[[BASE_ADDRESS]]
// CHECK:         %[[LOAD_PTR:.*]] = tt.make_tensor_ptr %[[BASE]], [%{{.*}}], [%{{.*}}], [%[[NEXT]]] {order = array<i32: 0>} : <tensor<32xf16>>
// CHECK:         %[[LOADED:.*]] = tt.load %[[LOAD_PTR]] : !tt.ptr<tensor<32xf16>>
// CHECK:         scf.yield %[[NEXT]], %[[LOADED]] : i32, tensor<32xf16>
// CHECK:       }
// CHECK:       tt.return %[[FOR]]#1 : tensor<32xf16>

// -----

module {
  tt.func public @for_block_ptr_dynamic_bounds_and_step(%base: !tt.ptr<f16>, %lb: index, %ub: index, %step: index, %delta: i32) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %final = scf.for %iv = %lb to %ub step %step iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<32xf16>>) {
      %next = tt.advance %ptr, [%delta] : !tt.ptr<tensor<32xf16>>
      scf.yield %next : !tt.ptr<tensor<32xf16>>
    }
    tt.return %final : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_dynamic_bounds_and_step
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}) -> (i32) {
// CHECK-NOT:     arith.muli
// CHECK:         %[[NEXT:.*]] = arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT]] : i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @for_block_ptr_iter_arg_delta(%base: !tt.ptr<f16>, %ub: index) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.for %iv = %c0 to %ub step %c1 iter_args(%ptr = %ptr0, %delta = %c1_i32) -> (!tt.ptr<tensor<32xf16>>, i32) {
      %next_ptr = tt.advance %ptr, [%delta] : !tt.ptr<tensor<32xf16>>
      %next_delta = arith.addi %delta, %c1_i32 : i32
      scf.yield %next_ptr, %next_delta : !tt.ptr<tensor<32xf16>>, i32
    }
    tt.return %res#0 : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_iter_arg_delta
// CHECK:       %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}, %[[DELTA:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK:         %[[NEXT_PTR:.*]] = arith.addi %[[OFF]], %[[DELTA]] : i32
// CHECK:         %[[NEXT_DELTA:.*]] = arith.addi %[[DELTA]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT_PTR]], %[[NEXT_DELTA]] : i32, i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]#0] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @for_block_ptr_multidim_delta(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<16x32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c2_i32, %c3_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<16x32xf16>>
    %final = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<16x32xf16>>) {
      %next = tt.advance %ptr, [%c1_i32, %c2_i32] : !tt.ptr<tensor<16x32xf16>>
      scf.yield %next : !tt.ptr<tensor<16x32xf16>>
    }
    tt.return %final : !tt.ptr<tensor<16x32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_multidim_delta
// CHECK:       %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[OFF0:.*]] = %{{.*}}, %[[OFF1:.*]] = %{{.*}})
// CHECK:         %[[NEXT0:.*]] = arith.addi %[[OFF0]], %{{.*}} : i32
// CHECK:         %[[NEXT1:.*]] = arith.addi %[[OFF1]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT0]], %[[NEXT1]] : i32, i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%[[FOR]]#0, %[[FOR]]#1] {{.*}}order = array<i32: 1, 0>{{.*}} : <tensor<16x32xf16>>

// -----

module {
  tt.func public @for_block_ptr_zero_trip(%base: !tt.ptr<f16>) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %final = scf.for %iv = %c4 to %c0 step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<32xf16>>) {
      %next = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %next : !tt.ptr<tensor<32xf16>>
    }
    tt.return %final : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_block_ptr_zero_trip
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}) -> (i32) {
// CHECK:         %[[NEXT:.*]] = arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT]] : i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @while_block_ptr_basic(%base: !tt.ptr<f16>, %n: i32) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.while (%i = %c0_i32, %ptr = %ptr0) : (i32, !tt.ptr<tensor<32xf16>>) -> (i32, !tt.ptr<tensor<32xf16>>) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %ptr : i32, !tt.ptr<tensor<32xf16>>
    } do {
    ^bb0(%i_a: i32, %ptr_a: !tt.ptr<tensor<32xf16>>):
      %next_i = arith.addi %i_a, %c1_i32 : i32
      %next_ptr = tt.advance %ptr_a, [%c1_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %next_i, %next_ptr : i32, !tt.ptr<tensor<32xf16>>
    }
    tt.return %res#1 : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @while_block_ptr_basic
// CHECK:       %[[WHILE:.*]]:2 = scf.while
// CHECK-SAME:  : (i32, i32) -> (i32, i32) {
// CHECK:       scf.condition(%{{.*}}) %{{.*}}, %{{.*}} : i32, i32
// CHECK:       } do {
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%{{.*}}] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>
// CHECK:       scf.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[WHILE]]#1] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @if_block_ptr_same_base_offsets(%base: !tt.ptr<f16>, %cond: i1) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %selected = scf.if %cond -> (!tt.ptr<tensor<32xf16>>) {
      %then_ptr = tt.advance %ptr0, [%c1_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %then_ptr : !tt.ptr<tensor<32xf16>>
    } else {
      %else_ptr = tt.advance %ptr0, [%c2_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %else_ptr : !tt.ptr<tensor<32xf16>>
    }
    tt.return %selected : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @if_block_ptr_same_base_offsets
// CHECK:       %[[OFF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[OFF]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @if_tensor_ptr_same_base_offsets(%base: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      %then_ptr = tt.addptr %splat, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %then_ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %else_ptr = tt.addptr %splat, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %else_ptr : tensor<4x!tt.ptr<f32>>
    }
    tt.return %selected : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @if_tensor_ptr_same_base_offsets
// CHECK:       %[[OFF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:       PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:  PointerDescriptorRebuild

// -----

module {
  tt.func public @if_tensor_ptr_same_base_chained_addptr(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %cond: i1) {
    %then_offset = arith.constant dense<2> : tensor<16xi32>
    %else_offset = arith.constant dense<5> : tensor<16xi32>
    %lane = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %selected = scf.if %cond -> (tensor<16x!tt.ptr<f32>>) {
      %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %lane_ptr = tt.addptr %base_tensor, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %then_ptr = tt.addptr %lane_ptr, %then_offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %then_ptr : tensor<16x!tt.ptr<f32>>
    } else {
      %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %lane_ptr = tt.addptr %base_tensor, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %else_ptr = tt.addptr %lane_ptr, %else_offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %else_ptr : tensor<16x!tt.ptr<f32>>
    }
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %loaded = tt.load %selected : tensor<16x!tt.ptr<f32>>
    tt.store %output_ptr, %loaded : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @if_tensor_ptr_same_base_chained_addptr
// CHECK:       %[[CHAINED_OFF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       %[[CHAINED_PTR:.*]] = tt.addptr
// CHECK-SAME:  PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:  PointerDescriptorRebuild
// CHECK:       tt.load %[[CHAINED_PTR]] : tensor<16x!tt.ptr<f32>>

// -----

module {
  tt.func public @while_block_ptr_large_step(%base: !tt.ptr<f16>, %n: i32) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c37_i32 = arith.constant 37 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %ptr0 = tt.make_tensor_ptr %base, [%c128_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.while (%i = %c0_i32, %ptr = %ptr0) : (i32, !tt.ptr<tensor<32xf16>>) -> (i32, !tt.ptr<tensor<32xf16>>) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %ptr : i32, !tt.ptr<tensor<32xf16>>
    } do {
    ^bb0(%i_a: i32, %ptr_a: !tt.ptr<tensor<32xf16>>):
      %next_i = arith.addi %i_a, %c1_i32 : i32
      %next_ptr = tt.advance %ptr_a, [%c37_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %next_i, %next_ptr : i32, !tt.ptr<tensor<32xf16>>
    }
    tt.return %res#1 : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @while_block_ptr_large_step
// CHECK-DAG:   %[[C37:.*]] = arith.constant 37 : i32
// CHECK:       %[[WHILE:.*]]:2 = scf.while
// CHECK-SAME:  : (i32, i32) -> (i32, i32)
// CHECK:       } do {
// CHECK:         %[[NEXT_OFF:.*]] = arith.addi %{{.*}}, %[[C37]] : i32
// CHECK:         scf.yield %{{.*}}, %[[NEXT_OFF]] : i32, i32

// -----

module {
  tt.func public @while_tensor_ptr_addptr_step(%base: !tt.ptr<f32>, %n: i32) -> tensor<4x!tt.ptr<f32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %res:2 = scf.while (%i = %c0_i32, %ptr = %splat) : (i32, tensor<4x!tt.ptr<f32>>) -> (i32, tensor<4x!tt.ptr<f32>>) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %ptr : i32, tensor<4x!tt.ptr<f32>>
    } do {
    ^bb0(%i_a: i32, %ptr_a: tensor<4x!tt.ptr<f32>>):
      %next_i = arith.addi %i_a, %c1_i32 : i32
      %step = tt.splat %c4_i32 : i32 -> tensor<4xi32>
      %next_ptr = tt.addptr %ptr_a, %step : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next_i, %next_ptr : i32, tensor<4x!tt.ptr<f32>>
    }
    tt.return %res#1 : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @while_tensor_ptr_addptr_step
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
// CHECK:       scf.while
// CHECK-SAME:  i32
// CHECK:       } do {
// CHECK:         %[[STEP:.*]] = tt.splat %{{.*}} : i32 -> tensor<4xi32>
// CHECK:         %[[NEXT_I:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:         %[[NEXT_OFF:.*]] = arith.addi %{{.*}}, %[[C4]] : i32
// CHECK:         scf.yield %[[NEXT_I]], %[[NEXT_OFF]] : i32, i32

// -----

module {
  tt.func public @for_tensor_ptr_preserves_wide_loop_offset(%base: !tt.ptr<f32>, %ub: index) -> tensor<4x!tt.ptr<f32>> {
    %c1_i32 = arith.constant 1 : i32
    %c8_i64 = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %wide_offset = tt.splat %c8_i64 : i64 -> tensor<4xi64>
    %narrow_step = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %splat, %wide_offset : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %final = scf.for %iv = %c0 to %ub step %c1 iter_args(%ptr = %ptr0) -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %narrow_step : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    tt.return %final : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @for_tensor_ptr_preserves_wide_loop_offset
// CHECK:       %[[STEP32:.*]] = arith.constant 1 : i32
// CHECK:       %[[INITIAL:.*]] = arith.constant 8 : i64
// CHECK:       %[[FOR:.*]] = scf.for
// CHECK-SAME:  iter_args(%[[LOOP_OFFSET:.*]] = %[[INITIAL]]) -> (i64) {
// CHECK:         %[[STEP64:.*]] = arith.extsi %[[STEP32]] : i32 to i64
// CHECK:         %[[NEXT:.*]] = arith.addi %[[LOOP_OFFSET]], %[[STEP64]] : i64
// CHECK:         scf.yield %[[NEXT]] : i64
// CHECK:       } {PointerDescriptorBoundary = array<i32: 0>}
// CHECK:       %[[TENSOR_OFFSET:.*]] = tt.splat %[[FOR]] : i64 -> tensor<4xi64>
// CHECK:       tt.addptr %{{.*}}, %{{.*}} {{.*}}PointerDescriptorRebuild{{.*}} : tensor<4x!tt.ptr<f32>>, tensor<4xi64>

// -----

module {
  tt.func public @for_if_block_ptr_same_base_post_advance(%base: !tt.ptr<f16>, %cond: i1) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %final = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<32xf16>>) {
      %selected = scf.if %cond -> (!tt.ptr<tensor<32xf16>>) {
        %then_ptr = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %then_ptr : !tt.ptr<tensor<32xf16>>
      } else {
        %else_ptr = tt.advance %ptr, [%c3_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %else_ptr : !tt.ptr<tensor<32xf16>>
      }
      %next = tt.advance %selected, [%c2_i32] : !tt.ptr<tensor<32xf16>>
      scf.yield %next : !tt.ptr<tensor<32xf16>>
    }
    tt.return %final : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @for_if_block_ptr_same_base_post_advance
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}) -> (i32) {
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         %[[NEXT:.*]] = arith.addi %[[SELECTED]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT]] : i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @while_if_block_ptr_same_base_post_advance(%base: !tt.ptr<f16>, %n: i32) -> !tt.ptr<tensor<32xf16>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.while (%i = %c0_i32, %ptr = %ptr0) : (i32, !tt.ptr<tensor<32xf16>>) -> (i32, !tt.ptr<tensor<32xf16>>) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %ptr : i32, !tt.ptr<tensor<32xf16>>
    } do {
    ^bb0(%i_a: i32, %ptr_a: !tt.ptr<tensor<32xf16>>):
      %is_one = arith.cmpi eq, %i_a, %c1_i32 : i32
      %selected = scf.if %is_one -> (!tt.ptr<tensor<32xf16>>) {
        %then_ptr = tt.advance %ptr_a, [%c3_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %then_ptr : !tt.ptr<tensor<32xf16>>
      } else {
        %else_ptr = tt.advance %ptr_a, [%c1_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %else_ptr : !tt.ptr<tensor<32xf16>>
      }
      %next_ptr = tt.advance %selected, [%c2_i32] : !tt.ptr<tensor<32xf16>>
      %next_i = arith.addi %i_a, %c1_i32 : i32
      scf.yield %next_i, %next_ptr : i32, !tt.ptr<tensor<32xf16>>
    }
    tt.return %res#1 : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL: tt.func public @while_if_block_ptr_same_base_post_advance
// CHECK:       %[[WHILE:.*]]:2 = scf.while
// CHECK-SAME:  (i32, i32) -> (i32, i32)
// CHECK:       } do {
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         %[[NEXT:.*]] = arith.addi %[[SELECTED]], %{{.*}} : i32
// CHECK:         scf.yield %{{.*}}, %[[NEXT]] : i32, i32
// CHECK:       }
// CHECK:       tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[WHILE]]#1] {{.*}}order = array<i32: 0>{{.*}} : <tensor<32xf16>>

// -----

module {
  tt.func public @for_if_tensor_ptr_same_base_post_addptr(%base: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %off3 = tt.splat %c3_i32 : i32 -> tensor<4xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %final = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %splat) -> (tensor<4x!tt.ptr<f32>>) {
      %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
        %then_ptr = tt.addptr %ptr, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        scf.yield %then_ptr : tensor<4x!tt.ptr<f32>>
      } else {
        %else_ptr = tt.addptr %ptr, %off3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        scf.yield %else_ptr : tensor<4x!tt.ptr<f32>>
      }
      %next = tt.addptr %selected, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    tt.return %final : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @for_if_tensor_ptr_same_base_post_addptr
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG:   %[[POST_STEP:.*]] = tt.splat %[[C2]] : i32 -> tensor<4xi32>
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}) -> (i32) {
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         %[[NEXT:.*]] = arith.addi %[[SELECTED]], %{{.*}} : i32
// CHECK:         scf.yield %[[NEXT]] : i32
// CHECK:       }
// CHECK:       PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:  PointerDescriptorRebuild

// -----

module {
  tt.func public @for_if_block_ptr_load_after_post_advance(%base: !tt.ptr<f16>, %cond: i1) -> tensor<32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %empty = tt.splat %c0_i32 : i32 -> tensor<32xi32>
    %empty_f16 = arith.sitofp %empty : tensor<32xi32> to tensor<32xf16>
    %ptr0 = tt.make_tensor_ptr %base, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %res:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0, %acc = %empty_f16) -> (!tt.ptr<tensor<32xf16>>, tensor<32xf16>) {
      %selected = scf.if %cond -> (!tt.ptr<tensor<32xf16>>) {
        %then_ptr = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %then_ptr : !tt.ptr<tensor<32xf16>>
      } else {
        %else_ptr = tt.advance %ptr, [%c3_i32] : !tt.ptr<tensor<32xf16>>
        scf.yield %else_ptr : !tt.ptr<tensor<32xf16>>
      }
      %next = tt.advance %selected, [%c2_i32] : !tt.ptr<tensor<32xf16>>
      %loaded = tt.load %next : !tt.ptr<tensor<32xf16>>
      scf.yield %next, %loaded : !tt.ptr<tensor<32xf16>>, tensor<32xf16>
    }
    tt.return %res#1 : tensor<32xf16>
  }
}

// CHECK-LABEL: tt.func public @for_if_block_ptr_load_after_post_advance
// CHECK:       %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}, %{{.*}} = %{{.*}})
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.addi %[[OFF]], %{{.*}} : i32
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         %[[NEXT:.*]] = arith.addi %[[SELECTED]], %{{.*}} : i32
// CHECK:         %[[LOAD_PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[NEXT]]] {order = array<i32: 0>} : <tensor<32xf16>>
// CHECK:         %[[LOADED:.*]] = tt.load %[[LOAD_PTR]] : !tt.ptr<tensor<32xf16>>
// CHECK:         scf.yield %[[NEXT]], %[[LOADED]] : i32, tensor<32xf16>
// CHECK:       }
// CHECK:       tt.return %[[FOR]]#1 : tensor<32xf16>

// -----

module {
  tt.func public @for_nested_if_block_ptr_load_after_post_advance(%base: !tt.ptr<f32>, %cond0: i1, %cond1: i1) -> tensor<16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %empty_i32 = tt.splat %c0_i32 : i32 -> tensor<16xi32>
    %empty_f32 = arith.sitofp %empty_i32 : tensor<16xi32> to tensor<16xf32>
    %ptr0 = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %res:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0, %acc = %empty_f32) -> (!tt.ptr<tensor<16xf32>>, tensor<16xf32>) {
      %selected = scf.if %cond0 -> (!tt.ptr<tensor<16xf32>>) {
        %a = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<16xf32>>
        %inner = scf.if %cond1 -> (!tt.ptr<tensor<16xf32>>) {
          %b = tt.advance %a, [%c2_i32] : !tt.ptr<tensor<16xf32>>
          scf.yield %b : !tt.ptr<tensor<16xf32>>
        } else {
          %b = tt.advance %a, [%c4_i32] : !tt.ptr<tensor<16xf32>>
          scf.yield %b : !tt.ptr<tensor<16xf32>>
        }
        scf.yield %inner : !tt.ptr<tensor<16xf32>>
      } else {
        %a = tt.advance %ptr, [%c3_i32] : !tt.ptr<tensor<16xf32>>
        %inner = scf.if %cond1 -> (!tt.ptr<tensor<16xf32>>) {
          %b = tt.advance %a, [%c1_i32] : !tt.ptr<tensor<16xf32>>
          scf.yield %b : !tt.ptr<tensor<16xf32>>
        } else {
          %b = tt.advance %a, [%c2_i32] : !tt.ptr<tensor<16xf32>>
          scf.yield %b : !tt.ptr<tensor<16xf32>>
        }
        scf.yield %inner : !tt.ptr<tensor<16xf32>>
      }
      %next = tt.advance %selected, [%c1_i32] : !tt.ptr<tensor<16xf32>>
      %loaded = tt.load %next : !tt.ptr<tensor<16xf32>>
      scf.yield %next, %loaded : !tt.ptr<tensor<16xf32>>, tensor<16xf32>
    }
    tt.return %res#1 : tensor<16xf32>
  }
}

// CHECK-LABEL: tt.func public @for_nested_if_block_ptr_load_after_post_advance
// CHECK:       %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}, %{{.*}} = %{{.*}})
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           %[[THEN_INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           } else {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           }
// CHECK:           scf.yield %[[THEN_INNER]] : i32
// CHECK:         } else {
// CHECK:           %[[ELSE_INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           } else {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           }
// CHECK:           scf.yield %[[ELSE_INNER]] : i32
// CHECK:         }
// CHECK:         %[[NEXT:.*]] = arith.addi %[[SELECTED]], %{{.*}} : i32
// CHECK:         %[[LOAD_PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[NEXT]]] {order = array<i32: 0>} : <tensor<16xf32>>
// CHECK:         %[[LOADED:.*]] = tt.load %[[LOAD_PTR]] : !tt.ptr<tensor<16xf32>>
// CHECK:         scf.yield %[[NEXT]], %[[LOADED]] : i32, tensor<16xf32>
// CHECK:       }
// CHECK:       tt.return %[[FOR]]#1 : tensor<16xf32>

// -----

module {
  tt.func public @for_nested_if_tensor_ptr_load_after_post_addptr(%base: !tt.ptr<f32>, %cond0: i1, %cond1: i1) -> tensor<4xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %off3 = tt.splat %c3_i32 : i32 -> tensor<4xi32>
    %off4 = tt.splat %c4_i32 : i32 -> tensor<4xi32>
    %empty_i32 = tt.splat %c0_i32 : i32 -> tensor<4xi32>
    %empty_f32 = arith.sitofp %empty_i32 : tensor<4xi32> to tensor<4xf32>
    %ptr0 = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %res:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0, %acc = %empty_f32) -> (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) {
      %selected = scf.if %cond0 -> (tensor<4x!tt.ptr<f32>>) {
        %a = tt.addptr %ptr, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %inner = scf.if %cond1 -> (tensor<4x!tt.ptr<f32>>) {
          %b = tt.addptr %a, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
          scf.yield %b : tensor<4x!tt.ptr<f32>>
        } else {
          %b = tt.addptr %a, %off4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
          scf.yield %b : tensor<4x!tt.ptr<f32>>
        }
        scf.yield %inner : tensor<4x!tt.ptr<f32>>
      } else {
        %a = tt.addptr %ptr, %off3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %inner = scf.if %cond1 -> (tensor<4x!tt.ptr<f32>>) {
          %b = tt.addptr %a, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
          scf.yield %b : tensor<4x!tt.ptr<f32>>
        } else {
          %b = tt.addptr %a, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
          scf.yield %b : tensor<4x!tt.ptr<f32>>
        }
        scf.yield %inner : tensor<4x!tt.ptr<f32>>
      }
      %next = tt.addptr %selected, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %loaded = tt.load %next : tensor<4x!tt.ptr<f32>>
      scf.yield %next, %loaded : tensor<4x!tt.ptr<f32>>, tensor<4xf32>
    }
    tt.return %res#1 : tensor<4xf32>
  }
}

// CHECK-LABEL: tt.func public @for_nested_if_tensor_ptr_load_after_post_addptr
// CHECK:       %[[FOR:.*]]:2 = scf.for
// CHECK-SAME:  i32
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           %[[THEN_INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           } else {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           }
// CHECK:           PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:      PointerDescriptorRebuild
// CHECK:           scf.yield %[[THEN_INNER]] : i32
// CHECK:         } else {
// CHECK:           %[[ELSE_INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           } else {
// CHECK:             scf.yield %{{.*}} : i32
// CHECK:           }
// CHECK:           PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:      PointerDescriptorRebuild
// CHECK:           scf.yield %[[ELSE_INNER]] : i32
// CHECK:         }
// CHECK:         arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:         tt.addptr %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:         %[[LOADED:.*]] = tt.load %{{.*}} : tensor<4x!tt.ptr<f32>>
// CHECK:         scf.yield %{{.*}}, %[[LOADED]] : i32, tensor<4xf32>
// CHECK:       }
// CHECK:       tt.return %[[FOR]]#1 : tensor<4xf32>

// -----

module {
  tt.func public @if_block_ptr_same_base_dynamic_shape(%base: !tt.ptr<f32>, %cond: i1, %shape0: i64, %shape1: i64) -> !tt.ptr<tensor<16xf32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %selected = scf.if %cond -> (!tt.ptr<tensor<16xf32>>) {
      %then_ptr = tt.make_tensor_ptr %base, [%shape0], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
      scf.yield %then_ptr : !tt.ptr<tensor<16xf32>>
    } else {
      %else_ptr = tt.make_tensor_ptr %base, [%shape1], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
      scf.yield %else_ptr : !tt.ptr<tensor<16xf32>>
    }
    tt.return %selected : !tt.ptr<tensor<16xf32>>
  }
}

// CHECK-LABEL: tt.func public @if_block_ptr_same_base_dynamic_shape(
// CHECK-SAME:  %[[BASE:[^ ,]+]]: !tt.ptr<f32>
// CHECK:       %[[BASE_ADDRESS:.*]] = tt.ptr_to_int %[[BASE]]
// CHECK:       %[[SHAPE:.*]] = scf.if %{{.*}} -> (i64) {
// CHECK:         scf.yield %{{.*}} : i64
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i64
// CHECK:       }
// CHECK-NOT:   tt.int_to_ptr %[[BASE_ADDRESS]]
// CHECK:       tt.make_tensor_ptr %[[BASE]], [%[[SHAPE]]], [%{{.*}}], [%{{.*}}] {PointerDescriptorRebuild, order = array<i32: 0>} : <tensor<16xf32>>

// -----

module {
  tt.func public @if_same_nested_result_fully_decoupled(%base: !tt.ptr<f32>, %cond0: i1, %cond1: i1) -> !tt.ptr<tensor<16xf32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %ptr = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %inner = scf.if %cond0 -> (!tt.ptr<tensor<16xf32>>) {
      %then_ptr = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %then_ptr : !tt.ptr<tensor<16xf32>>
    } else {
      %else_ptr = tt.advance %ptr, [%c2_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %else_ptr : !tt.ptr<tensor<16xf32>>
    }
    %outer = scf.if %cond1 -> (!tt.ptr<tensor<16xf32>>) {
      scf.yield %inner : !tt.ptr<tensor<16xf32>>
    } else {
      scf.yield %inner : !tt.ptr<tensor<16xf32>>
    }
    tt.return %outer : !tt.ptr<tensor<16xf32>>
  }
}

// CHECK-LABEL: tt.func public @if_same_nested_result_fully_decoupled
// CHECK:       %[[INNER_OFF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       scf.if %{{.*}} {
// CHECK-NEXT:  } else {
// CHECK-NEXT:  }
// CHECK:       %[[OUTER_PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[INNER_OFF]]] {PointerDescriptorRebuild, order = array<i32: 0>}
// CHECK:       tt.return %[[OUTER_PTR]] : !tt.ptr<tensor<16xf32>>

// -----

module {
  tt.func public @if_mixed_block_and_tensor_ptr(%block_base: !tt.ptr<f16>, %tensor_base: !tt.ptr<f32>, %cond: i1) -> (!tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %tensor_off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %tensor_off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %block_ptr = tt.make_tensor_ptr %block_base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
    %tensor_ptr = tt.splat %tensor_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %result:2 = scf.if %cond -> (!tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>) {
      %then_block = tt.advance %block_ptr, [%c1_i32] : !tt.ptr<tensor<16xf16>>
      %then_tensor = tt.addptr %tensor_ptr, %tensor_off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %then_block, %then_tensor : !tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>
    } else {
      %else_block = tt.advance %block_ptr, [%c2_i32] : !tt.ptr<tensor<16xf16>>
      %else_tensor = tt.addptr %tensor_ptr, %tensor_off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %else_block, %else_tensor : !tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>
    }
    tt.return %result#0, %result#1 : !tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @if_mixed_block_and_tensor_ptr
// CHECK:       %[[RESULT:.*]]:2 = scf.if %{{.*}} -> (i32, i32) {
// CHECK:         scf.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK:       }
// CHECK-DAG:   %[[BLOCK:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[RESULT]]#0] {PointerDescriptorRebuild, order = array<i32: 0>} : <tensor<16xf16>>
// CHECK-DAG:   %[[TENSOR:.*]] = tt.addptr %{{.*}}, %{{.*}} {{.*}}PointerDescriptorOffsetForm = "strided_1d"{{.*}}PointerDescriptorRebuild{{.*}} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:       tt.return %[[BLOCK]], %[[TENSOR]] : !tt.ptr<tensor<16xf16>>, tensor<4x!tt.ptr<f32>>

// -----

module {
  tt.func public @if_without_else_rewrites_nested_block_ptr(%base: !tt.ptr<f16>, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
    scf.if %cond {
      %final = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<16xf16>>) {
        %next = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<16xf16>>
        scf.yield %next : !tt.ptr<tensor<16xf16>>
      }
      %loaded = tt.load %final : !tt.ptr<tensor<16xf16>>
      scf.yield
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @if_without_else_rewrites_nested_block_ptr
// CHECK:       scf.if %{{.*}} {
// CHECK:         %[[FOR:.*]] = scf.for
// CHECK-SAME:    -> (i32) {
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         %[[PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}], [%{{.*}}], [%[[FOR]]] {{.*}}order = array<i32: 0>{{.*}} : <tensor<16xf16>>
// CHECK:         tt.load %[[PTR]] : !tt.ptr<tensor<16xf16>>
// CHECK:       }

// -----

module {
  tt.func public @sibling_if_result_initializes_for_tensor_ptr(%base: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %ptr0 = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      %then_ptr = tt.addptr %ptr0, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %then_ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %else_ptr = tt.addptr %ptr0, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %else_ptr : tensor<4x!tt.ptr<f32>>
    }
    %final = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %selected) -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    tt.return %final : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @sibling_if_result_initializes_for_tensor_ptr
// CHECK:       %[[IF_OFFSETS:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       PointerDescriptorOffsetForm = "strided_1d"
// CHECK-SAME:  PointerDescriptorRebuild
// CHECK:       %[[FOR_OFFSETS:.*]] = scf.for
// CHECK-SAME:  -> (i32) {
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       %[[FINAL:.*]] = tt.addptr %{{.*}}, %{{.*}} {{.*}}PointerDescriptorOffsetForm = "strided_1d"{{.*}}PointerDescriptorRebuild{{.*}} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:       tt.return %[[FINAL]] : tensor<4x!tt.ptr<f32>>
