// RUN: triton-opt --discrete-mask-access-conversion --split-input-file --verify-each %s | FileCheck %s --check-prefix=PREFIX
// RUN: triton-opt --triton-control-flow-opt --discrete-mask-access-conversion --triton-to-unstructure --triton-to-linalg --split-input-file --verify-each %s | FileCheck %s --check-prefix=LINALG --implicit-check-not=unrealized_conversion_cast --implicit-check-not=indirect_load

// Growing limits must not reopen lanes closed by a preceding iteration.
// PREFIX-LABEL: tt.func public @prefix_history
// PREFIX: scf.for {{.*}} iter_args(%[[BOUND:.*]] = %{{.*}}, {{.*}}) -> (i32, tensor<16xf32>)
// PREFIX: %[[LIMIT:.*]] = arith.select
// PREFIX: %[[NEXT:.*]] = arith.minsi %[[BOUND]], %[[LIMIT]] : i32
// PREFIX: %[[SPLAT:.*]] = tt.splat %[[NEXT]] : i32 -> tensor<16xi32>
// PREFIX: %[[MASK:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT]] : tensor<16xi32>
// PREFIX: tt.load %{{.*}}, %[[MASK]],
// PREFIX: scf.yield %[[NEXT]], {{.*}} : i32, tensor<16xf32>
// LINALG-LABEL: func.func @prefix_history
// LINALG: scf.for {{.*}} iter_args(%[[BOUND:.*]] = %{{.*}}, {{.*}}) -> (i32, tensor<16xf32>)
// LINALG: %[[NEXT:.*]] = arith.minsi %[[BOUND]], %{{.*}} : i32
// LINALG: %[[INDEX:.*]] = arith.index_cast %[[NEXT]] : i32 to index
// LINALG: %[[NONNEG:.*]] = arith.maxsi %[[INDEX]], %{{.*}} : index
// LINALG: %[[LENGTH:.*]] = arith.minsi %[[NONNEG]], %{{.*}} : index
// LINALG: memref.subview {{.*}}[0] [%[[LENGTH]]] [1]
// LINALG: memref.copy
// LINALG: scf.yield %[[NEXT]], {{.*}} : i32, tensor<16xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @prefix_history(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                                %steps: i32, %initial: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c12 = arith.constant 12 : i32
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %init_bound = tt.splat %initial : i32 -> tensor<16xi32>
    %init = arith.cmpi slt, %lane, %init_bound : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%mask = %init, %acc = %zeros) -> (tensor<16xi1>, tensor<16xf32>) : i32 {
      %odd = arith.andi %i, %c1 : i32
      %even = arith.cmpi eq, %odd, %c0 : i32
      %limit = arith.select %even, %c4, %c12 : i32
      %bound = tt.splat %limit : i32 -> tensor<16xi32>
      %current = arith.cmpi slt, %lane, %bound : tensor<16xi32>
      %next = arith.andi %current, %mask : tensor<16xi1>
      %value = tt.load %ptrs, %next, %zeros : tensor<16x!tt.ptr<f32>>
      %sum = arith.addf %acc, %value : tensor<16xf32>
      scf.yield %next, %sum : tensor<16xi1>, tensor<16xf32>
    }
    %output = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %output_ptrs, %result#1 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// An invariant mask can still use its initial continuous range even when a
// while permutes the before/after slots.
// PREFIX-LABEL: tt.func public @invariant_while_mask
// PREFIX-NOT: RuntimeLoopMask
// PREFIX: tt.load
// PREFIX-NOT: RuntimeLoopMask
// PREFIX: tt.return
// LINALG-LABEL: func.func @invariant_while_mask
// LINALG: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @invariant_while_mask(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %steps: i32, %limit: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %bound = tt.splat %limit : i32 -> tensor<16xi32>
    %init = arith.cmpi slt, %lane, %bound : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:3 = scf.while (%i = %c0, %mask = %init, %acc = %zeros) : (i32, tensor<16xi1>, tensor<16xf32>) -> (tensor<16xi1>, i32, tensor<16xf32>) {
      %keep = arith.cmpi slt, %i, %steps : i32
      scf.condition(%keep) %mask, %i, %acc : tensor<16xi1>, i32, tensor<16xf32>
    } do {
    ^bb0(%mask: tensor<16xi1>, %i: i32, %acc: tensor<16xf32>):
      %value = tt.load %ptrs, %mask, %zeros : tensor<16x!tt.ptr<f32>>
      %sum = arith.addf %acc, %value : tensor<16xf32>
      %inc = arith.addi %i, %c1 : i32
      scf.yield %inc, %mask, %sum : i32, tensor<16xi1>, tensor<16xf32>
    }
    %output = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %output_ptrs, %result#2 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Keep pre-update users and the loop result tied to their own boundaries.
// This also exercises an all-true initial mask and zero-trip result semantics.
// PREFIX-LABEL: tt.func public @prefix_before_and_after
// PREFIX: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[BOUND:.*]] = %{{.*}}, {{.*}}) -> (i32, tensor<16xf32>)
// PREFIX: %[[OLD_SPLAT:.*]] = tt.splat %[[BOUND]]
// PREFIX: %[[OLD_MASK:.*]] = arith.cmpi slt, %{{.*}}, %[[OLD_SPLAT]]
// PREFIX: tt.load %{{.*}}, %[[OLD_MASK]],
// PREFIX: %[[NEXT:.*]] = arith.minsi %[[BOUND]], %{{.*}} : i32
// PREFIX: scf.yield %[[NEXT]], {{.*}} : i32, tensor<16xf32>
// PREFIX: %[[FINAL_BOUND:.*]] = tt.splat %[[LOOP]]#0
// PREFIX: %[[FINAL_MASK:.*]] = arith.cmpi slt, %{{.*}}, %[[FINAL_BOUND]]
// PREFIX: arith.extui %[[FINAL_MASK]] : tensor<16xi1> to tensor<16xi32>
// LINALG-LABEL: func.func @prefix_before_and_after
// LINALG: scf.for {{.*}} -> (i32, tensor<16xf32>)
// LINALG: memref.copy
// LINALG: arith.minsi {{.*}} : i32
// LINALG: scf.yield {{.*}} : i32, tensor<16xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @prefix_before_and_after(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                                         %mask_out: !tt.ptr<i32>, %steps: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c15 = arith.constant 15 : i32
    %init = arith.constant dense<true> : tensor<16xi1>
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%mask = %init, %acc = %zeros) -> (tensor<16xi1>, tensor<16xf32>) : i32 {
      %value = tt.load %ptrs, %mask, %zeros : tensor<16x!tt.ptr<f32>>
      %sum = arith.addf %acc, %value : tensor<16xf32>
      %limit = arith.subi %c15, %i : i32
      %bound = tt.splat %limit : i32 -> tensor<16xi32>
      %current = arith.cmpi slt, %lane, %bound : tensor<16xi32>
      %next = arith.andi %mask, %current : tensor<16xi1>
      scf.yield %next, %sum : tensor<16xi1>, tensor<16xf32>
    }
    %output = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %output_ptrs, %result#1 : tensor<16x!tt.ptr<f32>>
    %mask_base = tt.splat %mask_out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %mask_ptrs = tt.addptr %mask_base, %lane : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %final = arith.extui %result#0 : tensor<16xi1> to tensor<16xi32>
    tt.store %mask_ptrs, %final : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}
