// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file --verify-each %s | FileCheck %s --check-prefix=MASK
// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=true compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=true compile-mode=simd_simt_template' '--triton-to-linalg=compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file --verify-each %s | FileCheck %s --check-prefix=LOWER --implicit-check-not=unrealized_conversion_cast --implicit-check-not=RuntimeLoopMask

// A suffix is outside the prefix rewrite, but is still an analyzable range.
// Replacing the iter_arg by its init would incorrectly forget earlier bounds.
// MASK-LABEL: tt.func public @loop_suffix
// MASK: scf.for {{.*}} -> (tensor<16xi1>, tensor<16xf32>)
// MASK: %[[NEXT:.*]] = arith.andi
// MASK: tt.load %{{.*}}, %[[NEXT]], %{{.*}} {MixCompileDiscreteMask, RuntimeLoopMask}
// LOWER-LABEL: func.func @loop_suffix
// LOWER: scf.for {{.*}} -> (tensor<16xi1>, tensor<16xf32>)
// LOWER: %[[NEXT:.*]] = arith.andi {{.*}} : tensor<16xi1>
// LOWER: func.call @triton_indirect_load({{.*}}%[[NEXT]],
// LOWER: scf.yield %[[NEXT]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @loop_suffix(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %steps: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c12 = arith.constant 12 : i32
    %init = arith.constant dense<true> : tensor<16xi1>
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%mask = %init, %acc = %zeros) -> (tensor<16xi1>, tensor<16xf32>) : i32 {
      %limit = arith.subi %c12, %i : i32
      %bound = tt.splat %limit : i32 -> tensor<16xi32>
      %current = arith.cmpi sge, %lane, %bound : tensor<16xi32>
      %next = arith.andi %mask, %current : tensor<16xi1>
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

// While before/after slots have different order. A changing mask must not be
// considered invariant just because the condition forwards it unchanged.
// MASK-LABEL: tt.func public @while_permuted_mask
// MASK: scf.while
// MASK: %[[NEXT:.*]] = arith.andi
// MASK: tt.load %{{.*}}, %[[NEXT]], %{{.*}} {MixCompileDiscreteMask, RuntimeLoopMask}
// LOWER-LABEL: func.func @while_permuted_mask
// LOWER: scf.while
// LOWER: %[[NEXT:.*]] = arith.andi {{.*}} : tensor<16xi1>
// LOWER: func.call @triton_indirect_load({{.*}}%[[NEXT]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @while_permuted_mask(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %steps: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c12 = arith.constant 12 : i32
    %init = arith.constant dense<true> : tensor<16xi1>
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:3 = scf.while (%i = %c0, %mask = %init, %acc = %zeros) : (i32, tensor<16xi1>, tensor<16xf32>) -> (tensor<16xi1>, i32, tensor<16xf32>) {
      %keep = arith.cmpi slt, %i, %steps : i32
      scf.condition(%keep) %mask, %i, %acc : tensor<16xi1>, i32, tensor<16xf32>
    } do {
    ^bb0(%mask: tensor<16xi1>, %i: i32, %acc: tensor<16xf32>):
      %limit = arith.subi %c12, %i : i32
      %bound = tt.splat %limit : i32 -> tensor<16xi32>
      %current = arith.cmpi sge, %lane, %bound : tensor<16xi32>
      %next = arith.andi %mask, %current : tensor<16xi1>
      %value = tt.load %ptrs, %next, %zeros : tensor<16x!tt.ptr<f32>>
      %sum = arith.addf %acc, %value : tensor<16xf32>
      %inc = arith.addi %i, %c1 : i32
      scf.yield %inc, %next, %sum : i32, tensor<16xi1>, tensor<16xf32>
    }
    %output = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %output_ptrs, %result#2 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
