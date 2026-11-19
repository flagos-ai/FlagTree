// RUN: not triton-opt --discrete-mask-access-conversion --triton-to-unstructure %s 2>&1 | FileCheck %s
// RUN: not triton-opt '--discrete-mask-access-conversion=compile-mode=simd_simt_template' '--triton-to-unstructure=compile-mode=simd_simt_template' %s 2>&1 | FileCheck %s

// A changing non-prefix mask cannot use SIMD's unguarded full-load/select.
// CHECK: error: cannot lower this loop-carried mask with a masked indirect access for the selected target and compile mode
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @unsupported_loop_mask(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %steps: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<true> : tensor<16xi1>
    %zeros = arith.constant dense<0.0> : tensor<16xf32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lane : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%mask = %init, %acc = %zeros) -> (tensor<16xi1>, tensor<16xf32>) : i32 {
      %bound = tt.splat %i : i32 -> tensor<16xi32>
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
