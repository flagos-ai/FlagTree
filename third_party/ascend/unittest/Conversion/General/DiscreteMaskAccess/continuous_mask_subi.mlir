// RUN: triton-opt --discrete-mask-access-conversion --split-input-file %s | FileCheck %s

// Continuous head mask written as (arange - C) >= 0 must be recognized by
// MaskState (via arith.subi). discrete-mask-access-conversion must NOT strip
// the mask into a full load + select.

// CHECK-LABEL: tt.func @continuous_head_mask_subi
// CHECK-NOT: arith.select
// CHECK: tt.load {{.*}}, %[[MASK:.*]]
// CHECK-NOT: arith.select
tt.func @continuous_head_mask_subi(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  %c0 = arith.constant dense<0> : tensor<256xi32>
  %c2 = arith.constant dense<2> : tensor<256xi32>
  %other = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %idx = arith.subi %offs, %c2 : tensor<256xi32>
  %mask = arith.cmpi sge, %idx, %c0 : tensor<256xi32>
  %x_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %x_ptrs = tt.addptr %x_splat, %idx : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %o_splat = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %o_ptrs = tt.addptr %o_splat, %offs : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %v = tt.load %x_ptrs, %mask, %other : tensor<256x!tt.ptr<f32>>
  tt.store %o_ptrs, %v : tensor<256x!tt.ptr<f32>>
  tt.return
}
