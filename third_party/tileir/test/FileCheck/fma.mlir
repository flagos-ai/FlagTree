// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts)" | FileCheck %s

module {
  tt.func public @fused_multiply_add(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %x = tt.load %a : !tt.ptr<f32>
    %y = tt.load %b : !tt.ptr<f32>
    %z = tt.load %c : !tt.ptr<f32>
    %product = arith.mulf %x, %y : f32
    %result = arith.addf %product, %z : f32
    tt.store %out, %result : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @fused_multiply_add
// CHECK: %[[RESULT:.*]] = fma
// CHECK-NOT: mulf
// CHECK: store_ptr_tko {{.*}}%[[RESULT]]
