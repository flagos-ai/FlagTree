// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.experimental\$func(fuse-fma)),reconcile-unrealized-casts, inline)" | FileCheck %s

// CHECK-LABEL: kernel_12
module @kernel_12 {
  // CHECK-NOT: @kernel
  tt.func private @kernel(%arg0: i32) {
    tt.return
  }

  // CHECK-LABEL: @main
  tt.func @main(%arg0: i32) {
    // CHECK-NOT: call
    tt.call @kernel(%arg0) : (i32) -> ()
    tt.return
  }
}
