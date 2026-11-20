// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s

// Layout rematerialization accepts dead sources as well as single-use and
// shared sources. Iterating drop_begin on an empty use list is invalid.
#layout = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module {
  // CHECK-LABEL: tt.func public @dead_layout_sources
  tt.func public @dead_layout_sources(%input: !tt.ptr<f32>) {
    %dead_constant = arith.constant dense<0.0> : tensor<1024xf32>
    %dead_range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %dead_splat = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %encoded = tle.gpu.set_layout %dead_constant {target_encoding = #layout} : tensor<1024xf32> -> tensor<1024xf32>
    // CHECK: tt.return
    tt.return
  }
}
