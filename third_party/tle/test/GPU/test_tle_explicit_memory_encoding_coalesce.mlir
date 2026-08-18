// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [1, 0]}>
#store_layout = #ttg.slice<{dim = 0, parent = #parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @explicit_store_memory_encoding
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.store
  // CHECK-SAME: tle.explicit_memory_encoding = #ttg.slice
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func @explicit_store_memory_encoding(%base: !tt.ptr<f32>) {
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #store_layout>
    %values = arith.constant dense<0.000000e+00> : tensor<64xf32, #store_layout>
    tt.store %ptrs, %values {tle.explicit_memory_encoding = #store_layout} : tensor<64x!tt.ptr<f32>, #store_layout>
    tt.return
  }
}
