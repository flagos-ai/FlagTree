// RUN: triton-opt %s -triton-tle-select-encodings | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions -triton-tle-select-encodings -tritongpu-remove-layout-conversions | FileCheck %s

// Pointer-convert folding is only an optimization preference. It must not change
// an explicitly selected load layout, even when it prefers blocked over dot.
// Otherwise the preserved explicit-memory attribute contradicts the result
// type and a subsequent dot needs shared-memory staging to undo the change.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @explicit_dot_load
  tt.func @explicit_dot_load(%pointer: tensor<2x128x!tt.ptr<bf16>, #blocked>, %other: tensor<128x64xbf16, #rhs>, %output: tensor<2x64x!tt.ptr<f32>, #mma>) {
    %ptr = ttg.convert_layout %pointer : tensor<2x128x!tt.ptr<bf16>, #blocked> -> tensor<2x128x!tt.ptr<bf16>, #lhs>
    // CHECK: %[[VALUE:.*]] = tt.load {{.*}}tle.explicit_memory_encoding = #ttg.dot_op<{{.*}}opIdx = 0{{.*}} : tensor<2x128x!tt.ptr<bf16>, #ttg.dot_op<{{.*}}opIdx = 0{{.*}}>>
    %value = tt.load %ptr {isVolatile = true, tle.explicit_memory_encoding = #lhs} : tensor<2x128x!tt.ptr<bf16>, #lhs>
    %zero = arith.constant dense<0.0> : tensor<2x64xf32, #mma>
    // CHECK-NOT: ttg.convert_layout %[[VALUE]]
    // CHECK-NOT: tt.load
    // CHECK: tt.dot %[[VALUE]],
    %dot = tt.dot %value, %other, %zero : tensor<2x128xbf16, #lhs> * tensor<128x64xbf16, #rhs> -> tensor<2x64xf32, #mma>
    tt.store %output, %dot : tensor<2x64x!tt.ptr<f32>, #mma>
    tt.return
  }

  // An ordinary access may still absorb the pointer conversion. The fix is
  // a legality condition, not removal of this optimization.
  // CHECK-LABEL: tt.func @ordinary_dot_load
  tt.func @ordinary_dot_load(%pointer: tensor<2x128x!tt.ptr<bf16>, #blocked>, %other: tensor<128x64xbf16, #rhs>, %output: tensor<2x64x!tt.ptr<f32>, #mma>) {
    %ptr = ttg.convert_layout %pointer : tensor<2x128x!tt.ptr<bf16>, #blocked> -> tensor<2x128x!tt.ptr<bf16>, #lhs>
    // CHECK: %[[ORDINARY:.*]] = tt.load %arg0 {{.*}} : tensor<2x128x!tt.ptr<bf16>, #blocked>
    %value = tt.load %ptr {isVolatile = true} : tensor<2x128x!tt.ptr<bf16>, #lhs>
    %zero = arith.constant dense<0.0> : tensor<2x64xf32, #mma>
    // CHECK: %[[BRIDGE:.*]] = ttg.convert_layout %[[ORDINARY]]
    // CHECK-NOT: tt.load
    // CHECK: tt.dot %[[BRIDGE]],
    %dot = tt.dot %value, %other, %zero : tensor<2x128xbf16, #lhs> * tensor<128x64xbf16, #rhs> -> tensor<2x64xf32, #mma>
    tt.store %output, %dot : tensor<2x64x!tt.ptr<f32>, #mma>
    tt.return
  }

  // CHECK-LABEL: tt.func @explicit_store
  tt.func @explicit_store(%pointer: tensor<2x64x!tt.ptr<f32>, #blocked>, %value: tensor<2x64xf32, #mma>) {
    // CHECK: %[[STORE_PTR:.*]] = ttg.convert_layout %arg0 {{.*}} -> tensor<2x64x!tt.ptr<f32>, #mma>
    %ptr = ttg.convert_layout %pointer : tensor<2x64x!tt.ptr<f32>, #blocked> -> tensor<2x64x!tt.ptr<f32>, #mma>
    // CHECK-NEXT: tt.store %[[STORE_PTR]], %arg1 {tle.explicit_memory_encoding = #mma}
    tt.store %ptr, %value {tle.explicit_memory_encoding = #mma} : tensor<2x64x!tt.ptr<f32>, #mma>
    // CHECK-NOT: tt.store
    tt.return
  }

  // Validate every user before mutating any one access. A shared conversion
  // with an explicitly constrained user must not be partially rewritten.
  // CHECK-LABEL: tt.func @mixed_users
  tt.func @mixed_users(%pointer: tensor<2x64x!tt.ptr<f32>, #blocked>, %value: tensor<2x64xf32, #mma>) {
    // CHECK: %[[MIXED_PTR:.*]] = ttg.convert_layout %arg0 {{.*}} -> tensor<2x64x!tt.ptr<f32>, #mma>
    %ptr = ttg.convert_layout %pointer : tensor<2x64x!tt.ptr<f32>, #blocked> -> tensor<2x64x!tt.ptr<f32>, #mma>
    // CHECK-NEXT: tt.store %[[MIXED_PTR]], %arg1
    tt.store %ptr, %value : tensor<2x64x!tt.ptr<f32>, #mma>
    // CHECK-NEXT: tt.store %[[MIXED_PTR]], %arg1 {tle.explicit_memory_encoding = #mma}
    tt.store %ptr, %value {tle.explicit_memory_encoding = #mma} : tensor<2x64x!tt.ptr<f32>, #mma>
    // CHECK-NOT: tt.store
    tt.return
  }

}
