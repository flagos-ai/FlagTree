// RUN: triton-opt %s -triton-tle-select-encodings | FileCheck %s

// Atomics share SelectEncodings' pointer-folding helper. Test its contract
// independently of RLC, which has its own atomic-result layout selection.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @explicit_atomic_rmw
  tt.func @explicit_atomic_rmw(%pointer: tensor<2x64x!tt.ptr<i32>, #blocked>, %value: tensor<2x64xi32, #mma>, %mask: tensor<2x64xi1, #mma>) {
    // CHECK: %[[PTR:.*]] = ttg.convert_layout %arg0 {{.*}} -> tensor<2x64x!tt.ptr<i32>, #mma>
    %ptr = ttg.convert_layout %pointer : tensor<2x64x!tt.ptr<i32>, #blocked> -> tensor<2x64x!tt.ptr<i32>, #mma>
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw add, relaxed, cta, %[[PTR]], %arg1, %arg2 {tle.explicit_memory_encoding = #mma}
    %old = tt.atomic_rmw add, relaxed, cta, %ptr, %value, %mask {tle.explicit_memory_encoding = #mma} : (tensor<2x64x!tt.ptr<i32>, #mma>, tensor<2x64xi32, #mma>, tensor<2x64xi1, #mma>) -> tensor<2x64xi32, #mma>
    // CHECK-NOT: tt.atomic_rmw
    tt.return
  }

  // CHECK-LABEL: tt.func @explicit_atomic_cas
  tt.func @explicit_atomic_cas(%pointer: tensor<2x64x!tt.ptr<i32>, #blocked>, %compare: tensor<2x64xi32, #mma>, %value: tensor<2x64xi32, #mma>) {
    // CHECK: %[[PTR:.*]] = ttg.convert_layout %arg0 {{.*}} -> tensor<2x64x!tt.ptr<i32>, #mma>
    %ptr = ttg.convert_layout %pointer : tensor<2x64x!tt.ptr<i32>, #blocked> -> tensor<2x64x!tt.ptr<i32>, #mma>
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas acq_rel, cta, %[[PTR]], %arg1, %arg2 {tle.explicit_memory_encoding = #mma}
    %old = tt.atomic_cas acq_rel, cta, %ptr, %compare, %value {tle.explicit_memory_encoding = #mma} : (tensor<2x64x!tt.ptr<i32>, #mma>, tensor<2x64xi32, #mma>, tensor<2x64xi32, #mma>) -> tensor<2x64xi32, #mma>
    // CHECK-NOT: tt.atomic_cas
    tt.return
  }
}
