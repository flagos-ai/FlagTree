// RUN: triton-opt %s --verify-diagnostics
// RUN: triton-opt %s --tritongpu-allocate-warp-groups --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-warp-specialize-to-llvm -reconcile-unrealized-casts | FileCheck %s

// The dynamic parent materializes a complete CTA tensor through shared memory.
// Its two static children can then start new register-local owner views, just
// as paired KDA results are split before reusing the owning default warps.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @materialized_parent
  // CHECK: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: nvvm.barrier0
  // CHECK: llvm.load {{.*}} !llvm.ptr<3>
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: llvm.inline_asm {{.*}}st.global
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: llvm.return
  tt.func @materialized_parent(%src: tensor<256xi32, #full>, %index: i32, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %whole = tle.extract_tile %src[%index] {tile_shape = array<i64: 128>} : tensor<256xi32, #full>, i32 -> tensor<128xi32, #full>
    %top = tle.extract_tile %whole[%c0] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    %bottom = tle.extract_tile %whole[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    %second = tt.addptr %ptr, %c64 : !tt.ptr<i32>, i32
    ttg.warp_specialize(%top, %bottom, %ptr, %second) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%a: tensor<64xi32, #half>, %b: tensor<64xi32, #half>, %p: !tt.ptr<i32>, %q: !tt.ptr<i32>) num_warps(2) {
      %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #half>
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #half>
      %pd = tt.addptr %ps, %r : tensor<64x!tt.ptr<i32>, #half>, tensor<64xi32, #half>
      tt.store %pd, %a : tensor<64x!tt.ptr<i32>, #half>
      ttg.warp_return
    }
    partition1(%a: tensor<64xi32, #half>, %b: tensor<64xi32, #half>, %p: !tt.ptr<i32>, %q: !tt.ptr<i32>) num_warps(2) {
      %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #half>
      %ps = tt.splat %q : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #half>
      %pd = tt.addptr %ps, %r : tensor<64x!tt.ptr<i32>, #half>, tensor<64xi32, #half>
      tt.store %pd, %b : tensor<64x!tt.ptr<i32>, #half>
      ttg.warp_return
    }
    : (tensor<64xi32, #half>, tensor<64xi32, #half>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    tt.return
  }
}
