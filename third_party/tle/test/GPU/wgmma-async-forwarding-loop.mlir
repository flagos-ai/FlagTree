// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -triton-tle-lower-wgmma | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Carrying edges can enter and leave nested loops without ordinary reads.
  // A loop that only forwards an accumulator must inherit its native type.
  // CHECK-LABEL: tt.func @nested_forwarding_carry
  // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
  // CHECK: %[[OUTER:.+]] = scf.for {{.*}} iter_args({{.*}} = %[[DOT]])
  // CHECK: %[[INNER:.+]] = scf.for
  // CHECK-NOT: ttng.warp_group_dot_wait
  // CHECK: scf.yield
  // CHECK: scf.yield %[[INNER]]
  // CHECK: ttng.warp_group_dot_wait %[[OUTER]] {pendings = 0 : i32}
  // CHECK-NOT: tle.wgmma
  // CHECK: tt.return
  tt.func @nested_forwarding_carry(%a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : tensor<64x16xf32, #blocked>
    %initial = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %outer = scf.for %i = %lb to %ub step %step iter_args(%acc = %initial) -> (tensor<64x16xf32, #blocked>) {
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_acc = %acc) -> (tensor<64x16xf32, #blocked>) {
        scf.yield %inner_acc : tensor<64x16xf32, #blocked>
      }
      scf.yield %inner : tensor<64x16xf32, #blocked>
    }
    %ready = tle.wgmma_wait %outer {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}
