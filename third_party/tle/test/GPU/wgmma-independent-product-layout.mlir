// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -split-input-file -triton-tle-lower-wgmma | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Carry the newly issued product while wait(1) releases the older product.
  // CHECK-LABEL: tt.func @two_independent_products_eight_warps
  // CHECK: %[[INITIAL:.+]] = ttng.warp_group_dot
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[INITIAL]],
  // CHECK: %[[NEXT:.+]] = ttng.warp_group_dot
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32}
  // CHECK: arith.addf
  // CHECK: scf.yield %[[NEXT]],
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NOT: tle.wgmma
  // CHECK: tt.return
  tt.func @two_independent_products_eight_warps(%a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 31 : index
    %step = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : tensor<64x16xf32, #blocked>
    %initial = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %loop:2 = scf.for %i = %lb to %ub step %step iter_args(%previous = %initial, %sum = %zero) -> (tensor<64x16xf32, #blocked>, tensor<64x16xf32, #blocked>) {
      %next = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
      %ready = tle.wgmma_wait %previous {pendings = 1 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
      %sum_next = arith.addf %ready, %sum : tensor<64x16xf32, #blocked>
      scf.yield %next, %sum_next : tensor<64x16xf32, #blocked>, tensor<64x16xf32, #blocked>
    }
    %ready = tle.wgmma_wait %loop#0 {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}
