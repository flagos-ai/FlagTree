// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -split-input-file -triton-tle-lower-wgmma | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Carry the newly issued product while wait(1) releases the older product.
  // CHECK-LABEL: tt.func @two_independent_products
  // CHECK: %[[INITIAL:.+]] = ttng.warp_group_dot
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[INITIAL]],
  // CHECK: %[[NEXT:.+]] = ttng.warp_group_dot
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32}
  // CHECK: arith.addf
  // CHECK: scf.yield %[[NEXT]],
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NOT: tle.wgmma
  // CHECK: tt.return
  tt.func @two_independent_products(%a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The initial value is ordinary; the backedge carries an async accumulator.
  // CHECK-LABEL: tt.func @ordinary_initial_value
  // CHECK: ttg.convert_layout
  // CHECK: scf.for
  // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
  // CHECK-NOT: ttng.warp_group_dot_wait
  // CHECK: scf.yield %[[DOT]]
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK: tt.return
  tt.func @ordinary_initial_value(%a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : tensor<64x16xf32, #blocked>
    %loop = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (tensor<64x16xf32, #blocked>) {
      %dot = tle.wgmma %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
      scf.yield %dot : tensor<64x16xf32, #blocked>
    }
    %ready = tle.wgmma_wait %loop {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Carrying edges can enter and leave nested loops without ordinary reads.
  // CHECK-LABEL: tt.func @nested_carry
  // CHECK: ttng.warp_group_dot
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: ttng.warp_group_dot
  // CHECK: scf.yield
  // CHECK: scf.yield
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK: tt.return
  tt.func @nested_carry(%a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : tensor<64x16xf32, #blocked>
    %initial = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %outer = scf.for %i = %lb to %ub step %step iter_args(%acc = %initial) -> (tensor<64x16xf32, #blocked>) {
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_acc = %acc) -> (tensor<64x16xf32, #blocked>) {
        %dot = tle.wgmma %a, %b, %inner_acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
        scf.yield %dot : tensor<64x16xf32, #blocked>
      }
      scf.yield %inner : tensor<64x16xf32, #blocked>
    }
    %ready = tle.wgmma_wait %outer {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}
