// RUN: triton-opt %s -split-input-file -triton-tle-lower-wgmma | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_single_wgmma
  tt.func @lower_single_wgmma(
      %a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // CHECK: %[[ZERO:.+]] = arith.constant
    // CHECK-NEXT: %[[ACC:.+]] = ttg.convert_layout %[[ZERO]]
    // CHECK-NEXT: %[[DOT:.+]] = ttng.warp_group_dot %a, %b, %[[ACC]]
    // CHECK-NEXT: %[[WAIT:.+]] = ttng.warp_group_dot_wait %[[DOT]] {pendings = 0 : i32}
    // CHECK-NEXT: ttg.convert_layout %[[WAIT]]
    %dot = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_chained_wgmma
  tt.func @lower_chained_wgmma(
      %a0: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b0: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %a1: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b1: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // CHECK: %[[ZERO:.+]] = arith.constant
    // CHECK-NEXT: %[[ACC:.+]] = ttg.convert_layout %[[ZERO]]
    // CHECK-NEXT: %[[DOT0:.+]] = ttng.warp_group_dot %a0, %b0, %[[ACC]]
    // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot %a1, %b1, %[[DOT0]]
    // CHECK-NEXT: %[[WAIT:.+]] = ttng.warp_group_dot_wait %[[DOT1]] {pendings = 0 : i32}
    // CHECK-NEXT: ttg.convert_layout %[[WAIT]]
    %dot0 = tle.wgmma %a0, %b0, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %dot1 = tle.wgmma %a1, %b1, %dot0 {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %wait = tle.wgmma_wait %dot1 {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_register_a_wgmma
  tt.func @lower_register_a_wgmma(
      %a: tensor<64x16xf16, #blocked>,
      %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // CHECK: %[[ZERO:.+]] = arith.constant
    // CHECK-NEXT: %[[ACC:.+]] = ttg.convert_layout %[[ZERO]]
    // CHECK-NEXT: %[[A:.+]] = ttg.convert_layout %a
    // CHECK-NEXT: %[[DOT:.+]] = ttng.warp_group_dot %[[A]], %b, %[[ACC]]
    // CHECK-NEXT: %[[WAIT:.+]] = ttng.warp_group_dot_wait %[[DOT]] {pendings = 0 : i32}
    // CHECK-NEXT: ttg.convert_layout %[[WAIT]]
    %dot = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #blocked> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_loop_carried_wgmma
  tt.func @lower_loop_carried_wgmma(
      %a0: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b0: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %a1: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b1: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // CHECK: %[[ACC:.+]] = ttg.convert_layout %{{.+}}
    // CHECK-NEXT: %[[DOT0:.+]] = ttng.warp_group_dot %a0, %b0, %[[ACC]]
    // CHECK-NEXT: %[[LOOP:.+]] = scf.for {{.*}} iter_args({{.*}} = %[[DOT0]])
    // CHECK: %[[DOT1:.+]] = ttng.warp_group_dot %a1, %b1, %{{.+}}
    // CHECK-NEXT: %[[WAIT1:.+]] = ttng.warp_group_dot_wait %[[DOT1]] {pendings = 1 : i32}
    // CHECK-NOT: ttg.convert_layout
    // CHECK: scf.yield %[[WAIT1]]
    // CHECK: %[[WAIT0:.+]] = ttng.warp_group_dot_wait %[[LOOP]] {pendings = 0 : i32}
    // CHECK-NEXT: ttg.convert_layout %[[WAIT0]]
    %dot0 = tle.wgmma %a0, %b0, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %loop = scf.for %i = %lb to %ub step %step iter_args(%acc = %dot0) -> (tensor<64x16xf32, #blocked>) {
      %dot1 = tle.wgmma %a1, %b1, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
      %wait1 = tle.wgmma_wait %dot1 {pendings = 1 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
      scf.yield %wait1 : tensor<64x16xf32, #blocked>
    }
    %wait0 = tle.wgmma_wait %loop {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_wait1_ordinary_use
  tt.func @lower_wait1_ordinary_use(
      %a0: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b0: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %a1: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b1: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // CHECK: %[[ACC0:.+]] = ttg.convert_layout %{{.+}}
    // CHECK-NEXT: %[[DOT0:.+]] = ttng.warp_group_dot %a0, %b0, %[[ACC0]]
    // CHECK-NEXT: %[[ACC1:.+]] = ttg.convert_layout %{{.+}}
    // CHECK-NEXT: %{{.+}} = ttng.warp_group_dot %a1, %b1, %[[ACC1]]
    // CHECK-NEXT: %[[WAIT:.+]] = ttng.warp_group_dot_wait %[[DOT0]] {pendings = 1 : i32}
    // CHECK-NEXT: %[[RELEASED:.+]] = ttg.convert_layout %[[WAIT]]
    // CHECK-NEXT: "tt.reduce"(%[[RELEASED]])
    %dot0 = tle.wgmma %a0, %b0, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %dot1 = tle.wgmma %a1, %b1, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %wait = tle.wgmma_wait %dot0 {pendings = 1 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %red = "tt.reduce"(%wait) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %wait1 = tle.wgmma_wait %dot1 {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}
