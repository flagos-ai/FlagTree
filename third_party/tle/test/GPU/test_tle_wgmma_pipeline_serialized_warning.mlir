// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s

// A dot cannot stay properly async only because its uses are after a wait in
// the next loop iteration. That would leave the WGMMA pipeline stage open
// across the loop backedge, which ptxas serializes.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: async_blocked_across_backedge_wait
  tt.func @async_blocked_across_backedge_wait(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>) {
    %cst = arith.constant dense<64> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
    // CHECK:          %[[LOOP:[^ :]+]]{{.*}} scf.for {{.*}} iter_args(%[[PREV_DOT2:[^ ]+]]
    // CHECK:            %[[DOT0:.+]] = ttng.warp_group_dot
    // CHECK-NOT:        ttng.warp_group_dot_wait
    // CHECK:            %[[DOT1:.+]] = ttng.warp_group_dot
    // CHECK-NEXT:       ttng.warp_group_dot_wait %[[DOT1]]
    // CHECK-SAME:         {pendings = 0 : i32}
    // CHECK:            %[[DOT2:.+]] = ttng.warp_group_dot
    // CHECK-NEXT:       %[[WAIT2:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT2]]
    // CHECK-SAME:         {pendings = 0 : i32}
    // CHECK:          scf.yield %[[WAIT2]]#0
    // CHECK:          ttng.warp_group_dot_wait %[[LOOP]]#3 {pendings = 0 : i32}
    %17:4 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%prev_dot2 = %cst_3, %arg5 = %16, %prev_dot1 = %cst_2, %prev_dot0 = %cst_2) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>)  : i32 {
      %dot0 = ttng.warp_group_dot %19, %20, %prev_dot1 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %dot1 = ttng.warp_group_dot %19, %20, %prev_dot1 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %dot1.1 = arith.addf %dot1, %dot1 : tensor<128x16xf32, #mma1>
      %l = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %c = ttg.local_alloc %l : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %23 = ttg.memdesc_trans %c {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared1, #smem> -> !ttg.memdesc<16x64xf16, #shared, #smem>
      %prev_dot2.1 = arith.addf %prev_dot2, %prev_dot2 : tensor<128x64xf32, #mma>
      %dot2 = ttng.warp_group_dot %cst_4, %23, %prev_dot2.1 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>> * !ttg.memdesc<16x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %dot2, %26, %dot1.1, %dot0 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>
    }
    tt.return %17#0, %17#2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  }
}
