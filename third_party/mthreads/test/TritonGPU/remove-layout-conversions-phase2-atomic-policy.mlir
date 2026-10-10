// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=ALLOW,PRESERVE

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // ALLOW-LABEL: tt.func @atomic_tail_allowed
  // ALLOW-NOT: ttg.convert_layout
  // ALLOW: tt.atomic_rmw
  tt.func @atomic_tail_allowed(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-atomic-writeback-max-elements-per-thread-ratio" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // PRESERVE-LABEL: tt.func @atomic_tail_policy_preserves
  // PRESERVE: ttg.convert_layout
  // PRESERVE: tt.atomic_rmw
  tt.func @atomic_tail_policy_preserves(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}
