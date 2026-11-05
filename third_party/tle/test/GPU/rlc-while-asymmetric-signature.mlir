// RUN: triton-opt %s -tritongpu-remove-layout-conversions -verify-each | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#matrix = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The polling value is needed by the before region only. The after region
  // takes no arguments and the while has no results, but its yield still feeds
  // the before argument. It must not be tied to a nonexistent while result.
  // CHECK-LABEL: tt.func @poll_without_results
  // CHECK: scf.while {{.*}} : (tensor<4xi64, {{.*}}>) -> ()
  // CHECK: scf.condition({{%.*}})
  // CHECK: tt.atomic_rmw add, acquire, sys
  // CHECK: scf.yield {{.*}} : tensor<4xi64,
  // CHECK: tt.return
  tt.func @poll_without_results(%flags: tensor<4x!tt.ptr<i64>, #blocked>, %initial: tensor<4xi64, #blocked>, %epoch: i64) {
    %zero = arith.constant dense<0> : tensor<4xi64, #blocked>
    %mask = arith.constant dense<true> : tensor<4xi1, #blocked>
    scf.while (%seen = %initial) : (tensor<4xi64, #blocked>) -> () {
      %minimum = "tt.reduce"(%seen) <{axis = 0 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %min = arith.minui %lhs, %rhs : i64
        tt.reduce.return %min : i64
      }) : (tensor<4xi64, #blocked>) -> i64
      %pending = arith.cmpi ult, %minimum, %epoch : i64
      scf.condition(%pending)
    } do {
      %next = tt.atomic_rmw add, acquire, sys, %flags, %zero, %mask : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi1, #blocked>) -> tensor<4xi64, #blocked>
      scf.yield %next : tensor<4xi64, #blocked>
    }
    tt.return
  }

  // Even with one input and one result, the two signatures need not have the
  // same rank. Yield targets the flat before argument, while condition forwards
  // the matrix to the after argument and result.
  // CHECK-LABEL: tt.func @different_region_ranks
  // CHECK: scf.while {{.*}} : (tensor<4xi64, {{.*}}>) -> tensor<2x2xi64,
  // CHECK: scf.condition({{%.*}}) {{.*}} : tensor<2x2xi64,
  // CHECK: ^bb0({{%.*}}: tensor<2x2xi64,
  // CHECK: scf.yield {{.*}} : tensor<4xi64,
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @different_region_ranks(%initial: tensor<4xi64, #blocked>, %step: tensor<4xi64, #blocked>, %output: tensor<2x2x!tt.ptr<i64>, #matrix>, %again: i1) {
    %result = scf.while (%seen = %initial) : (tensor<4xi64, #blocked>) -> tensor<2x2xi64, #matrix> {
      %forwarded = tt.reshape %seen : tensor<4xi64, #blocked> -> tensor<2x2xi64, #matrix>
      scf.condition(%again) %forwarded : tensor<2x2xi64, #matrix>
    } do {
    ^bb0(%matrix_value: tensor<2x2xi64, #matrix>):
      %flat = tt.reshape %matrix_value : tensor<2x2xi64, #matrix> -> tensor<4xi64, #blocked>
      %next = arith.addi %flat, %step : tensor<4xi64, #blocked>
      scf.yield %next : tensor<4xi64, #blocked>
    }
    tt.store %output, %result : tensor<2x2x!tt.ptr<i64>, #matrix>
    tt.return
  }
}
