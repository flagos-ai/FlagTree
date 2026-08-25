// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s --implicit-check-not='ttg.convert_layout {{.*}}!tt.ptr<i32>'

#blocked_a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-LABEL: tt.func public @shared_constant_encoding_domains
// CHECK: ttg.convert_layout {{.*}} -> tensor<128xi32, #[[BLOCKED_A:blocked[0-9]*]]>
// CHECK: ttg.convert_layout {{.*}} -> tensor<128xi32, #[[BLOCKED_B:blocked[0-9]*]]>
// CHECK-DAG: arith.constant {{.*}}dense<1> : tensor<128xi32, #[[BLOCKED_A]]>
// CHECK-DAG: arith.constant {{.*}}dense<1> : tensor<128xi32, #[[BLOCKED_B]]>
// CHECK: arith.addi {{.*}} : tensor<128xi32, #[[BLOCKED_A]]>
// CHECK: arith.addi {{.*}} : tensor<128xi32, #[[BLOCKED_B]]>
module {
  tt.func public @shared_constant_encoding_domains() attributes {noinline = false} {
    %lhs_value = arith.constant dense<0> : tensor<128xi32>
    %lhs = tle.encoding %lhs_value {target_encoding = #blocked_a} : tensor<128xi32> -> tensor<128xi32>
    %rhs_value = arith.constant dense<0> : tensor<128xi32>
    %rhs = tle.encoding %rhs_value {target_encoding = #blocked_b} : tensor<128xi32> -> tensor<128xi32>
    %shared = arith.constant dense<1> : tensor<128xi32>
    %lhs_sum = arith.addi %lhs, %shared : tensor<128xi32>
    %rhs_sum = arith.addi %rhs, %shared : tensor<128xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func public @shared_splat_encoding_domains
  // CHECK: ttg.convert_layout {{.*}} -> tensor<128xi32, #[[SPLAT_A:blocked[0-9]*]]>
  // CHECK: ttg.convert_layout {{.*}} -> tensor<128xi32, #[[SPLAT_B:blocked[0-9]*]]>
  // CHECK-DAG: tt.splat {{.*}} -> tensor<128x!tt.ptr<i32>, #[[SPLAT_A]]>
  // CHECK-DAG: tt.splat {{.*}} -> tensor<128x!tt.ptr<i32>, #[[SPLAT_B]]>
  // CHECK-DAG: tt.addptr {{.*}} : tensor<128x!tt.ptr<i32>, #[[SPLAT_A]]>, tensor<128xi32, #[[SPLAT_A]]>
  // CHECK-DAG: tt.addptr {{.*}} : tensor<128x!tt.ptr<i32>, #[[SPLAT_B]]>, tensor<128xi32, #[[SPLAT_B]]>
  tt.func public @shared_splat_encoding_domains(%base: !tt.ptr<i32>) attributes {noinline = false} {
    %lhs_value = arith.constant dense<0> : tensor<128xi32>
    %lhs = tle.encoding %lhs_value {target_encoding = #blocked_a} : tensor<128xi32> -> tensor<128xi32>
    %rhs_value = arith.constant dense<0> : tensor<128xi32>
    %rhs = tle.encoding %rhs_value {target_encoding = #blocked_b} : tensor<128xi32> -> tensor<128xi32>
    %shared = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %lhs_ptr = tt.addptr %shared, %lhs : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %rhs_ptr = tt.addptr %shared, %rhs : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.return
  }
}
