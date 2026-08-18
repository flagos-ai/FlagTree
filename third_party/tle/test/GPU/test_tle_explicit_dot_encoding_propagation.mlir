// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module {
  // CHECK-LABEL: tt.func public @chained_dot_accumulator
  tt.func public @chained_dot_accumulator() attributes {noinline = false} {
    %lhs_value = arith.constant dense<0.0> : tensor<32x32xbf16>
    %lhs_encoded = tle.gpu.set_layout %lhs_value {target_encoding = #lhs} : tensor<32x32xbf16> -> tensor<32x32xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<32x8xbf16>
    %rhs_encoded = tle.gpu.set_layout %rhs_value {target_encoding = #rhs} : tensor<32x8xbf16> -> tensor<32x8xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<32x8xf32>
    %acc_encoded = tle.gpu.set_layout %acc_value {target_encoding = #mma} : tensor<32x8xf32> -> tensor<32x8xf32>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[FIRST:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {{.*}} -> tensor<32x8xf32, #mma>
    %first = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NEXT: %[[SECOND:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[FIRST]] {{.*}} -> tensor<32x8xf32, #mma>
    %second = tt.dot %lhs_encoded, %rhs_encoded, %first : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NOT: ttg.convert_layout
    tt.return
  }
}
