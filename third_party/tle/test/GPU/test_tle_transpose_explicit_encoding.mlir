// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s

// The requested consumer encoding is not necessarily the transpose encoding
// inferred from its source (even equivalent Dot/Linear attrs are distinct).
// This reproduces the layout boundary exposed when an attention tile helper
// is inlined; no pipe, model, or generated template is needed.
// Pointer arguments and stores isolate the view conversion from tensor-valued
// function signature inference, which is outside this test's contract.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module {
  // CHECK-LABEL: tt.func @constant_transpose_to_dot
  tt.func @constant_transpose_to_dot(%output: !tt.ptr<bf16>) {
    %input = arith.constant dense<1.0> : tensor<64x128xbf16>
    %trans = tt.trans %input {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    // Constant folding is legal and must retain the result encoding.
    // CHECK: %[[RESULT:.*]] = arith.constant {{.*}} : tensor<128x64xbf16, #ttg.dot_op<{{.*}}>>
    %result = tle.gpu.set_layout %trans {target_encoding = #rhs} : tensor<128x64xbf16> -> tensor<128x64xbf16>
    %output_ptrs = tt.splat %output : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    // CHECK: tt.store %{{.*}}, %[[RESULT]]
    tt.store %output_ptrs, %result : tensor<128x64x!tt.ptr<bf16>>
    tt.return
  }

  // CHECK-LABEL: tt.func @transpose_to_dot
  tt.func @transpose_to_dot(%input: !tt.ptr<bf16>, %output: !tt.ptr<bf16>) {
    %input_ptrs = tt.splat %input : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %input_value = tt.load %input_ptrs : tensor<64x128x!tt.ptr<bf16>>
    // CHECK: %[[TRANS:.*]] = tt.trans
    %trans = tt.trans %input_value {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    // CHECK: %[[RESULT:.*]] = ttg.convert_layout %[[TRANS]] {{.*}} -> tensor<128x64xbf16, #ttg.dot_op<{{.*}}>>
    %result = tle.gpu.set_layout %trans {target_encoding = #rhs} : tensor<128x64xbf16> -> tensor<128x64xbf16>
    %output_ptrs = tt.splat %output : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    // CHECK: tt.store %{{.*}}, %[[RESULT]]
    tt.store %output_ptrs, %result : tensor<128x64x!tt.ptr<bf16>>
    tt.return
  }
}
