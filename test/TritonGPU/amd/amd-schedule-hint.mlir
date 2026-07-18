// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" | FileCheck %s -check-prefix=INSTR_HINT
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=LOWER_HINT

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [32, 32, 8], isTransposed = true}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
// INSTR_HINT-LABEL: @insert_schedule_hint
// LOWER_HINT-LABEL: @insert_schedule_hint
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @insert_schedule_hint(
    %lb : index, %ub : index, %step : index,
    %arg0: tensor<128x128xf32, #dot_op_a>,
    %arg1: tensor<128x128xf32, #dot_op_b>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // INSTR_HINT: scf.for
    // INSTR_HINT-NEXT: amdg.instruction_sched_hint
    // INSTR_HINT-SAME: variant = #amdg.SchedHintVariant<attention>

    // LOWER_HINT: scf.for
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-COUNT-2: tt.dot
    // LOWER_HINT: rocdl.iglp.opt 2
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-NEXT: scf.yield
    %loop = scf.for %iv = %lb to %ub step %step iter_args(%c = %cst) -> (tensor<128x128xf32, #mma>) {
      %4 = tt.dot %arg0, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      %5 = math.exp2 %4 : tensor<128x128xf32, #mma>
      %6 = ttg.convert_layout %5 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #dot_op_a>
      %7 = tt.dot %6, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      scf.yield %7 : tensor<128x128xf32, #mma>
    }
    %8 = ttg.convert_layout %loop : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
