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

// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm --llvm-optimize-for-nvvm-target | FileCheck %s

// CHECK-LABEL: dedup_by_constancy_full
// CHECK-COUNT-2: llvm.add
// CHECK-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK: llvm.sdiv
// CHECK-NOT: llvm.sdiv
// CHECK: llvm.getelementptr %arg0[[[REGISTER:%[0-9]+]]]
// CHECK-COUNT-7: llvm.getelementptr %arg0[[[REGISTER]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER]]]
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_full(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<256> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_by_constancy_partial
// CHECK-COUNT-4: llvm.add
// CHECK-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK-COUNT-2: llvm.sdiv
// CHECK-NOT: llvm.sdiv
// CHECK: llvm.getelementptr %arg0[[[REGISTER1:%[0-9]+]]]
// CHECK-COUNT-3: llvm.getelementptr %arg0[[[REGISTER1]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER1]]]
// CHECK: llvm.getelementptr %arg0[[[REGISTER2:%[0-9]+]]]
// CHECK-COUNT-3: llvm.getelementptr %arg0[[[REGISTER2]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER2]]]
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_partial(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<4> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
