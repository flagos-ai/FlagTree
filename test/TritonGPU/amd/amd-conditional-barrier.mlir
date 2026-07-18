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

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx942" | FileCheck %s

module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @conditional_barrier() {
    // CHECK-LABEL: llvm.func @conditional_barrier

    // CHECK:   %[[CMP0:.+]] = llvm.icmp "ne" %[[OP0:.+]], %[[OP1:.+]] : i32
    // CHECK:   %[[CMP1:.+]] = llvm.icmp "eq" %[[OP0]], %[[OP1]] : i32
    // CHECK:   llvm.cond_br %[[CMP0]], ^bb1, ^bb2
    // CHECK: ^bb1:
    // CHECK:   rocdl.s.barrier
    // CHECK:   llvm.br ^bb2
    // CHECK: ^bb2:
    // CHECK:   llvm.add
    // CHECK:   llvm.cond_br %[[CMP1]], ^bb3, ^bb4
    // CHECK: ^bb3:
    // CHECK:   rocdl.s.barrier
    // CHECK:   llvm.br ^bb4
    // CHECK: ^bb4:
    // CHECK:   llvm.return

    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = rocdl.workitem.id.x : i32
    %1 = arith.divsi %0, %c256_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = arith.cmpi eq, %1, %c0_i32 : i32
    amdg.cond_barrier %2
    %4 = arith.addi %0, %c256_i32 : i32
    amdg.cond_barrier %3
    tt.return
  }
}
