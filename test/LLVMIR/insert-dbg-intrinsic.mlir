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

// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32 } {
  llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>,
                        %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<1>) {
    %constant_i32 = llvm.mlir.constant(3 : index) : i32

    // CHECK: %pid = rocdl.workgroup.id.x
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %pid :
    %pid = rocdl.workgroup.id.x : i32 loc(#loc14)

    // CHECK: %block_start = llvm.mul %pid
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %block_start :
    %block_start = llvm.mul %pid, %constant_i32 : i32 loc(#loc15)

    // CHECK: %offsets = llvm.add %block_start
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %offsets :
    %offsets = llvm.add %block_start, %constant_i32 : i32 loc(#loc16)
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32 loc(#loc17)

    llvm.return
  }
}
#loc2 = loc("01-vector-add.py":39:10)
#loc3 = loc("01-vector-add.py":44:18)
#loc5 = loc("01-vector-add.py":45:14)
#loc6 = loc("01-vector-add.py":47:11)
#loc14 = loc("pid"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("offsets"(#loc5))
#loc17 = loc("mask"(#loc6))
