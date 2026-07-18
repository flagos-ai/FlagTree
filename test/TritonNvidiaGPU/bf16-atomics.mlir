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

// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// CHECK: llvm.atomicrmw fadd

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:80",
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  tt.func public @triton_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.load %arg1 : !tt.ptr<bf16>
    %2 = tt.addptr %arg2, %0 : !tt.ptr<bf16>, i64
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %1, %true {allocation.offset = 0 : i32} : (!tt.ptr<bf16>, bf16, i1) -> bf16
    tt.store %arg3, %3 : !tt.ptr<bf16>
    tt.return
  }
}


// CHECK: atom.global.gpu.acq_rel.add.noftz.bf16

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:90",
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  tt.func public @triton_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.load %arg1 : !tt.ptr<bf16>
    %2 = tt.addptr %arg2, %0 : !tt.ptr<bf16>, i64
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %1, %true {allocation.offset = 0 : i32} : (!tt.ptr<bf16>, bf16, i1) -> bf16
    tt.store %arg3, %3 : !tt.ptr<bf16>
    tt.return
  }
}
