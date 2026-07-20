/*
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: arrive_barrier_release_fence
  tt.func @arrive_barrier_release_fence(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r"
    ttng.arrive_barrier %alloc, 2 {release_fence = true} : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: participant_arrive_barrier_release_fence
  tt.func @participant_arrive_barrier_release_fence(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "@$0 membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r"
    ttng.arrive_barrier %alloc, 64 {participant_arrive = true, release_fence = true} : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
