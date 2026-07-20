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

// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -convert-warp-specialize-to-llvm -canonicalize=region-simplify=disabled | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @do_not_remat_special_register_capture
llvm.func @do_not_remat_special_register_capture() attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: [[CTAID:%.*]] = nvvm.read.ptx.sreg.ctaid.x
  // CHECK-NEXT: [[PID:%.*]] = llvm.udiv [[CTAID]], [[C4]] : i32
  // CHECK: ^bb4:
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NOT: nvvm.read.ptx.sreg.ctaid.x
  // CHECK-NOT: llvm.load
  // CHECK-NEXT: "use"([[PID]])
  // CHECK-NOT: !llvm.struct<packed (i32)>
  %c4 = llvm.mlir.constant(4 : i32) : i32
  %ctaid = nvvm.read.ptx.sreg.ctaid.x : i32
  %pid = llvm.udiv %ctaid, %c4 : i32
  ttg.warp_specialize(%pid) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: i32) num_warps(1) {
    "use"(%arg0) : (i32) -> ()
    ttg.warp_return
  } : (i32) -> ()
  llvm.return
}

}
