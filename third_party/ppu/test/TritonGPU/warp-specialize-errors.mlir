// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file --convert-warp-specialize-to-llvm-ppu -verify-diagnostics

// An outlined helper is compiled for one warp group size (ws_num_warps): its
// barriers count that group's threads. Called from a group of another size it
// would wait for threads that never arrive, so the lowering reports the call.

// A worker partition calling a helper compiled for the default group.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, "ttg.threads-per-warp" = 32 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

llvm.func internal @helper_nw4() attributes {"ws_num_warps" = 4 : i32} {
  nvvm.barrier0
  llvm.return
}

llvm.func @kernel() attributes {allocation.offset = 32 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    llvm.call @helper_nw4() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    // expected-error @below {{helper 'helper_nw4' is compiled for 4 warps (ws_num_warps) but called from a warp group of 1 warps}}
    llvm.call @helper_nw4() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

// The default group calling a helper compiled for a worker partition.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, "ttg.threads-per-warp" = 32 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

llvm.func internal @helper_nw1() attributes {"ws_num_warps" = 1 : i32} {
  nvvm.barrier0
  llvm.return
}

llvm.func @kernel() attributes {allocation.offset = 32 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    // expected-error @below {{helper 'helper_nw1' is compiled for 1 warps (ws_num_warps) but called from a warp group of 4 warps}}
    llvm.call @helper_nw1() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    llvm.call @helper_nw1() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

// A helper calling a helper compiled for another size: the callee runs in the
// caller's group.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, "ttg.threads-per-warp" = 32 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

llvm.func internal @helper_nw1() attributes {"ws_num_warps" = 1 : i32} {
  // expected-error @below {{helper 'helper_nw4' is compiled for 4 warps (ws_num_warps) but called from a warp group of 1 warps}}
  llvm.call @helper_nw4() : () -> ()
  llvm.return
}

llvm.func internal @helper_nw4() attributes {"ws_num_warps" = 4 : i32} {
  nvvm.barrier0
  llvm.return
}

llvm.func @kernel() attributes {allocation.offset = 32 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    llvm.call @helper_nw4() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    llvm.call @helper_nw1() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
