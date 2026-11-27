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
