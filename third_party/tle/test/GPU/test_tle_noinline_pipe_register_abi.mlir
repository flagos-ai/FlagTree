// RUN: triton-opt %s --split-input-file --tritongpu-allocate-warp-groups | FileCheck %s

// Device calls require a fixed register ABI. Dynamic setmaxnreg budgets cannot
// be assumed across them, even if the producer has requestedRegisters.
// Physical worker allocation uses ordinary four-warp padding.
// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func private @role() attributes {noinline = true} {
    tt.return
  }
  tt.func @kernel() {
    // CHECK: ttg.warp_specialize() attributes {requestedRegisters = array<i32: 32, 16, 16>, warpGroupStartIds = array<i32: 6, 4, 7>}
    // CHECK-NOT: actualRegisters
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 32>}
    default {
      tt.call @role() : () -> ()
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      tt.call @role() : () -> ()
      ttg.warp_return
    } : () -> ()
    // CHECK: partition1() num_warps(2)
    // CHECK: partition2() num_warps(1)
    tt.return
  }
}

// -----

// An ordinary region still rounds up to four workers and receives a register
// reallocation plan. The fixed device-call ABI must not disable this behavior
// in call-free IR.
// CHECK: module attributes {ttg.maxnreg = 128 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32}
module attributes {"ttg.maxnreg" = 128 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @ordinary_padding
  tt.func @ordinary_padding() {
    // CHECK: actualRegisters = array<i32: 232, 24, 24, 24>
    // CHECK-SAME: requestedRegisters = array<i32: 24, 16, 16>
    // CHECK-SAME: warpGroupStartIds = array<i32: 6, 4, 7>
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

// A user cap remains a cap, not permission to synthesize a setmaxnreg plan for
// a fixed device-call ABI; this must also hold with a full four-warp worker.
// CHECK: module attributes {ttg.maxnreg = 128 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32}
module attributes {"ttg.maxnreg" = 128 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @callee() attributes {noinline = true} { tt.return }
  // CHECK-LABEL: tt.func @fixed_call_registers
  tt.func @fixed_call_registers() {
    // CHECK: attributes {requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 32>}
    default {
      tt.call @callee() : () -> ()
      ttg.warp_yield
    }
    partition0() num_warps(4) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}
