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

// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {
}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 20 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @kernel() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 18, 4, 12, 16, 19>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition3() num_warps(2)
  // CHECK: partition4() num_warps(1)
  tt.return
}

}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 16 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @two_warp_specialize() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 12, 14, 4, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition2() num_warps(8)
  // CHECK: partition3() num_warps(1)

  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 14, 4, 12, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  } : () -> ()

  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 168 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @setmaxnreg() {
  // CHECK: actualRegisters = array<i32: 208, 80, 80, 80>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 48, 80, 48>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(2) {
    ttg.warp_return
  }
  partition2() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 128 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @steal_from_default() {
  // CHECK: actualRegisters = array<i32: 64, 192>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 192>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
