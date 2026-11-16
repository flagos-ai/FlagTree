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

// RUN: triton-opt %s -split-input-file -tritongpu-merge-duplicate-loads | FileCheck %s --check-prefixes=COMMON,ALL
// RUN: triton-opt %s -split-input-file -tritongpu-merge-duplicate-loads=mode=any | FileCheck %s --check-prefixes=COMMON,ANY

// The same 64 page indices are loaded in two layouts (as RemoveLayoutConversions
// leaves them for the K and V address computations). The pass keeps one load
// and converts its layout for the other user.

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#S0 = #ttg.slice<{dim = 0, parent = #BL}>
#S1 = #ttg.slice<{dim = 1, parent = #AL}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// COMMON-LABEL: @both_pipelined
// COMMON: scf.for
// COMMON: %[[KEEP:.*]] = tt.load {{.*}} {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #{{.*}}>
// COMMON-NOT: tt.load
// COMMON: ttg.convert_layout %[[KEEP]]
// COMMON: scf.yield
tt.func @both_pipelined(%lb : i32, %ub : i32, %step : i32, %base : !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) {
  %c64 = arith.constant 64 : i32
  %z0 = arith.constant dense<0> : tensor<64xi32, #S0>
  %z1 = arith.constant dense<0> : tensor<64xi32, #S1>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc0 = %z0, %acc1 = %z1) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) : i32 {
    %k = arith.muli %iv, %c64 : i32
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S0>
    %k0 = tt.splat %k : i32 -> tensor<64xi32, #S0>
    %o0 = arith.addi %k0, %r0 : tensor<64xi32, #S0>
    %b0 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S0>
    %p0 = tt.addptr %b0, %o0 : tensor<64x!tt.ptr<i32>, #S0>, tensor<64xi32, #S0>
    %v0 = tt.load %p0 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S0>
    %r1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S1>
    %k1 = tt.splat %k : i32 -> tensor<64xi32, #S1>
    %o1 = arith.addi %k1, %r1 : tensor<64xi32, #S1>
    %b1 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S1>
    %p1 = tt.addptr %b1, %o1 : tensor<64x!tt.ptr<i32>, #S1>, tensor<64xi32, #S1>
    %v1 = tt.load %p1 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S1>
    %n0 = arith.addi %acc0, %v0 : tensor<64xi32, #S0>
    %n1 = arith.addi %acc1, %v1 : tensor<64xi32, #S1>
    scf.yield %n0, %n1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
  }
  tt.return %loop#0, %loop#1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
}

// A global store between the two loads may change the data: keep both.
// COMMON-LABEL: @store_between
// COMMON: scf.for
// COMMON: tt.load
// COMMON: tt.store
// COMMON: tt.load
// COMMON-NOT: ttg.convert_layout
// COMMON: scf.yield
tt.func @store_between(%lb : i32, %ub : i32, %step : i32, %base : !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out : tensor<64x!tt.ptr<i32>, #S0>) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) {
  %c64 = arith.constant 64 : i32
  %z0 = arith.constant dense<0> : tensor<64xi32, #S0>
  %z1 = arith.constant dense<0> : tensor<64xi32, #S1>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc0 = %z0, %acc1 = %z1) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) : i32 {
    %k = arith.muli %iv, %c64 : i32
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S0>
    %k0 = tt.splat %k : i32 -> tensor<64xi32, #S0>
    %o0 = arith.addi %k0, %r0 : tensor<64xi32, #S0>
    %b0 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S0>
    %p0 = tt.addptr %b0, %o0 : tensor<64x!tt.ptr<i32>, #S0>, tensor<64xi32, #S0>
    %v0 = tt.load %p0 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S0>
    tt.store %out, %v0 : tensor<64x!tt.ptr<i32>, #S0>
    %r1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S1>
    %k1 = tt.splat %k : i32 -> tensor<64xi32, #S1>
    %o1 = arith.addi %k1, %r1 : tensor<64xi32, #S1>
    %b1 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S1>
    %p1 = tt.addptr %b1, %o1 : tensor<64x!tt.ptr<i32>, #S1>, tensor<64xi32, #S1>
    %v1 = tt.load %p1 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S1>
    %n0 = arith.addi %acc0, %v0 : tensor<64xi32, #S0>
    %n1 = arith.addi %acc1, %v1 : tensor<64xi32, #S1>
    scf.yield %n0, %n1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
  }
  tt.return %loop#0, %loop#1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
}

// Only the second copy is pipelined. mode=all leaves the group alone;
// mode=any keeps the pipelined copy, moves it (with its address computation)
// in front of the first copy and converts it for the first user.
// COMMON-LABEL: @one_pipelined
// COMMON: scf.for
// ALL: tt.load {{.*}} : tensor<64x!tt.ptr<i32>, #{{.*}}>
// ALL: tt.load {{.*}} {tt.latency = 1 : i32}
// ALL-NOT: ttg.convert_layout
// ANY: %[[KEEP:.*]] = tt.load {{.*}} {tt.latency = 1 : i32}
// ANY-NOT: tt.load
// ANY: ttg.convert_layout %[[KEEP]]
// COMMON: scf.yield
tt.func @one_pipelined(%lb : i32, %ub : i32, %step : i32, %base : !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) {
  %c64 = arith.constant 64 : i32
  %z0 = arith.constant dense<0> : tensor<64xi32, #S0>
  %z1 = arith.constant dense<0> : tensor<64xi32, #S1>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc0 = %z0, %acc1 = %z1) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) : i32 {
    %k = arith.muli %iv, %c64 : i32
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S0>
    %k0 = tt.splat %k : i32 -> tensor<64xi32, #S0>
    %o0 = arith.addi %k0, %r0 : tensor<64xi32, #S0>
    %b0 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S0>
    %p0 = tt.addptr %b0, %o0 : tensor<64x!tt.ptr<i32>, #S0>, tensor<64xi32, #S0>
    %v0 = tt.load %p0 : tensor<64x!tt.ptr<i32>, #S0>
    %n0 = arith.addi %acc0, %v0 : tensor<64xi32, #S0>
    %r1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S1>
    %k1 = tt.splat %k : i32 -> tensor<64xi32, #S1>
    %o1 = arith.addi %k1, %r1 : tensor<64xi32, #S1>
    %b1 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S1>
    %p1 = tt.addptr %b1, %o1 : tensor<64x!tt.ptr<i32>, #S1>, tensor<64xi32, #S1>
    %v1 = tt.load %p1 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S1>
    %n1 = arith.addi %acc1, %v1 : tensor<64xi32, #S1>
    scf.yield %n0, %n1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
  }
  tt.return %loop#0, %loop#1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
}

// Different masks are different loads.
// COMMON-LABEL: @different_masks
// COMMON: scf.for
// COMMON: tt.load
// COMMON: tt.load
// COMMON-NOT: ttg.convert_layout
// COMMON: scf.yield
tt.func @different_masks(%lb : i32, %ub : i32, %step : i32, %base : !tt.ptr<i32> {tt.divisibility = 16 : i32}, %n : i32) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) {
  %c64 = arith.constant 64 : i32
  %z0 = arith.constant dense<0> : tensor<64xi32, #S0>
  %z1 = arith.constant dense<0> : tensor<64xi32, #S1>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc0 = %z0, %acc1 = %z1) -> (tensor<64xi32, #S0>, tensor<64xi32, #S1>) : i32 {
    %k = arith.muli %iv, %c64 : i32
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S0>
    %k0 = tt.splat %k : i32 -> tensor<64xi32, #S0>
    %o0 = arith.addi %k0, %r0 : tensor<64xi32, #S0>
    %n0s = tt.splat %n : i32 -> tensor<64xi32, #S0>
    %m0 = arith.cmpi slt, %o0, %n0s : tensor<64xi32, #S0>
    %b0 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S0>
    %p0 = tt.addptr %b0, %o0 : tensor<64x!tt.ptr<i32>, #S0>, tensor<64xi32, #S0>
    %v0 = tt.load %p0, %m0 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S0>
    %r1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #S1>
    %k1 = tt.splat %k : i32 -> tensor<64xi32, #S1>
    %o1 = arith.addi %k1, %r1 : tensor<64xi32, #S1>
    %n1s = tt.splat %n : i32 -> tensor<64xi32, #S1>
    %m1 = arith.cmpi sle, %o1, %n1s : tensor<64xi32, #S1>
    %b1 = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #S1>
    %p1 = tt.addptr %b1, %o1 : tensor<64x!tt.ptr<i32>, #S1>, tensor<64xi32, #S1>
    %v1 = tt.load %p1, %m1 {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>, #S1>
    %n0 = arith.addi %acc0, %v0 : tensor<64xi32, #S0>
    %n1 = arith.addi %acc1, %v1 : tensor<64xi32, #S1>
    scf.yield %n0, %n1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
  }
  tt.return %loop#0, %loop#1 : tensor<64xi32, #S0>, tensor<64xi32, #S1>
}
}
