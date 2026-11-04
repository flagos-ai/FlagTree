// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
//
// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=8 enable-source-remat=false' -verify-each | FileCheck %s
//
// A W4 noinline worker retains its ttg.num-warps function context when the
// surrounding launch defaults to W8. Dot's result encoding, ExpandDims' sliced
// layout fallback and Split's normalized source encoding must use that context.
// The verifier rejects each original W8 fallback inside the W4 function.
// Original loop arguments retain the same scope after signature conversion
// temporarily detaches their blocks from the function.
//
// CHECK: warpsPerCTA = [4
// CHECK-LABEL: tt.func private @{{.*}}dot_worker
module {
  tt.func public @dot_entry(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ttg.warp_specialize(%arg0, %arg1, %arg2) attributes {requestedRegisters = array<i32: 96>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !tt.ptr<f8E4M3FN>, %arg4: !tt.ptr<f8E4M3FN>, %arg5: !tt.ptr<f32>) num_warps(4) {
      tt.call @__main__.dot_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg3, %arg4, %arg5) : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
      ttg.warp_return
    } : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
    tt.return
  }
  tt.func private @__main__.dot_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f32>) attributes {noinline = true, "ttg.num-warps" = 4 : i32} {
    %cst = arith.constant dense<8> : tensor<64x1xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<64x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x8xf32>
    %cst_2 = arith.constant dense<8> : tensor<128x1xi32>
    %cst_3 = arith.constant dense<128> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %4 = arith.muli %3, %cst_3 : tensor<64x1xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %8 = tt.broadcast %6 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
    %9 = tt.broadcast %7 : tensor<1x128xi32> -> tensor<64x128xi32>
    %10 = tt.addptr %8, %9 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
    %11 = tt.load %10 : tensor<64x128x!tt.ptr<f8E4M3FN>>
    %12 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %13 = arith.muli %12, %cst_2 : tensor<128x1xi32>
    %14 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
    %15 = tt.addptr %14, %13 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
    %16 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %17 = tt.broadcast %15 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x8x!tt.ptr<f8E4M3FN>>
    %18 = tt.broadcast %16 : tensor<1x8xi32> -> tensor<128x8xi32>
    %19 = tt.addptr %17, %18 : tensor<128x8x!tt.ptr<f8E4M3FN>>, tensor<128x8xi32>
    %20 = tt.load %19 : tensor<128x8x!tt.ptr<f8E4M3FN>>
    %21 = tt.dot %11, %20, %cst_1, inputPrecision = tf32 {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf8E4M3FN> * tensor<128x8xf8E4M3FN> -> tensor<64x8xf32>
    %22 = math.fma %21, %cst_0, %21 : tensor<64x8xf32>
    %23 = arith.muli %3, %cst : tensor<64x1xi32>
    %24 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %26 = tt.broadcast %25 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x8x!tt.ptr<f32>>
    %27 = tt.broadcast %16 : tensor<1x8xi32> -> tensor<64x8xi32>
    %28 = tt.addptr %26, %27 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
    tt.store %28, %22 : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// CHECK: warpsPerCTA = [4
// CHECK-LABEL: tt.func private @{{.*}}expand_worker
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @expand_entry(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ttg.warp_specialize(%arg0, %arg1, %arg2) attributes {requestedRegisters = array<i32: 96>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !tt.ptr<f8E4M3FN>, %arg4: !tt.ptr<f8E4M3FN>, %arg5: !tt.ptr<f32>) num_warps(4) {
      tt.call @__main__.expand_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg3, %arg4, %arg5) : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
      ttg.warp_return
    } : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
    tt.return
  }
  tt.func private @__main__.expand_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f32>) attributes {noinline = true, "ttg.num-warps" = 4 : i32} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<64x1xf32>
    %cst_1 = arith.constant dense<8> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f8E4M3FN>>, tensor<64xi32>
    %3 = tt.load %2 : tensor<64x!tt.ptr<f8E4M3FN>>
    %4 = tt.fp_to_fp %3 : tensor<64xf8E4M3FN> -> tensor<64xf32>
    %5 = tle.gpu.set_layout %4 {target_encoding = #ttg.slice<{dim = 0, parent = #blocked}>} : tensor<64xf32> -> tensor<64xf32>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %7 = math.fma %6, %cst_0, %cst : tensor<64x1xf32>
    %8 = tt.broadcast %7 : tensor<64x1xf32> -> tensor<64x8xf32>
    %9 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %10 = arith.muli %9, %cst_1 : tensor<64x1xi32>
    %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %13 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %15 = tt.broadcast %12 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x8x!tt.ptr<f32>>
    %16 = tt.broadcast %14 : tensor<1x8xi32> -> tensor<64x8xi32>
    %17 = tt.addptr %15, %16 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
    tt.store %17, %8 : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// CHECK: warpsPerCTA = [4
// CHECK-LABEL: tt.func private @{{.*}}split_worker
module {
  tt.func public @split_entry(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ttg.warp_specialize(%arg0, %arg1, %arg2) attributes {requestedRegisters = array<i32: 96>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !tt.ptr<f8E4M3FN>, %arg4: !tt.ptr<f8E4M3FN>, %arg5: !tt.ptr<f32>) num_warps(4) {
      tt.call @__main__.split_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg3, %arg4, %arg5) : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
      ttg.warp_return
    } : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f32>) -> ()
    tt.return
  }
  tt.func private @__main__.split_worker__Pfp8e4nv_Pfp8e4nv_Pfp32___TLEWSNW4(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f32>) attributes {noinline = true, "ttg.num-warps" = 4 : i32} {
    %cst = arith.constant dense<2> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = arith.muli %1, %cst : tensor<64x1xi32>
    %3 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.broadcast %2 : tensor<64x1xi32> -> tensor<64x2xi32>
    %6 = tt.broadcast %4 : tensor<1x2xi32> -> tensor<64x2xi32>
    %7 = arith.addi %5, %6 : tensor<64x2xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x2x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %7 : tensor<64x2x!tt.ptr<f8E4M3FN>>, tensor<64x2xi32>
    %10 = tt.load %9 : tensor<64x2x!tt.ptr<f8E4M3FN>>
    %11 = tt.fp_to_fp %10 : tensor<64x2xf8E4M3FN> -> tensor<64x2xf32>
    %outLHS, %outRHS = tt.split %11 : tensor<64x2xf32> -> tensor<64xf32>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %13 = tt.addptr %12, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %14 = arith.addf %outLHS, %outRHS : tensor<64xf32>
    tt.store %13, %14 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A loop must preserve the W4 worker context while converting its body.
// CHECK: warpsPerCTA = [4
// CHECK-LABEL: tt.func private @worker
module {
  tt.func public @entry(%out: !tt.ptr<f32>) {
    ttg.warp_specialize(%out) attributes {requestedRegisters = array<i32: 96>}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !tt.ptr<f32>) num_warps(4) {
      tt.call @worker(%arg) : (!tt.ptr<f32>) -> ()
      ttg.warp_return
    } : (!tt.ptr<f32>) -> ()
    tt.return
  }
  tt.func private @worker(%out: !tt.ptr<f32>) attributes {noinline = true, "ttg.num-warps" = 4 : i32} {
    %zero = arith.constant dense<0.0> : tensor<64x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> tensor<64x8xf32> {
      %value = math.fma %acc, %zero, %zero : tensor<64x8xf32>
      scf.yield %value : tensor<64x8xf32>
    }
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    tt.store %ptr, %result : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}
