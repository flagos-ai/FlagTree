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

// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s --check-prefixes=COMMON,OFF
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true" | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-small-component-solving=false" | FileCheck %s --check-prefixes=COMMON,BWD
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-cost-based-resolution=false enable-store-layout-rematerialization=false" | FileCheck %s --check-prefixes=COMMON,OFF

// Phase 1b: the shared math chain is too expensive for backward remat, so the
// baseline keeps both writeback converts. Backward propagation moves the chain
// onto the store layout instead.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func public @backward_writeback
  // OFF-COUNT-2: ttg.convert_layout
  // ON-NOT: ttg.convert_layout
  // ON: math.exp %{{.*}} : tensor<1024xf32, #[[L:.*]]>
  // ON-NOT: ttg.convert_layout
  // ON: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[L]]>
  // BWD-NOT: ttg.convert_layout
  // BWD: math.exp %{{.*}} : tensor<1024xf32, #[[L:.*]]>
  // BWD-NOT: ttg.convert_layout
  // BWD: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[L]]>
  // COMMON: tt.return
  tt.func public @backward_writeback(%out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %one = arith.constant dense<1.000000e+00> : tensor<1024xf32, #blocked>
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %f = arith.sitofp %r : tensor<1024xi32, #blocked> to tensor<1024xf32, #blocked>
    %e0 = math.exp %f : tensor<1024xf32, #blocked>
    %e1 = math.log %e0 : tensor<1024xf32, #blocked>
    %e2 = math.sin %e1 : tensor<1024xf32, #blocked>
    %e3 = math.cos %e2 : tensor<1024xf32, #blocked>
    %e4 = math.sqrt %e3 : tensor<1024xf32, #blocked>
    %a = arith.addf %e4, %one : tensor<1024xf32, #blocked>
    %b = arith.mulf %e4, %e4 : tensor<1024xf32, #blocked>
    %ca = ttg.convert_layout %a : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    %cb = ttg.convert_layout %b : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %p0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %p0o = tt.addptr %p0, %offs : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    tt.store %p0o, %ca : tensor<1024x!tt.ptr<f32>, #blocked1>
    %p1 = tt.splat %out1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %p1o = tt.addptr %p1, %offs : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    tt.store %p1o, %cb : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Phase 2: the two reduce results are written back through converts that share
// a mask; the closed component (stores, pointers, mask) is retagged onto the
// reduce's slice layout as a whole.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func public @reduce_writeback
  // OFF-COUNT-2: ttg.convert_layout
  // BWD-COUNT-2: ttg.convert_layout
  // ON-NOT: ttg.convert_layout
  // ON: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked}>>
  // ON-NOT: ttg.convert_layout
  // ON: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
  // COMMON: tt.return
  tt.func public @reduce_writeback(%x: tensor<64x1024xf16, #blocked>, %idx: tensor<64x1024xi32, #blocked>, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %M: i32) {
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %ms = tt.splat %M : i32 -> tensor<64xi32, #blocked1>
    %mask = arith.cmpi slt, %rm, %ms : tensor<64xi32, #blocked1>
    %0:2 = "tt.reduce"(%x, %idx) <{axis = 1 : i32}> ({
    ^bb0(%v0: f16, %i0: i32, %v1: f16, %i1: i32):
      %gt = arith.cmpf ogt, %v0, %v1 : f16
      %v = arith.select %gt, %v0, %v1 : f16
      %i = arith.select %gt, %i0, %i1 : i32
      tt.reduce.return %v, %i : f16, i32
    }) : (tensor<64x1024xf16, #blocked>, tensor<64x1024xi32, #blocked>) -> (tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>)
    %vp = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
    %vpo = tt.addptr %vp, %rm : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
    %vc = ttg.convert_layout %0#0 : tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf16, #blocked1>
    tt.store %vpo, %vc, %mask : tensor<64x!tt.ptr<f16>, #blocked1>
    %ip = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked1>
    %ipo = tt.addptr %ip, %rm : tensor<64x!tt.ptr<i32>, #blocked1>, tensor<64xi32, #blocked1>
    %ic = ttg.convert_layout %0#1 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi32, #blocked1>
    tt.store %ipo, %ic, %mask : tensor<64x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----

// Phase 3: the convert feeding a result-unused atomic from an MMA value is
// dropped by rematerializing the pointer chain in the MMA layout. A coalesced
// tt.store keeps its convert, since the MMA layout would break the coalescing.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func public @atomic_writeback
  // OFF: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
  // OFF: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}}, %{{.*}} : (tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xf32, #blocked>)
  // ON-NOT: ttg.convert_layout
  // ON: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}}, %{{.*}} : (tensor<64x64x!tt.ptr<f32>, #mma>, tensor<64x64xf32, #mma>)
  // BWD-NOT: ttg.convert_layout
  // BWD: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}}, %{{.*}} : (tensor<64x64x!tt.ptr<f32>, #mma>, tensor<64x64xf32, #mma>)
  // COMMON: tt.return
  tt.func public @atomic_writeback(%c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %acc: tensor<64x64xf32, #mma>) {
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rm2 = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %ns = tt.splat %N : i32 -> tensor<64x1xi32, #blocked>
    %row = arith.muli %rm2, %ns : tensor<64x1xi32, #blocked>
    %base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %prow = tt.addptr %base, %row : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %pb = tt.broadcast %prow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %cb = tt.broadcast %rn2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %ptr = tt.addptr %pb, %cb : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked>
    tt.return
  }

  // COMMON-LABEL: tt.func public @contiguous_store
  // COMMON: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
  // COMMON: tt.store %{{.*}}, %{{.*}} : tensor<64x64x!tt.ptr<f32>, #blocked>
  tt.func public @contiguous_store(%c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %acc: tensor<64x64xf32, #mma>) {
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rm2 = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %ns = tt.splat %N : i32 -> tensor<64x1xi32, #blocked>
    %row = arith.muli %rm2, %ns : tensor<64x1xi32, #blocked>
    %base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %prow = tt.addptr %base, %row : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %pb = tt.broadcast %prow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %cb = tt.broadcast %rn2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %ptr = tt.addptr %pb, %cb : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    tt.store %ptr, %val : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
