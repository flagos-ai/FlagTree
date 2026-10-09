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
// dropped by rematerializing the pointer chain in the MMA layout, unless that
// narrows a vector (sm90 fadd) atomic. A coalesced tt.store keeps its convert,
// since the MMA layout would break the coalescing.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
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
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %rm2 = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %ns = tt.splat %N : i32 -> tensor<64x1xi32, #blocked1>
    %row = arith.muli %rm2, %ns : tensor<64x1xi32, #blocked1>
    %base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
    %prow = tt.addptr %base, %row : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
    %pb = tt.broadcast %prow : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %cb = tt.broadcast %rn2 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %ptr = tt.addptr %pb, %cb : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked1>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked1>
    tt.return
  }

  // COMMON-LABEL: tt.func public @vector_atomic_writeback
  // COMMON: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #[[$V:blocked[0-9]*]]>
  // COMMON: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}}, %{{.*}} : (tensor<64x64x!tt.ptr<f32>, #[[$V]]>, tensor<64x64xf32, #[[$V]]>)
  tt.func public @vector_atomic_writeback(%c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %acc: tensor<64x64xf32, #mma>) {
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
  // COMMON: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #[[$V]]>
  // COMMON: tt.store %{{.*}}, %{{.*}} : tensor<64x64x!tt.ptr<f32>, #[[$V]]>
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

// -----

// Phase 2 keeps a 64-bit store that already writes one 128-bit vector per
// thread: widening it to the load layout would split each thread's run into
// two strided stores, worse than the convert it removes. A store whose mask
// rules out vectorization writes element by element in either layout, so it
// may still widen.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func public @cast_to_f64
  // COMMON: tt.load %{{.*}} : tensor<512x!tt.ptr<f16>, #[[LD:blocked[0-9]*]]>
  // COMMON: ttg.convert_layout %{{.*}} : tensor<512x{{f16|f32}}, #[[LD]]> -> tensor<512x{{f16|f32}}, #[[ST:blocked[0-9]*]]>
  // COMMON: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f64>, #[[ST]]>
  tt.func public @cast_to_f64(%in0_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out0_ptr: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %s0: i32 {tt.divisibility = 16 : i32}, %num_tasks: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<512xi64, #blocked>
    %offset0 = arith.constant 512 : i64
    %pid = tt.get_program_id x : i32
    %pid_0 = arith.extsi %pid : i32 to i64
    %offset0_1 = arith.muli %pid_0, %offset0 : i64
    %offset0_2 = arith.trunci %offset0_1 : i64 to i32
    %in0_bptr = arith.extsi %s0 : i32 to i64
    %in0_bptr_3 = arith.extsi %offset0_2 : i32 to i64
    %in0 = tt.splat %in0_ptr : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %in0_4 = tt.splat %in0_bptr_3 : i64 -> tensor<512xi64, #blocked>
    %in0_5 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %in0_6 = arith.extsi %in0_5 : tensor<512xi32, #blocked> to tensor<512xi64, #blocked>
    %in0_7 = arith.addi %in0_4, %in0_6 : tensor<512xi64, #blocked>
    %in0_8 = tt.addptr %in0, %in0_7 : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi64, #blocked>
    %in0_9 = arith.cmpi sge, %in0_7, %cst : tensor<512xi64, #blocked>
    %in0_10 = tt.splat %in0_bptr : i64 -> tensor<512xi64, #blocked>
    %in0_11 = arith.cmpi slt, %in0_7, %in0_10 : tensor<512xi64, #blocked>
    %in0_12 = arith.andi %in0_9, %in0_11 : tensor<512xi1, #blocked>
    %in0_13 = ttg.convert_layout %in0_8 : tensor<512x!tt.ptr<f16>, #blocked> -> tensor<512x!tt.ptr<f16>, #blocked1>
    %in0_14 = ttg.convert_layout %in0_12 : tensor<512xi1, #blocked> -> tensor<512xi1, #blocked1>
    %in0_15 = tt.load %in0_13, %in0_14 : tensor<512x!tt.ptr<f16>, #blocked1>
    %in0_16 = ttg.convert_layout %in0_15 : tensor<512xf16, #blocked1> -> tensor<512xf16, #blocked>
    %0 = arith.extf %in0_16 : tensor<512xf16, #blocked> to tensor<512xf32, #blocked>
    %1 = arith.extf %0 : tensor<512xf32, #blocked> to tensor<512xf64, #blocked>
    %2 = tt.splat %out0_ptr : !tt.ptr<f64> -> tensor<512x!tt.ptr<f64>, #blocked>
    %3 = tt.addptr %2, %in0_7 : tensor<512x!tt.ptr<f64>, #blocked>, tensor<512xi64, #blocked>
    %4 = ttg.convert_layout %3 : tensor<512x!tt.ptr<f64>, #blocked> -> tensor<512x!tt.ptr<f64>, #blocked2>
    %5 = ttg.convert_layout %1 : tensor<512xf64, #blocked> -> tensor<512xf64, #blocked2>
    %6 = ttg.convert_layout %in0_12 : tensor<512xi1, #blocked> -> tensor<512xi1, #blocked2>
    tt.store %4, %5, %6 : tensor<512x!tt.ptr<f64>, #blocked2>
    tt.return
  }

  // COMMON-LABEL: tt.func public @cast_to_f64_misaligned_mask
  // OFF: ttg.convert_layout %{{.*}} : tensor<512x{{f16|f32}}, #{{.*}}> -> tensor<512x{{f16|f32}}, #{{.*}}>
  // ON-NOT: ttg.convert_layout
  // ON: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f64>, #{{.*}}>
  tt.func public @cast_to_f64_misaligned_mask(%in0_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out0_ptr: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %s0: i32 {tt.divisibility = 16 : i32}, %num_tasks: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<512xi64, #blocked>
    %offset0 = arith.constant 512 : i64
    %pid = tt.get_program_id x : i32
    %pid_0 = arith.extsi %pid : i32 to i64
    %offset0_1 = arith.muli %pid_0, %offset0 : i64
    %offset0_2 = arith.trunci %offset0_1 : i64 to i32
    %in0_bptr = arith.extsi %s0 : i32 to i64
    %in0_bptr_3 = arith.extsi %offset0_2 : i32 to i64
    %in0 = tt.splat %in0_ptr : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %in0_4 = tt.splat %in0_bptr_3 : i64 -> tensor<512xi64, #blocked>
    %in0_5 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %in0_6 = arith.extsi %in0_5 : tensor<512xi32, #blocked> to tensor<512xi64, #blocked>
    %in0_7 = arith.addi %in0_4, %in0_6 : tensor<512xi64, #blocked>
    %in0_8 = tt.addptr %in0, %in0_7 : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi64, #blocked>
    %in0_9 = arith.cmpi sge, %in0_7, %cst : tensor<512xi64, #blocked>
    %in0_10 = tt.splat %in0_bptr : i64 -> tensor<512xi64, #blocked>
    %in0_11 = arith.cmpi slt, %in0_7, %in0_10 : tensor<512xi64, #blocked>
    %in0_12 = arith.andi %in0_9, %in0_11 : tensor<512xi1, #blocked>
    %in0_13 = ttg.convert_layout %in0_8 : tensor<512x!tt.ptr<f16>, #blocked> -> tensor<512x!tt.ptr<f16>, #blocked1>
    %in0_14 = ttg.convert_layout %in0_12 : tensor<512xi1, #blocked> -> tensor<512xi1, #blocked1>
    %in0_15 = tt.load %in0_13, %in0_14 : tensor<512x!tt.ptr<f16>, #blocked1>
    %in0_16 = ttg.convert_layout %in0_15 : tensor<512xf16, #blocked1> -> tensor<512xf16, #blocked>
    %0 = arith.extf %in0_16 : tensor<512xf16, #blocked> to tensor<512xf32, #blocked>
    %1 = arith.extf %0 : tensor<512xf32, #blocked> to tensor<512xf64, #blocked>
    %2 = tt.splat %out0_ptr : !tt.ptr<f64> -> tensor<512x!tt.ptr<f64>, #blocked>
    %3 = tt.addptr %2, %in0_7 : tensor<512x!tt.ptr<f64>, #blocked>, tensor<512xi64, #blocked>
    %4 = ttg.convert_layout %3 : tensor<512x!tt.ptr<f64>, #blocked> -> tensor<512x!tt.ptr<f64>, #blocked2>
    %5 = ttg.convert_layout %1 : tensor<512xf64, #blocked> -> tensor<512xf64, #blocked2>
    %one = arith.constant dense<1> : tensor<512xi64, #blocked>
    %odd = arith.andi %in0_7, %one : tensor<512xi64, #blocked>
    %even = arith.cmpi eq, %odd, %cst : tensor<512xi64, #blocked>
    %6 = ttg.convert_layout %even : tensor<512xi1, #blocked> -> tensor<512xi1, #blocked2>
    tt.store %4, %5, %6 : tensor<512x!tt.ptr<f64>, #blocked2>
    tt.return
  }
}

// -----

// Component resolution: the histogram accumulator of the loop is resolved as
// a whole. Moving it onto the blocked parent synthesized from the loaded
// vector's layout makes the in-loop convert free (a register relabel) and
// leaves only the small reduced result to convert after the loop.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // ON-DAG: #[[$P:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // COMMON-LABEL: tt.func public @loop_histogram
  // OFF: scf.for
  // OFF: ttg.convert_layout %{{.*}} : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
  // ON: scf.for {{.*}} -> (tensor<16x1024xi64, #[[$P]]>)
  // ON: ttg.convert_layout %{{.*}} -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #[[$P]]}>>
  // ON: scf.yield
  // ON: tt.reduce
  // ON: ttg.convert_layout %{{.*}} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #[[$P]]}>>
  tt.func public @loop_histogram(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %zero = arith.constant dense<0> : tensor<16x1024xi64, #blocked>
    %fifteen = arith.constant dense<15> : tensor<1024xi16, #blocked1>
    %bins = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bins2 = tt.expand_dims %bins {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %binsb = tt.broadcast %bins2 : tensor<16x1xi32, #blocked> -> tensor<16x1024xi32, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked1>
    %acc = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%acc_i = %zero) -> (tensor<16x1024xi64, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<i16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<i16>, #blocked1>
      %key = arith.andi %x, %fifteen : tensor<1024xi16, #blocked1>
      %keyc = ttg.convert_layout %key : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %keyi = arith.extui %keyc : tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %key2 = tt.expand_dims %keyi {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
      %keyb = tt.broadcast %key2 : tensor<1x1024xi32, #blocked> -> tensor<16x1024xi32, #blocked>
      %hit = arith.cmpi eq, %binsb, %keyb : tensor<16x1024xi32, #blocked>
      %hit64 = arith.extui %hit : tensor<16x1024xi1, #blocked> to tensor<16x1024xi64, #blocked>
      %next = arith.addi %acc_i, %hit64 : tensor<16x1024xi64, #blocked>
      scf.yield %next : tensor<16x1024xi64, #blocked>
    }
    %count = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %s = arith.addi %a, %b : i64
      tt.reduce.return %s : i64
    }) : (tensor<16x1024xi64, #blocked>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %count : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Component resolution prices loop-carried values through scf.for only; a
// tensor carried by scf.while is left to the per-value resolution.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func public @while_carried
  // COMMON: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  // COMMON: scf.while ({{.*}}) : (tensor<64x64xf32, #blocked1>, i32)
  // COMMON: math.exp %{{.*}} : tensor<64x64xf32, #blocked1>
  // COMMON-NOT: ttg.convert_layout
  // COMMON: tt.store %{{.*}}, %{{.*}} : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.func public @while_carried(%a: tensor<64x64xf32, #blocked>, %b: tensor<64x64xf32, #blocked1>, %n: i32, %out: tensor<64x64x!tt.ptr<f32>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r:2 = scf.while (%x = %a, %i = %c0) : (tensor<64x64xf32, #blocked>, i32) -> (tensor<64x64xf32, #blocked>, i32) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %x, %i : tensor<64x64xf32, #blocked>, i32
    } do {
    ^bb0(%y: tensor<64x64xf32, #blocked>, %j: i32):
      %bc = ttg.convert_layout %b : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %s = arith.addf %y, %bc : tensor<64x64xf32, #blocked>
      %e = math.exp %s : tensor<64x64xf32, #blocked>
      %j1 = arith.addi %j, %c1 : i32
      scf.yield %e, %j1 : tensor<64x64xf32, #blocked>, i32
    }
    %rc = ttg.convert_layout %r#0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.store %out, %rc : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
