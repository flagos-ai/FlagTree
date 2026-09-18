// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=0" | FileCheck %s --check-prefix=MASK0
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=1" | FileCheck %s --check-prefix=PHASE1A
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=3" | FileCheck %s --check-prefix=PHASE1B
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefix=PHASE2
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=8" | FileCheck %s --check-prefix=PHASE3

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#phase1b_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#phase1b_dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#phase2_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#phase2_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#wmma = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // A shared expensive producer keeps the baseline rematerializer from
  // duplicating the chain. Phase 1b can instead retag the producer/writeback
  // component and eliminate both blocked-to-blocked conversions.
  // PHASE1A-LABEL: tt.func @phase1b_blocked_writeback
  // PHASE1A: ttg.convert_layout
  // PHASE1B-LABEL: tt.func @phase1b_blocked_writeback
  // PHASE1B-NOT: ttg.convert_layout
  // PHASE1B: tt.return
  tt.func @phase1b_blocked_writeback(%base: !tt.ptr<f32>, %seed: i32) {
    %root = tt.splat %seed : i32 -> tensor<256xi32, #phase1b_src>
    %root_f = arith.sitofp %root : tensor<256xi32, #phase1b_src> to tensor<256xf32, #phase1b_src>
    %shared0 = math.exp2 %root_f : tensor<256xf32, #phase1b_src>
    %shared1 = math.log2 %shared0 : tensor<256xf32, #phase1b_src>
    %shared2 = math.sin %shared1 : tensor<256xf32, #phase1b_src>
    %shared3 = math.cos %shared2 : tensor<256xf32, #phase1b_src>
    %one = arith.constant dense<1.0> : tensor<256xf32, #phase1b_src>
    %left = arith.addf %shared3, %one : tensor<256xf32, #phase1b_src>
    %right = arith.mulf %shared3, %one : tensor<256xf32, #phase1b_src>
    %left_out = ttg.convert_layout %left : tensor<256xf32, #phase1b_src> -> tensor<256xf32, #phase1b_dst>
    %right_out = ttg.convert_layout %right : tensor<256xf32, #phase1b_src> -> tensor<256xf32, #phase1b_dst>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #phase1b_dst>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #phase1b_dst>
    %left_ptrs = tt.addptr %base_splat, %offsets : tensor<256x!tt.ptr<f32>, #phase1b_dst>, tensor<256xi32, #phase1b_dst>
    %c256 = arith.constant dense<256> : tensor<256xi32, #phase1b_dst>
    %right_offsets = arith.addi %offsets, %c256 : tensor<256xi32, #phase1b_dst>
    %right_ptrs = tt.addptr %base_splat, %right_offsets : tensor<256x!tt.ptr<f32>, #phase1b_dst>, tensor<256xi32, #phase1b_dst>
    tt.store %left_ptrs, %left_out : tensor<256x!tt.ptr<f32>, #phase1b_dst>
    tt.store %right_ptrs, %right_out : tensor<256x!tt.ptr<f32>, #phase1b_dst>
    tt.return
  }

  // The input is scattered along phase2_src's fastest dimension. Phase 2 may
  // retag the closed producer component to phase2_dst without degrading a
  // coalesced load; Phase 1b deliberately leaves the load boundary untouched.
  // PHASE1B-LABEL: tt.func @phase2_scatter_load_component
  // PHASE1B: ttg.convert_layout
  // PHASE2-LABEL: tt.func @phase2_scatter_load_component
  // PHASE2-NOT: ttg.convert_layout
  // PHASE2: tt.return
  tt.func @phase2_scatter_load_component(%input: !tt.ptr<f16>, %output: tensor<64x16x!tt.ptr<f16>, #phase2_dst>, %stride: i32) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #phase2_src}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #phase2_src}>> -> tensor<64x1xi32, #phase2_src>
    %rows_full = tt.broadcast %rows_2d : tensor<64x1xi32, #phase2_src> -> tensor<64x16xi32, #phase2_src>
    %stride_full = tt.splat %stride : i32 -> tensor<64x16xi32, #phase2_src>
    %row_offsets = arith.muli %rows_full, %stride_full : tensor<64x16xi32, #phase2_src>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #phase2_src}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #phase2_src}>> -> tensor<1x16xi32, #phase2_src>
    %cols_full = tt.broadcast %cols_2d : tensor<1x16xi32, #phase2_src> -> tensor<64x16xi32, #phase2_src>
    %offsets = arith.addi %row_offsets, %cols_full : tensor<64x16xi32, #phase2_src>
    %input_splat = tt.splat %input : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #phase2_src>
    %input_ptrs = tt.addptr %input_splat, %offsets : tensor<64x16x!tt.ptr<f16>, #phase2_src>, tensor<64x16xi32, #phase2_src>
    %value = tt.load %input_ptrs : tensor<64x16x!tt.ptr<f16>, #phase2_src>
    %converted = ttg.convert_layout %value : tensor<64x16xf16, #phase2_src> -> tensor<64x16xf16, #phase2_dst>
    tt.store %output, %converted : tensor<64x16x!tt.ptr<f16>, #phase2_dst>
    tt.return
  }

  // MASK0-LABEL: tt.func @musa_wmma_atomic_writeback
  // MASK0: ttg.convert_layout
  // MASK0: tt.atomic_rmw
  // PHASE3-LABEL: tt.func @musa_wmma_atomic_writeback
  // PHASE3-NOT: ttg.convert_layout
  // PHASE3: tt.atomic_rmw
  tt.func @musa_wmma_atomic_writeback(%acc: tensor<16x64xf32, #wmma>, %base: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %rows_full = tt.broadcast %rows_2d : tensor<16x1xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %cols_full = tt.broadcast %cols_2d : tensor<1x64xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %offsets = arith.addi %rows_full, %cols_full : tensor<16x64xi32, #blocked>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x64x!tt.ptr<f32>, #blocked>, tensor<16x64xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<16x64xi1, #blocked>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #wmma> -> tensor<16x64xf32, #blocked>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #blocked>, tensor<16x64xf32, #blocked>, tensor<16x64xi1, #blocked>) -> tensor<16x64xf32, #blocked>
    tt.return
  }

  // MASK0-LABEL: tt.func @musa_sqmma_atomic_writeback
  // MASK0: ttg.convert_layout
  // MASK0: tt.atomic_rmw
  // PHASE3-LABEL: tt.func @musa_sqmma_atomic_writeback
  // PHASE3-NOT: ttg.convert_layout
  // PHASE3: tt.atomic_rmw
  tt.func @musa_sqmma_atomic_writeback(%acc: tensor<16x64xf32, #sqmma>, %base: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %rows_full = tt.broadcast %rows_2d : tensor<16x1xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %cols_full = tt.broadcast %cols_2d : tensor<1x64xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %offsets = arith.addi %rows_full, %cols_full : tensor<16x64xi32, #blocked>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x64x!tt.ptr<f32>, #blocked>, tensor<16x64xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<16x64xi1, #blocked>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #sqmma> -> tensor<16x64xf32, #blocked>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #blocked>, tensor<16x64xf32, #blocked>, tensor<16x64xi1, #blocked>) -> tensor<16x64xf32, #blocked>
    tt.return
  }
}
