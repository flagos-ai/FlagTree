// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=0" | FileCheck %s --check-prefix=MASK0
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=8" | FileCheck %s --check-prefix=PHASE3

#blocked_atomic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_store = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // MASK0-LABEL: tt.func @atomic_writeback
  // MASK0: ttg.convert_layout
  // MASK0: tt.atomic_rmw
  // PHASE3-LABEL: tt.func @atomic_writeback
  // PHASE3-NOT: ttg.convert_layout
  // PHASE3: tt.atomic_rmw
  tt.func @atomic_writeback(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_atomic}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_atomic}>> -> tensor<16x1xi32, #blocked_atomic>
    %rows_full = tt.broadcast %rows_2d : tensor<16x1xi32, #blocked_atomic> -> tensor<16x64xi32, #blocked_atomic>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_atomic}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_atomic}>> -> tensor<1x64xi32, #blocked_atomic>
    %cols_full = tt.broadcast %cols_2d : tensor<1x64xi32, #blocked_atomic> -> tensor<16x64xi32, #blocked_atomic>
    %offsets = arith.addi %rows_full, %cols_full : tensor<16x64xi32, #blocked_atomic>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked_atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x64x!tt.ptr<f32>, #blocked_atomic>, tensor<16x64xi32, #blocked_atomic>
    %mask = arith.constant dense<true> : tensor<16x64xi1, #blocked_atomic>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #blocked_atomic>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #blocked_atomic>, tensor<16x64xf32, #blocked_atomic>, tensor<16x64xi1, #blocked_atomic>) -> tensor<16x64xf32, #blocked_atomic>
    tt.return
  }

  // MASK0-LABEL: tt.func @coalesced_store
  // MASK0: ttg.convert_layout
  // MASK0: tt.store
  // PHASE3-LABEL: tt.func @coalesced_store
  // PHASE3: ttg.convert_layout
  // PHASE3: tt.store
  tt.func @coalesced_store(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_store}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_store}>> -> tensor<16x1xi32, #blocked_store>
    %rows_full = tt.broadcast %rows_2d : tensor<16x1xi32, #blocked_store> -> tensor<16x64xi32, #blocked_store>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_store}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_store}>> -> tensor<1x64xi32, #blocked_store>
    %cols_full = tt.broadcast %cols_2d : tensor<1x64xi32, #blocked_store> -> tensor<16x64xi32, #blocked_store>
    %offsets = arith.addi %rows_full, %cols_full : tensor<16x64xi32, #blocked_store>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked_store>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x64x!tt.ptr<f32>, #blocked_store>, tensor<16x64xi32, #blocked_store>
    %mask = arith.constant dense<true> : tensor<16x64xi1, #blocked_store>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #blocked_store>
    tt.store %ptrs, %value, %mask : tensor<16x64x!tt.ptr<f32>, #blocked_store>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-cached-load-cost-per-byte" = 8 : i32, "ttg.rlc-convert-cost-per-byte" = 32 : i32, "ttg.rlc-convert-minimum-element-bits" = 32 : i32, "ttg.rlc-convert-minimum-elements" = 32 : i32, "ttg.rlc-expensive-math-cost-per-byte" = 8 : i32, "ttg.rlc-inter-warp-reduce-cost" = 8 : i32, "ttg.rlc-minimum-writeback-bits" = 128 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // PHASE3-LABEL: tt.func @backend_policy_minimum_writeback_bits
  // PHASE3: ttg.convert_layout
  // PHASE3: tt.atomic_rmw
  tt.func @backend_policy_minimum_writeback_bits(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>) {
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
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #blocked>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #blocked>, tensor<16x64xf32, #blocked>, tensor<16x64xi1, #blocked>) -> tensor<16x64xf32, #blocked>
    tt.return
  }
}
