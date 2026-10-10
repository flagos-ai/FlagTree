// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=UNGUARDED,VECTORIZED,UNSUPPORTED
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The dynamic stride makes the load non-coalesced, so Phase 2 may retag the
  // closed load/sitofp component to the wider store layout.
  // UNGUARDED-LABEL: tt.func @int_to_fp_retag_allowed
  // UNGUARDED-NOT: ttg.convert_layout
  // UNGUARDED: tt.store
  tt.func @int_to_fp_retag_allowed(%input: !tt.ptr<i32>, %output: tensor<256x!tt.ptr<f32>, #dst>, %stride: i32) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %stride_splat = tt.splat %stride : i32 -> tensor<256xi32, #src>
    %offset = arith.muli %index, %stride_splat : tensor<256xi32, #src>
    %input_splat = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #src>
    %input_ptrs = tt.addptr %input_splat, %offset : tensor<256x!tt.ptr<i32>, #src>, tensor<256xi32, #src>
    %loaded = tt.load %input_ptrs : tensor<256x!tt.ptr<i32>, #src>
    %value = arith.sitofp %loaded : tensor<256xi32, #src> to tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-int-to-fp-vector-width-mask" = 20 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // Both layouts own two elements per thread, so MUSA can keep the original
  // conversion width explicitly while the whole producer chain retags.
  // VECTORIZED-LABEL: tt.func @int_to_fp_retag_vector_width_preserved
  // VECTORIZED: arith.sitofp
  // VECTORIZED-SAME: {"ttg.rlc-preserve-int-to-fp-vector-width" = 2 : i32}
  // VECTORIZED-NOT: ttg.convert_layout
  // VECTORIZED: tt.store
  tt.func @int_to_fp_retag_vector_width_preserved(%input: !tt.ptr<i32>, %output: tensor<256x!tt.ptr<f32>, #dst>, %stride: i32) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %stride_splat = tt.splat %stride : i32 -> tensor<256xi32, #src>
    %offset = arith.muli %index, %stride_splat : tensor<256xi32, #src>
    %input_splat = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #src>
    %input_ptrs = tt.addptr %input_splat, %offset : tensor<256x!tt.ptr<i32>, #src>, tensor<256xi32, #src>
    %loaded = tt.load %input_ptrs : tensor<256x!tt.ptr<i32>, #src>
    %value = arith.sitofp %loaded : tensor<256xi32, #src> to tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-int-to-fp-vector-width-mask" = 20 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The supported i32 -> f32 conversion and its exact i1 tails all retag to
  // #dst. The marker makes the backend lowering preserve two-wide conversion.
  // VECTORIZED-LABEL: tt.func @int_to_fp_vector_width_keeps_profitable_tail
  // VECTORIZED: %[[FLOAT:.*]] = arith.sitofp {{.*}} {"ttg.rlc-preserve-int-to-fp-vector-width" = 2 : i32} : tensor<256xi32, #blocked> to tensor<256xf32, #blocked>
  // VECTORIZED: %[[PRED:.*]] = arith.cmpf ogt, %[[FLOAT]], {{.*}} : tensor<256xf32, #blocked>
  // VECTORIZED: arith.uitofp %[[PRED]] : tensor<256xi1, #blocked> to tensor<256xf32, #blocked>
  // VECTORIZED-NOT: ttg.convert_layout
  // VECTORIZED: tt.store
  // VECTORIZED: arith.extui %[[PRED]] : tensor<256xi1, #blocked> to tensor<256xi8, #blocked>
  // VECTORIZED-NOT: ttg.convert_layout
  // VECTORIZED: tt.store
  tt.func @int_to_fp_vector_width_keeps_profitable_tail(%input: !tt.ptr<i32>, %value_output: tensor<256x!tt.ptr<f32>, #dst>, %mask_output: tensor<256x!tt.ptr<i8>, #dst>, %stride: i32, %threshold: f32) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %stride_splat = tt.splat %stride : i32 -> tensor<256xi32, #src>
    %offset = arith.muli %index, %stride_splat : tensor<256xi32, #src>
    %input_splat = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #src>
    %input_ptrs = tt.addptr %input_splat, %offset : tensor<256x!tt.ptr<i32>, #src>, tensor<256xi32, #src>
    %loaded = tt.load %input_ptrs : tensor<256x!tt.ptr<i32>, #src>
    %value = arith.sitofp %loaded : tensor<256xi32, #src> to tensor<256xf32, #src>
    %threshold_splat = tt.splat %threshold : f32 -> tensor<256xf32, #src>
    %predicate = arith.cmpf ogt, %value, %threshold_splat : tensor<256xf32, #src>
    %float_mask = arith.uitofp %predicate : tensor<256xi1, #src> to tensor<256xf32, #src>
    %float_converted = ttg.convert_layout %float_mask : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %value_output, %float_converted : tensor<256x!tt.ptr<f32>, #dst>
    %byte_mask = arith.extui %predicate : tensor<256xi1, #src> to tensor<256xi8, #src>
    %byte_converted = ttg.convert_layout %byte_mask : tensor<256xi8, #src> -> tensor<256xi8, #dst>
    tt.store %mask_output, %byte_converted : tensor<256x!tt.ptr<i8>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-int-to-fp-vector-width-mask" = 20 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The old layout owns one element per thread while the proposed layout owns
  // two, so no supported vector-width contract exists and the convert stays.
  // UNSUPPORTED-LABEL: tt.func @int_to_fp_unsupported_width_stays_guarded
  // UNSUPPORTED: arith.sitofp {{.*}} : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
  // UNSUPPORTED: ttg.convert_layout
  // UNSUPPORTED: tt.store
  tt.func @int_to_fp_unsupported_width_stays_guarded(%input: !tt.ptr<i32>, %output: tensor<128x!tt.ptr<f32>, #dst>, %stride: i32) {
    %index = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #src>
    %stride_splat = tt.splat %stride : i32 -> tensor<128xi32, #src>
    %offset = arith.muli %index, %stride_splat : tensor<128xi32, #src>
    %input_splat = tt.splat %input : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #src>
    %input_ptrs = tt.addptr %input_splat, %offset : tensor<128x!tt.ptr<i32>, #src>, tensor<128xi32, #src>
    %loaded = tt.load %input_ptrs : tensor<128x!tt.ptr<i32>, #src>
    %value = arith.sitofp %loaded : tensor<128xi32, #src> to tensor<128xf32, #src>
    %converted = ttg.convert_layout %value : tensor<128xf32, #src> -> tensor<128xf32, #dst>
    tt.store %output, %converted : tensor<128x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=preserve reason=int-to-fp-vector-width
// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=preserve reason=int-to-fp-contiguity-boundary
