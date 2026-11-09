// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=false" | FileCheck %s --check-prefix=CONTRACT
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=CONTRACT,P2
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=8" | FileCheck %s --check-prefixes=CONTRACT,P3
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=15" | FileCheck %s --check-prefixes=CONTRACT,P3
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE
// Also run by python/test/unit/runtime/test_musa_rlc_pass.py using libtriton.

// vendor-contract
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, tle.enable_encoding_rematerialization} {
  // CONTRACT-LABEL: tt.func @explicit_sqmma_store(
  // CONTRACT: %[[VALUE:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #blocked}
  // CONTRACT: tt.store {{.*}}, %[[VALUE]], {{.*}} : tensor<16x64x!tt.ptr<f32>, #blocked>
  tt.func @explicit_sqmma_store(%acc: tensor<16x64xf32, #sqmma>, %ptrs: tensor<16x64x!tt.ptr<f32>, #blocked>, %mask: tensor<16x64xi1, #blocked>) {
    %value = ttg.convert_layout %acc {"tle.explicit_encoding.0" = #blocked} : tensor<16x64xf32, #sqmma> -> tensor<16x64xf32, #blocked>
    tt.store %ptrs, %value, %mask : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
 // CONTRACT-LABEL: tt.func @volatile_duplicate(
 // CONTRACT: tt.load {{.*}} {isVolatile = true}
 // CONTRACT-NOT: tt.load
 // CONTRACT: tt.return
 tt.func @volatile_duplicate(%base: !tt.ptr<f32>) -> (tensor<64xf32, #src>, tensor<64xf32, #dst>) {
  %i = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #src>
  %b = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #src>
  %p = tt.addptr %b, %i : tensor<64x!tt.ptr<f32>, #src>, tensor<64xi32, #src>
  %v = tt.load %p {isVolatile = true} : tensor<64x!tt.ptr<f32>, #src>
  %c = ttg.convert_layout %v : tensor<64xf32, #src> -> tensor<64xf32, #dst>
  tt.return %v, %c : tensor<64xf32, #src>, tensor<64xf32, #dst>
 }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
 // CONTRACT-LABEL: tt.func @repeated_address_remat(
 // CONTRACT: tt.load
 // CONTRACT: tt.load
 // CONTRACT-NOT: ttg.convert_layout
 // CONTRACT: tt.return
 tt.func @repeated_address_remat(%base: !tt.ptr<f32>) -> (tensor<256xf32, #src>, tensor<256xf32, #dst>) {
  %b = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #src>
  %v = tt.load %b : tensor<256x!tt.ptr<f32>, #src>
  %c = ttg.convert_layout %v : tensor<256xf32, #src> -> tensor<256xf32, #dst>
  tt.return %v, %c : tensor<256xf32, #src>, tensor<256xf32, #dst>
 }
}

// -----

// tle-sqmma
// The established TLE SQMMA writeback adapter is independent of RLC enhancement.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, tle.enable_encoding_rematerialization} {
  // CONTRACT-LABEL: tt.func @sqmma_store(
  // CONTRACT-SAME: %[[ACC:[a-zA-Z0-9_]+]]: tensor<16x64xf32, #{{[^>]+}}>
  // CONTRACT-NOT: ttg.convert_layout %[[ACC]]
  // CONTRACT: tt.store {{.*}}, %[[ACC]], {{.*}} : tensor<16x64x!tt.ptr<f32>, #{{[^>]+}}>
  tt.func @sqmma_store(%acc: tensor<16x64xf32, #sqmma>, %ptrs: tensor<16x64x!tt.ptr<f32>, #blocked>, %mask: tensor<16x64xi1, #blocked>) {
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #sqmma> -> tensor<16x64xf32, #blocked>
    tt.store %ptrs, %value, %mask : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// explicit-control-flow
// Explicit layouts remain binding across if yields and loop-carried values.
// CONTRACT-DAG: #[[DST:[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2],
#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31", tle.enable_encoding_rematerialization} {
  // CONTRACT-LABEL: tt.func @explicit_control_flow(
  // CONTRACT: scf.for
  // CONTRACT: scf.if
  // CONTRACT: %[[THEN:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CONTRACT: scf.yield %[[THEN]] : tensor<256xf32, #[[DST]]>
  // CONTRACT: } else {
  // CONTRACT: %[[ELSE:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CONTRACT: scf.yield %[[ELSE]] : tensor<256xf32, #[[DST]]>
  // CONTRACT: scf.while
  // CONTRACT: scf.condition
  // CONTRACT: %[[BODY:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CONTRACT: tt.return
  tt.func @explicit_control_flow(%input: tensor<256xf32, #src>, %other: tensor<256xf32, #src>, %cond: i1, %n: index) -> tensor<256xf32, #src> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %i = %c0 to %n step %c1 iter_args(%carry = %input) -> tensor<256xf32, #src> {
      %chosen = scf.if %cond -> tensor<256xf32, #dst> {
        %a = ttg.convert_layout %carry {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
        scf.yield %a : tensor<256xf32, #dst>
      } else {
        %b = ttg.convert_layout %other {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
        scf.yield %b : tensor<256xf32, #dst>
      }
      %back = ttg.convert_layout %chosen : tensor<256xf32, #dst> -> tensor<256xf32, #src>
      scf.yield %back : tensor<256xf32, #src>
    }
    %result = scf.while (%carry = %loop) : (tensor<256xf32, #src>) -> tensor<256xf32, #src> {
      scf.condition(%cond) %carry : tensor<256xf32, #src>
    } do {
    ^bb0(%carry: tensor<256xf32, #src>):
      %explicit = ttg.convert_layout %carry {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
      %back = ttg.convert_layout %explicit : tensor<256xf32, #dst> -> tensor<256xf32, #src>
      scf.yield %back : tensor<256xf32, #src>
    }
    tt.return %result : tensor<256xf32, #src>
  }
}

// -----

// phase2-atomic-policy
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // P2-LABEL: tt.func @atomic_tail_allowed
  // P2-NOT: ttg.convert_layout
  // P2: tt.atomic_rmw
  tt.func @atomic_tail_allowed(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-atomic-writeback-max-elements-per-thread-ratio" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // P2-LABEL: tt.func @atomic_tail_policy_preserves
  // P2: ttg.convert_layout
  // P2: tt.atomic_rmw
  tt.func @atomic_tail_policy_preserves(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

// phase2-int-to-fp-policy
#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The dynamic stride makes the load non-coalesced, so Phase 2 may retag the
  // closed load/sitofp component to the wider store layout.
  // P2-LABEL: tt.func @int_to_fp_retag_allowed
  // P2-NOT: ttg.convert_layout
  // P2: tt.store
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
  // P2-LABEL: tt.func @int_to_fp_retag_vector_width_preserved
  // P2: arith.sitofp
  // P2-SAME: {"ttg.rlc-preserve-int-to-fp-vector-width" = 2 : i32}
  // P2-NOT: ttg.convert_layout
  // P2: tt.store
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
  // P2-LABEL: tt.func @int_to_fp_vector_width_keeps_profitable_tail
  // P2: %[[FLOAT:.*]] = arith.sitofp {{.*}} {"ttg.rlc-preserve-int-to-fp-vector-width" = 2 : i32} : tensor<256xi32, #blocked> to tensor<256xf32, #blocked>
  // P2: %[[PRED:.*]] = arith.cmpf ogt, %[[FLOAT]], {{.*}} : tensor<256xf32, #blocked>
  // P2: arith.uitofp %[[PRED]] : tensor<256xi1, #blocked> to tensor<256xf32, #blocked>
  // P2-NOT: ttg.convert_layout
  // P2: tt.store
  // P2: arith.extui %[[PRED]] : tensor<256xi1, #blocked> to tensor<256xi8, #blocked>
  // P2-NOT: ttg.convert_layout
  // P2: tt.store
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
  // P2-LABEL: tt.func @int_to_fp_unsupported_width_stays_guarded
  // P2: arith.sitofp {{.*}} : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
  // P2: ttg.convert_layout
  // P2: tt.store
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

// -----

// writeback-effects
#b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
  // An impure address producer must execute only once, keeping the value convert.
  // P3-LABEL: tt.func @impure_address
  // P3: tt.elementwise_inline_asm
  // P3-SAME: pure = false
  // P3-NOT: tt.elementwise_inline_asm
  // P3: ttg.convert_layout
  // P3-NOT: tt.elementwise_inline_asm
  // P3: tt.atomic_rmw
  // P3-NOT: tt.elementwise_inline_asm
  // P3: tt.return
  tt.func @impure_address(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %offset: i32) {
    %x = tt.splat %offset : i32 -> tensor<16x64xi32, #b>
    %i = tt.elementwise_inline_asm "mov $0, $1" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %x : tensor<16x64xi32, #b> -> tensor<16x64xi32, #b>
    %p = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %p, %i : tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xi32, #b>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }

  // The same chain marked pure still permits writeback rematerialization.
  // P3-LABEL: tt.func @pure_address
  // P3-NOT: ttg.convert_layout
  // P3: tt.elementwise_inline_asm
  // P3-SAME: pure = true
  // P3-NOT: ttg.convert_layout
  // P3: tt.atomic_rmw
  // P3-NOT: ttg.convert_layout
  // P3: tt.return
  tt.func @pure_address(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %offset: i32) {
    %x = tt.splat %offset : i32 -> tensor<16x64xi32, #b>
    %i = tt.elementwise_inline_asm "mov $0, $1" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %x : tensor<16x64xi32, #b> -> tensor<16x64xi32, #b>
    %p = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %p, %i : tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xi32, #b>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }
}
