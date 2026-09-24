// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups --verify-diagnostics | FileCheck %s

// A used tensor capture pins partition2 to its original physical owner range.
// The unused tensor arguments do not prevent the other partitions from using
// ordinary WS largest-first assignment, which aligns the four-warp partition.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#pair = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @pinned_capture
  tt.func @pinned_capture(%src: tensor<256xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c3 = arith.constant 3 : i32
    %last = tle.extract_tile %src[%c3] {tile_shape = array<i64: 64>} : tensor<256xi32, #full>, i32 -> tensor<64xi32, #pair>
    // CHECK: ttg.warp_specialize({{.*}}) attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 4, 0, 6>}
    ttg.warp_specialize(%last, %ptr) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) {
      ttg.warp_return
    }
    partition1(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(4) {
      ttg.warp_return
    }
    partition2(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %pointers, %v : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    : (tensor<64xi32, #pair>, !tt.ptr<i32>) -> ()
    tt.return
  }
}

// -----

// Pinned [0, 2) and [6, 8) owners leave four free warps, but no aligned
// four-warp range. Moving either captured register tuple would lose data.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#pair = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fragmented_owners(%src: tensor<256xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c3 = arith.constant 3 : i32
    %first = tle.extract_tile %src[%c0] {tile_shape = array<i64: 64>} : tensor<256xi32, #full>, i32 -> tensor<64xi32, #pair>
    %last = tle.extract_tile %src[%c3] {tile_shape = array<i64: 64>} : tensor<256xi32, #full>, i32 -> tensor<64xi32, #pair>
    // No WGMMA: the unaligned middle range remains legal between pinned views.
    // CHECK: ttg.warp_specialize{{.*}}warpGroupStartIds = array<i32: 0, 2, 6>
    ttg.warp_specialize(%first, %last, %ptr) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %pointers, %a : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    partition1(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(4) {
      ttg.warp_return
    }
    partition2(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %pointers, %b : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    : (tensor<64xi32, #pair>, tensor<64xi32, #pair>, !tt.ptr<i32>) -> ()
    tt.return
  }
}

// -----

// Explicit physical assignments must still form a disjoint CTA partition.
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @overlapping_ranges() {
    // expected-error @+1 {{reuse partition #1 overlaps another physical warp range}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 0, 6>}
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    partition1() num_warps(4) { ttg.warp_return }
    partition2() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @range_beyond_cta() {
    // expected-error @+1 {{reuse partition #2 physical warp range is outside the CTA}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 2, 7>}
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    partition1() num_warps(4) { ttg.warp_return }
    partition2() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @negative_start() {
    // expected-error @+1 {{reuse partition #0 physical warp range is outside the CTA}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: -1, 2, 6>}
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    partition1() num_warps(4) { ttg.warp_return }
    partition2() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

// An otherwise disjoint, aligned reassignment cannot move used registers.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#pair = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @moved_capture(%src: tensor<256xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %first = tle.extract_tile %src[%c0] {tile_shape = array<i64: 64>} : tensor<256xi32, #full>, i32 -> tensor<64xi32, #pair>
    // expected-error @+1 {{reuse partition #0 with tensor captures must start at warp 0}}
    ttg.warp_specialize(%first, %ptr) attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 4, 0, 6>}
    default { ttg.warp_yield }
    partition0(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %pointers, %v : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    partition1(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(4) { ttg.warp_return }
    partition2(%v: tensor<64xi32, #pair>, %p: !tt.ptr<i32>) num_warps(2) { ttg.warp_return }
    : (tensor<64xi32, #pair>, !tt.ptr<i32>) -> ()
    tt.return
  }
}

// -----


// Externally supplied physical assignments must also satisfy the WGMMA
// hardware requirement: a group of four warps begins at a multiple of four.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.target = "cuda:90"} {
  tt.func @unaligned_wgmma(%a: !ttg.memdesc<64x32xf16, #shared, #smem>, %b: !ttg.memdesc<32x64xf16, #shared1, #smem>) {
    ttg.warp_specialize(%a, %b) attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 2, 6>}
    default { ttg.warp_yield }
    partition0(%pa: !ttg.memdesc<64x32xf16, #shared, #smem>, %pb: !ttg.memdesc<32x64xf16, #shared1, #smem>) num_warps(2) { ttg.warp_return }
    partition1(%pa: !ttg.memdesc<64x32xf16, #shared, #smem>, %pb: !ttg.memdesc<32x64xf16, #shared1, #smem>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
      // expected-error @+1 {{requires a physical warp group start aligned to 4; reuse partition #1 starts at 2}}
      %dot = ttng.warp_group_dot %pa, %pb, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x64xf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      ttg.warp_return
    }
    partition2(%pa: !ttg.memdesc<64x32xf16, #shared, #smem>, %pb: !ttg.memdesc<32x64xf16, #shared1, #smem>) num_warps(2) { ttg.warp_return }
    : (!ttg.memdesc<64x32xf16, #shared, #smem>, !ttg.memdesc<32x64xf16, #shared1, #smem>) -> ()
    tt.return
  }
}

// -----

// Fixed owners [0, 2), [10, 12), and [12, 16) leave [2, 10) free.
// Placing the first four-warp partition at aligned [4, 8) fragments the
// remaining space and prevents the other four-warp partition from fitting.
// Restore the original valid ranges [2, 6), [6, 10) instead of rejecting.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#pair = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#quad = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 16 : i32} {
  // CHECK-LABEL: tt.func @fragmented_by_aligned_first_fit
  tt.func @fragmented_by_aligned_first_fit(%src: tensor<512xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %first = tle.extract_tile %src[%c0] {tile_shape = array<i64: 64>} : tensor<512xi32, #full>, i32 -> tensor<64xi32, #pair>
    %middle = tle.extract_tile %src[%c5] {tile_shape = array<i64: 64>} : tensor<512xi32, #full>, i32 -> tensor<64xi32, #pair>
    %last = tle.extract_tile %src[%c3] {tile_shape = array<i64: 128>} : tensor<512xi32, #full>, i32 -> tensor<128xi32, #quad>
    // CHECK: ttg.warp_specialize{{.*}}warpGroupStartIds = array<i32: 0, 2, 6, 10, 12>
    ttg.warp_specialize(%first, %middle, %last, %ptr) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %c: tensor<128xi32, #quad>, %p: !tt.ptr<i32>) num_warps(2) {
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %ps, %a : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    partition1(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %c: tensor<128xi32, #quad>, %p: !tt.ptr<i32>) num_warps(4) { ttg.warp_return }
    partition2(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %c: tensor<128xi32, #quad>, %p: !tt.ptr<i32>) num_warps(4) { ttg.warp_return }
    partition3(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %c: tensor<128xi32, #quad>, %p: !tt.ptr<i32>) num_warps(2) {
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #pair>
      tt.store %ps, %b : tensor<64x!tt.ptr<i32>, #pair>
      ttg.warp_return
    }
    partition4(%a: tensor<64xi32, #pair>, %b: tensor<64xi32, #pair>, %c: tensor<128xi32, #quad>, %p: !tt.ptr<i32>) num_warps(4) {
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #quad>
      tt.store %ps, %c : tensor<128x!tt.ptr<i32>, #quad>
      ttg.warp_return
    }
    : (tensor<64xi32, #pair>, tensor<64xi32, #pair>, tensor<128xi32, #quad>, !tt.ptr<i32>) -> ()
    tt.return
  }
}
