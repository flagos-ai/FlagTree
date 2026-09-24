// RUN: triton-opt %s --verify-diagnostics

// Nested register views keep the physical owner range of their root source.
// The second extraction is relative to `%bottom`, so its identity index must
// preserve bottom's [2, 4) owner range instead of resetting it to [0, 2).
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @nested_slice(%src: tensor<128xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bottom = tle.extract_tile %src[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    %identity = tle.extract_tile %bottom[%c0] {tile_shape = array<i64: 64>} : tensor<64xi32, #half>, i32 -> tensor<64xi32, #half>
    ttg.warp_specialize(%identity, %ptr) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%v: tensor<64xi32, #half>, %p: !tt.ptr<i32>) num_warps(2) {
      ttg.warp_return
    }
    partition1(%v: tensor<64xi32, #half>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #half>
      tt.store %pointers, %v : tensor<64x!tt.ptr<i32>, #half>
      ttg.warp_return
    }
    : (tensor<64xi32, #half>, !tt.ptr<i32>) -> ()
    tt.return
  }

  // The same nested value belongs to [2, 4), so using it in partition0 must
  // be rejected as an ownership mismatch.
  tt.func @wrong_partition(%src: tensor<128xi32, #full>, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bottom = tle.extract_tile %src[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    %identity = tle.extract_tile %bottom[%c0] {tile_shape = array<i64: 64>} : tensor<64xi32, #half>, i32 -> tensor<64xi32, #half>
    // expected-error @+1 {{warp slice ownership does not match reuse partition #0}}
    ttg.warp_specialize(%identity, %ptr) attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0(%v: tensor<64xi32, #half>, %p: !tt.ptr<i32>) num_warps(2) {
      %pointers = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #half>
      tt.store %pointers, %v : tensor<64x!tt.ptr<i32>, #half>
      ttg.warp_return
    }
    partition1(%v: tensor<64xi32, #half>, %p: !tt.ptr<i32>) num_warps(2) { ttg.warp_return }
    : (tensor<64xi32, #half>, !tt.ptr<i32>) -> ()
    tt.return
  }
}
