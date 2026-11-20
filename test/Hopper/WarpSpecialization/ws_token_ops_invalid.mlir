// RUN: triton-opt %s -split-input-file -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_num_buffers() {
    // expected-error @+1 {{requires positive numBuffers}}
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 0 : i32} : tensor<0x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_full_count() {
    // expected-error @+1 {{requires positive full_count}}
    %token = nvws.create_token {full_count = 0 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_negative_empty_count() {
    // expected-error @+1 {{requires positive empty_count}}
    %token = nvws.create_token {empty_count = -1 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_release_count() {
    %idx = arith.constant 0 : i32
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    // expected-error @+1 {{requires positive release_count}}
    nvws.consumer_release %token, %idx {release_count = 0 : i32} : tensor<2x!nvws.token>, i32
    tt.return
  }
}
