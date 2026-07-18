// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>

module attributes {tle.enable_encoding_rematerialization, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_mma_state_without_layout_staging
  // CHECK: tt.make_range {{.*}} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
  // CHECK: tt.make_range {{.*}} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x32x!tt.ptr<f32>, #mma>
  tt.func @store_mma_state_without_layout_staging(%base: !tt.ptr<f32>, %state: tensor<128x32xf32, #mma>, %block_v: i32, %dv: i32) {
    %row_stride = arith.constant dense<32> : tensor<128x1xi32, #blocked>
    %row = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %row_offset = arith.muli %row2d, %row_stride : tensor<128x1xi32, #blocked>
    %base2d = tt.splat %base : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %row_ptr = tt.addptr %base2d, %row_offset : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %row_ptr_b = tt.broadcast %row_ptr : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x32x!tt.ptr<f32>, #blocked>

    %block = tt.splat %block_v : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col_local = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col = arith.addi %col_local, %block : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_b = tt.broadcast %col2d : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %ptr = tt.addptr %row_ptr_b, %col_b : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>

    %dv_s = tt.splat %dv : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask1 = arith.cmpi slt, %col, %dv_s : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask2d = tt.expand_dims %mask1 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
    %mask = tt.broadcast %mask2d : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>

    %state_blocked = ttg.convert_layout %state : tensor<128x32xf32, #mma> -> tensor<128x32xf32, #blocked>
    tt.store %ptr, %state_blocked, %mask : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tle.enable_encoding_rematerialization, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @local_store_source_keeps_blocked_across_aliasing_write
  // CHECK: ttg.local_store %{{.*}}, %{{.*}} : tensor<64x64xf32, #mma>
  // CHECK: ttg.local_store %{{.*}}, %{{.*}} : tensor<64x64xbf16, #blocked>
  // CHECK-NOT: ttg.local_store %{{.*}}, %{{.*}} : tensor<64x64xbf16, #mma>
  tt.func @local_store_source_keeps_blocked_across_aliasing_write(
      %src: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %dst: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>) {
    %blocked_load = ttg.local_load %src : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    %one_blocked = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %blocked_scaled = arith.mulf %blocked_load, %one_blocked : tensor<64x64xf32, #blocked>

    %mma_load = ttg.local_load %src : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #mma>
    %one_mma = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #mma>
    %mma_scaled = arith.mulf %mma_load, %one_mma : tensor<64x64xf32, #mma>
    ttg.local_store %mma_scaled, %src : tensor<64x64xf32, #mma> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>

    %blocked_exp = math.exp2 %blocked_scaled : tensor<64x64xf32, #blocked>
    %blocked_sum = arith.addf %blocked_exp, %one_blocked : tensor<64x64xf32, #blocked>
    %blocked_store_value = arith.truncf %blocked_sum : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    ttg.local_store %blocked_store_value, %dst : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tle.enable_encoding_rematerialization, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @local_store_source_reuses_available_mma_subtree
  // CHECK: math.exp2 {{.*}} : tensor<64x64xf32, #mma>
  // CHECK: arith.addf {{.*}} : tensor<64x64xf32, #mma>
  // CHECK: arith.truncf {{.*}} : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
  // CHECK: ttg.local_store %{{.*}}, %{{.*}} : tensor<64x64xbf16, #mma>
  // CHECK-NOT: ttg.local_store %{{.*}}, %{{.*}} : tensor<64x64xbf16, #blocked>
  tt.func @local_store_source_reuses_available_mma_subtree(
      %src: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %dst: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>,
      %sink: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
    %blocked_load = ttg.local_load %src : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    %one_blocked = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %blocked_scaled = arith.mulf %blocked_load, %one_blocked : tensor<64x64xf32, #blocked>

    %mma_load = ttg.local_load %src : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #mma>
    %one_mma = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #mma>
    %mma_scaled = arith.mulf %mma_load, %one_mma : tensor<64x64xf32, #mma>

    %blocked_exp = math.exp2 %blocked_scaled : tensor<64x64xf32, #blocked>
    %blocked_sum = arith.addf %blocked_exp, %one_blocked : tensor<64x64xf32, #blocked>
    %blocked_store_value = arith.truncf %blocked_sum : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    ttg.local_store %blocked_store_value, %dst : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    ttg.local_store %mma_scaled, %sink : tensor<64x64xf32, #mma> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}
