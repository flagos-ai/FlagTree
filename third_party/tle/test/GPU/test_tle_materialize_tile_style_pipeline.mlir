// RUN: triton-opt %s -triton-tle-materialize-tile-style-pipeline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @candidate_loop
  // CHECK: scf.for
  // CHECK-NOT: tt.num_stages
  // CHECK-NOT: tt.scheduled_max_stage
  // CHECK: } {tle.async_tile_producer_count = 0 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  tt.func @candidate_loop() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %v = arith.index_cast %i : index to i32
      scf.yield
    } {loop.stage = 2 : i32, tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @fallback_loop
  // CHECK: scf.for
  // CHECK: } {tt.num_stages = 2 : i32}
  // CHECK-NOT: tle.explicit_tile_style_pipeline
  tt.func @fallback_loop() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.yield
    } {tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @fallback_loop_with_region
  // CHECK: scf.for
  // CHECK-NOT: tt.num_stages
  // CHECK: } {tle.async_tile_producer_count = 0 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  tt.func @fallback_loop_with_region() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %c0 to %c2 step %c1 {
      scf.if %true {
      }
      scf.yield
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @fallback_loop_with_reduce
  // CHECK: "tt.reduce"
  // CHECK-NOT: tt.num_stages
  // CHECK: } {tle.async_tile_producer_count = 0 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  tt.func @fallback_loop_with_reduce(%arg0: tensor<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %v = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg0) -> (tensor<4x4xf32>) {
      %r = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %c = arith.addf %a, %b : f32
        tt.reduce.return %c : f32
      }) : (tensor<4x4xf32>) -> tensor<4xf32>
      %e = tt.expand_dims %r {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
      %b = tt.broadcast %e : tensor<4x1xf32> -> tensor<4x4xf32>
      scf.yield %b : tensor<4x4xf32>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @loop_with_async_tile_producer
  // CHECK: scf.if
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: scf.for {{.*}} iter_args({{.*}}!ttg.async.token, {{.*}}i32, {{.*}}!ttg.async.token, {{.*}}i32)
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_wait {{.*}} {num = 1 : i32}
  // CHECK: ttng.warp_group_dot
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: } {tle.async_tile_producer_count = 1 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  // CHECK: ttg.async_wait {{.*}} {num = 0 : i32}
  // CHECK: ttg.local_dealloc
  tt.func @loop_with_async_tile_producer(
      %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>,
      %b_ptr: tensor<64x512x!tt.ptr<bf16>, #blocked>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant dense<true> : tensor<64x512xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64x512xbf16, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%cur = %acc) -> (tensor<64x64xf32, #mma>) {
      %b = tt.load %b_ptr, %mask, %other {loop.stage = 0 : i32} : tensor<64x512x!tt.ptr<bf16>, #blocked>
      %b_smem = ttg.local_alloc %b {loop.stage = 0 : i32} : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      %b_view = tle.memdesc_wgmma_view %b_smem {loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %b_view, %cur {inputPrecision = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @loop_with_rematerializable_seed_before_dot
  // CHECK: scf.if
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16
  // CHECK: scf.for
  // CHECK: arith.cmpi
  // CHECK: arith.andi
  // CHECK: tt.expand_dims
  // CHECK: arith.select
  // CHECK: tt.broadcast
  // CHECK: ttg.async_wait {{.*}} {num = 1 : i32}
  // CHECK: ttng.warp_group_dot
  // CHECK: } {tle.async_tile_producer_count = 1 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  // CHECK: ttg.async_wait {{.*}} {num = 0 : i32}
  tt.func @loop_with_rematerializable_seed_before_dot(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>,
      %b_ptr: tensor<64x64x!tt.ptr<bf16>, #blocked>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #mma>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<1x64xf32, #mma>
    %mask = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%cur = %acc) -> (tensor<64x64xf32, #mma>) {
      %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %c0_t = tt.splat %c0_i32 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %c63_t = tt.splat %c63_i32 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %lo = arith.cmpi sge, %range, %c0_t : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %hi = arith.cmpi sle, %range, %c63_t : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %mask_ids = arith.andi %lo, %hi : tensor<64xi1, #ttg.slice<{dim = 0, parent = #mma}>>
      %b = tt.load %b_ptr, %mask, %other {loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      %b_smem = ttg.local_alloc %b {loop.stage = 0 : i32} : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
      %mask_row = tt.expand_dims %mask_ids {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi1, #mma>
      %seed_row = arith.select %mask_row, %cst_0, %cst_1 : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
      %seed = tt.broadcast %seed_row : tensor<1x64xf32, #mma> -> tensor<64x64xf32, #mma>
      %b_view = tle.memdesc_wgmma_view %b_smem {loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %b_view, %seed {inputPrecision = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @loop_with_two_async_tile_producers
  // CHECK: scf.if
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: scf.for {{.*}} iter_args({{.*}}!ttg.async.token, {{.*}}!ttg.async.token, {{.*}}i32, {{.*}}!ttg.async.token, {{.*}}!ttg.async.token, {{.*}}i32)
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 1 : i32}
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: } {tle.async_tile_producer_count = 2 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0 : i32}
  // CHECK: ttg.local_dealloc
  // CHECK: ttg.local_dealloc
  tt.func @loop_with_two_async_tile_producers(
      %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>,
      %b_ptr: tensor<64x512x!tt.ptr<bf16>, #blocked>,
      %c_ptr: tensor<64x64x!tt.ptr<bf16>, #blocked>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %mask_b = arith.constant dense<true> : tensor<64x512xi1, #blocked>
    %other_b = arith.constant dense<0.000000e+00> : tensor<64x512xbf16, #blocked>
    %mask_c = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %other_c = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%cur = %acc) -> (tensor<64x64xf32, #mma>) {
      %b = tt.load %b_ptr, %mask_b, %other_b {loop.stage = 0 : i32} : tensor<64x512x!tt.ptr<bf16>, #blocked>
      %b_smem = ttg.local_alloc %b {loop.stage = 0 : i32} : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      %c = tt.load %c_ptr, %mask_c, %other_c {loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      %c_smem = ttg.local_alloc %c {loop.stage = 0 : i32} : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
      %c_view = tle.memdesc_wgmma_view %c_smem {loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      %tmp = ttng.warp_group_dot %c_smem, %c_view, %cur {inputPrecision = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      %b_view = tle.memdesc_wgmma_view %b_smem {loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %b_view, %tmp {inputPrecision = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @loop_with_direct_async_tile_family
  // CHECK: scf.if
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: scf.for {{.*}} iter_args({{.*}}!ttg.async.token, {{.*}}i32, {{.*}}!ttg.async.token, {{.*}}i32)
  // CHECK: tle.memdesc_wgmma_view
  // CHECK: ttg.async_wait {{.*}} {num = 1 : i32}
  // CHECK: ttng.warp_group_dot
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: } {tle.async_tile_producer_count = 1 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  // CHECK: ttg.async_wait {{.*}} {num = 0 : i32}
  tt.func @loop_with_direct_async_tile_family(
      %base: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>,
      %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %other = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %mask = arith.constant dense<true> : tensor<64x128xi1, #blocked>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %k = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<64x1xi32, #blocked>
    %m_off = arith.muli %m, %stride : tensor<64x1xi32, #blocked>
    %m_off_b = tt.broadcast %m_off : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %k_b = tt.broadcast %k : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %offs = arith.addi %m_off_b, %k_b : tensor<64x128xi32, #blocked>
    %ptr = tt.splat %base : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptr, %offs : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> tensor<64x64xf32, #mma> {
      %sub = ttg.memdesc_subslice %dst[0, 0] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable, 64x512>
      %tok = ttg.async_copy_global_to_local %ptrs, %sub mask %mask other %other {loop.cluster = 0 : i32, loop.stage = 0 : i32, tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable, 64x512>
      %commit = ttg.async_commit_group tokens %tok {loop.cluster = 0 : i32, loop.stage = 0 : i32}
      %wait = ttg.async_wait %commit {loop.cluster = 0 : i32, loop.stage = 0 : i32, num = 0 : i32}
      %view = tle.memdesc_wgmma_view %dst {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @loop_with_plain_async_tile_family_fallback
  // CHECK: scf.for
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tle.explicit_tile_style_pipeline
  // CHECK: } {tt.num_stages = 2 : i32}
  tt.func @loop_with_plain_async_tile_family_fallback(
      %base: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>,
      %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %other = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %mask = arith.constant dense<true> : tensor<64x128xi1, #blocked>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %k = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<64x1xi32, #blocked>
    %m_off = arith.muli %m, %stride : tensor<64x1xi32, #blocked>
    %m_off_b = tt.broadcast %m_off : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %k_b = tt.broadcast %k : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %offs = arith.addi %m_off_b, %k_b : tensor<64x128xi32, #blocked>
    %ptr = tt.splat %base : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptr, %offs : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> tensor<64x64xf32, #mma> {
      %sub = ttg.memdesc_subslice %dst[0, 0] : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable, 64x512>
      %tok = ttg.async_copy_global_to_local %ptrs, %sub mask %mask other %other : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable, 64x512>
      %commit = ttg.async_commit_group tokens %tok
      %wait = ttg.async_wait %commit {num = 0 : i32}
      %view = tle.memdesc_wgmma_view %dst {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @dynamic_loop_with_direct_async_tile_family
  // CHECK: scf.if
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x512xbf16
  // CHECK: ttg.memdesc_index
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: scf.if
  // CHECK: arith.subi %arg0
  // CHECK: scf.for {{.*}} iter_args({{.*}}!ttg.async.token, {{.*}}i32, {{.*}}!ttg.async.token, {{.*}}i32)
  // CHECK: ttg.async_wait {{.*}} {num = 1 : i32}
  // CHECK: ttng.warp_group_dot
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: } {tle.async_tile_producer_count = 1 : i32, tle.explicit_tile_style_pipeline = 1 : i32, tle.tile_style_pipeline = 1 : i32}
  // CHECK: ttg.async_wait {{.*}} {num = 0 : i32}
  tt.func @dynamic_loop_with_direct_async_tile_family(
      %ub: i32,
      %base: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>,
      %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %other = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %mask = arith.constant dense<true> : tensor<64x128xi1, #blocked>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %k = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<64x1xi32, #blocked>
    %m_off = arith.muli %m, %stride : tensor<64x1xi32, #blocked>
    %m_off_b = tt.broadcast %m_off : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %k_b = tt.broadcast %k : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %offs = arith.addi %m_off_b, %k_b : tensor<64x128xi32, #blocked>
    %ptr = tt.splat %base : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptr, %offs : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
    %out = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %sub = ttg.memdesc_subslice %dst[0, 0] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable, 64x512>
      %tok = ttg.async_copy_global_to_local %ptrs, %sub mask %mask other %other {loop.cluster = 0 : i32, loop.stage = 0 : i32, tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable, 64x512>
      %commit = ttg.async_commit_group tokens %tok {loop.cluster = 0 : i32, loop.stage = 0 : i32}
      %wait = ttg.async_wait %commit {loop.cluster = 0 : i32, loop.stage = 0 : i32, num = 0 : i32}
      %view = tle.memdesc_wgmma_view %dst {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
