// RUN: triton-opt %s -triton-tle-tile-style-pipeline-schedule | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @tile_style_positive
  // CHECK: scf.for
  // CHECK: tt.load {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: tle.memdesc_wgmma_view {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32
  // CHECK: ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32}
  tt.func @tile_style_positive(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %k = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<64x1xi32, #blocked>
    %m_off = arith.muli %m, %stride : tensor<64x1xi32, #blocked>
    %m_off_b = tt.broadcast %m_off : tensor<64x1xi32, #blocked> -> tensor<64x512xi32, #blocked>
    %k_b = tt.broadcast %k : tensor<1x512xi32, #blocked> -> tensor<64x512xi32, #blocked>
    %offs = arith.addi %m_off_b, %k_b : tensor<64x512xi32, #blocked>
    %ptr = tt.splat %base : !tt.ptr<bf16> -> tensor<64x512x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptr, %offs : tensor<64x512x!tt.ptr<bf16>, #blocked>, tensor<64x512xi32, #blocked>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> tensor<64x64xf32, #mma> {
      %b = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x512x!tt.ptr<bf16>, #blocked>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      %b_view = tle.memdesc_wgmma_view %b_smem {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %b_view, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @tile_style_fallback
  // CHECK: scf.for
  // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
  tt.func @tile_style_fallback(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zeros = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %ptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zeros) -> tensor<4xf32> {
      %x = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<4x!tt.ptr<f32>>
      %y = arith.addf %x, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<4xf32>
      scf.yield %y : tensor<4xf32>
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %out : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @tile_style_async_copy_positive
  // CHECK: ttg.memdesc_subslice {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, tle.local_ptr_async_store}
  // CHECK: ttg.async_commit_group {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: tle.memdesc_wgmma_view {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32
  // CHECK: ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32}
  tt.func @tile_style_async_copy_positive(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
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
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @tile_style_async_copy_reschedules_stage1_producers
  // CHECK: ttg.memdesc_subslice {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, tle.local_ptr_async_store}
  // CHECK: ttg.async_commit_group {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.async_wait {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, num = 0 : i32}
  // CHECK: tle.memdesc_wgmma_view {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32
  // CHECK: ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32}
  tt.func @tile_style_async_copy_reschedules_stage1_producers(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
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
      %sub = ttg.memdesc_subslice %dst[0, 0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable, 64x512>
      %tok = ttg.async_copy_global_to_local %ptrs, %sub mask %mask other %other {loop.cluster = 0 : i32, loop.stage = 1 : i32, tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable, 64x512>
      %commit = ttg.async_commit_group tokens %tok {loop.cluster = 0 : i32, loop.stage = 1 : i32}
      %wait = ttg.async_wait %commit {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
      %view = tle.memdesc_wgmma_view %dst {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @tile_style_async_copy_without_schedule
  // CHECK: scf.for
  // CHECK: ttg.memdesc_subslice {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, tle.local_ptr_async_store}
  // CHECK: ttg.async_commit_group {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // CHECK: ttg.async_wait {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, num = 0 : i32}
  // CHECK: tle.memdesc_wgmma_view {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32
  // CHECK: ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: } {tle.tile_style_pipeline = 1 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32}
  tt.func @tile_style_async_copy_without_schedule(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
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
      %tok = ttg.async_copy_global_to_local %ptrs, %sub mask %mask other %other {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable, 64x512>
      %commit = ttg.async_commit_group tokens %tok
      %wait = ttg.async_wait %commit {num = 0 : i32}
      %view = tle.memdesc_wgmma_view %dst {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @tile_style_async_copy_without_tle_provenance
  // CHECK: scf.for
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tle.tile_style_pipeline
  // CHECK: } {tt.num_stages = 2 : i32}
  tt.func @tile_style_async_copy_without_tle_provenance(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
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
}
