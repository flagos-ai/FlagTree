// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' \
// RUN:                --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file \
// RUN: | FileCheck %s

// -----
// Adjacent 16-f32 tiles form a small contiguous DMA. The pass should merge
// 16 tiles per program, drop the all-true tile mask, and record the launch-grid
// shrink metadata on the tile program-id axis.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 16 : i32
// CHECK-LABEL: func.func @chunk_coalesce_simple
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_coalesce_simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Unmasked kernels do not carry a static tile count in the IR. The pass cannot
// prove runtime grid[axis] is >= H and divisible by H, so it must leave the
// kernel uncoalesced.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_skip_unmasked_unknown_grid
// CHECK-NOT: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_skip_unmasked_unknown_grid(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Unmasked kernels with hacc.grid_num_tiles can coalesce when the hint gives
// the exact compile-time tile count.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 16 : i32
// CHECK-NOT: hacc.grid_num_tiles
// CHECK-LABEL: func.func @tile_chunk_unmasked_with_grid_hint
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 128 : i32} {
  tt.func public @tile_chunk_unmasked_with_grid_hint(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A one-tile mask has BOUND == tileLen. It is provably all true for the only
// tile, but the static tile count is one, so there is nothing to coalesce.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_single_full_tile
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_single_full_tile(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c16_tensor = arith.constant dense<16> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c16_tensor : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Reading num_programs on the coalesced axis is unsafe because the host launcher
// divides that grid dimension by H. The pass must leave the kernel uncoalesced.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_reads_num_programs
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_reads_num_programs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %num = tt.get_num_programs x : i32
    %c16 = arith.constant 16 : i32
    %c512 = arith.constant dense<512> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %num_splat = tt.splat %num : i32 -> tensor<16xi32>
    %guard_offs = arith.addi %offs, %num_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %guard_offs, %c512 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Partial tail masks are not separable after prepending the H lane, so the pass
// must keep the original one-tile program shape.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_partial_tail
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_partial_tail(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c250 = arith.constant dense<250> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c250 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A 2-D masked block may contain the tile-index signature on the outer grid
// axis, but a dynamic boundary mask on that axis means grid/H is not provably
// exact and the lifted mask is not a single structured slice. Keep the original
// program shape instead.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_skip_2d_dynamic_boundary
// CHECK-NOT: tensor<2x256x256
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_skip_2d_dynamic_boundary(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %m: i32, %n: i32) {
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32>

    %row_blk = arith.muli %pid_m, %c256 : i32
    %row_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %row_splat = tt.splat %row_blk : i32 -> tensor<256xi32>
    %row = arith.addi %row_splat, %row_range : tensor<256xi32>
    %row_2d = tt.expand_dims %row {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>

    %col_blk = arith.muli %pid_n, %c256 : i32
    %col_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %col_splat = tt.splat %col_blk : i32 -> tensor<256xi32>
    %col = arith.addi %col_splat, %col_range : tensor<256xi32>
    %col_2d = tt.expand_dims %col {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

    %n_row = tt.splat %n : i32 -> tensor<256x1xi32>
    %row_offset = arith.muli %row_2d, %n_row : tensor<256x1xi32>
    %row_offset_bc = tt.broadcast %row_offset : tensor<256x1xi32> -> tensor<256x256xi32>
    %col_bc = tt.broadcast %col_2d : tensor<1x256xi32> -> tensor<256x256xi32>
    %offsets = arith.addi %row_offset_bc, %col_bc : tensor<256x256xi32>

    %m_bound = tt.splat %m : i32 -> tensor<256x1xi32>
    %row_mask = arith.cmpi slt, %row_2d, %m_bound : tensor<256x1xi32>
    %row_mask_bc = tt.broadcast %row_mask : tensor<256x1xi1> -> tensor<256x256xi1>
    %n_bound = tt.splat %n : i32 -> tensor<1x256xi32>
    %col_mask = arith.cmpi slt, %col_2d, %n_bound : tensor<1x256xi32>
    %col_mask_bc = tt.broadcast %col_mask : tensor<1x256xi1> -> tensor<256x256xi1>
    %mask = arith.andi %row_mask_bc, %col_mask_bc : tensor<256x256xi1>

    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offsets : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<256x256x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offsets : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
    tt.store %dst_ptr, %val, %mask : tensor<256x256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// BUG REPRO: Unmasked kernel with grid_num_tiles=1 — only one tile, nothing to
// coalesce. The pass must NOT rewrite (H would be 0 or 1). Previously this
// could crash or produce dead code.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-NOT: hacc.grid_num_tiles
// CHECK-LABEL: func.func @tile_chunk_grid_hint_one_tile
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 1 : i32} {
  tt.func public @tile_chunk_grid_hint_one_tile(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// BUG REPRO: grid_num_tiles is a prime number (e.g. 127) that is not divisible
// by any H in [hMin, maxH]. chooseH must return 0 and the pass must bail.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-NOT: hacc.grid_num_tiles
// CHECK-LABEL: func.func @tile_chunk_prime_num_tiles
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 127 : i32} {
  tt.func public @tile_chunk_prime_num_tiles(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// BUG REPRO: tileLen=4 with f16 (blockBytes=4*2=8). hMin = ceil(512/8) = 64.
// maxH is capped at 16 < hMin=64, so the pass must bail (cannot reach
// kMinContigBytes within UB budget).
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_tiny_tile_bails
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_tiny_tile_bails(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %c1024 = arith.constant dense<1024> : tensor<4xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %blk = arith.muli %pid, %c4 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<4xi32>
    %offs = arith.addi %blk_splat, %range : tensor<4xi32>
    %mask = arith.cmpi slt, %offs, %c1024 : tensor<4xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<4x!tt.ptr<f16>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    tt.store %dst_ptr, %val, %mask : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----
// Large footprint forces maxH < 2 due to UB overflow. The pass must bail
// instead of setting maxH=2 and overflowing. Here the kernel has two large
// 256x256xf32 tensors => footprintUnit = 2*256*256*4 = 524288 > kUBBytesBudget.
// maxH = floor(131072/524288) = 0 < 2 → bail.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_ub_overflow_bails
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_ub_overflow_bails(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c256 = arith.constant 256 : i32
    %c4096 = arith.constant dense<4096> : tensor<256xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %blk = arith.muli %pid, %c256 : i32
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<256xi32>
    %offs = arith.addi %blk_splat, %range : tensor<256xi32>
    %mask = arith.cmpi slt, %offs, %c4096 : tensor<256xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    // Heavy compute: expand to NxN to inflate footprint
    %exp = tt.expand_dims %val {axis = 1 : i32} : tensor<256xf32> -> tensor<256x1xf32>
    %bcast = tt.broadcast %exp : tensor<256x1xf32> -> tensor<256x256xf32>
    %red = "tt.reduce"(%bcast) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<256x256xf32>) -> tensor<256xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %dst_ptr, %red, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// The two same-axis program-id values are canonicalized to one launch argument
// by the preceding Unstructure stage.  Keep the original Chunk behavior:
// operate on that canonical single seed.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 16 : i32
// CHECK-LABEL: func.func @tile_chunk_duplicate_pid
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_duplicate_pid(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid1 = tt.get_program_id x : i32
    %pid2 = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid1, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    // pid2 used for a different base offset (not matching the seed pattern)
    %off2 = arith.muli %pid2, %c16 : i32
    %off2_splat = tt.splat %off2 : i32 -> tensor<16xi32>
    %dst_offs = arith.addi %off2_splat, %range : tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %dst_offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Pid-derived offset goes through arith.remsi (unsafe op). The pass must bail.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_unsafe_remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_unsafe_remsi(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c8 = arith.constant 8 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %rem_pid = arith.remsi %pid, %c8 : i32
    %blk = arith.muli %rem_pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Mask is used by operations OTHER than load/store (e.g. arith.select). The
// pass must bail because dropping the mask would be incorrect.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_mask_escapes_to_where
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_mask_escapes_to_where(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    // mask used by select (non load/store use) -> must bail
    %selected = arith.select %mask, %val, %one : tensor<16xi1>, tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %selected, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Coalescing with scan and reduce in the region — must succeed (these ops are
// liftable). tileLen=16, BOUND=256 => numTiles=16, H should be chosen.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32
// CHECK-LABEL: func.func @tile_chunk_with_scan_reduce
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_with_scan_reduce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %scanned = "tt.scan"(%val) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }) : (tensor<16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %scanned, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// grid_num_tiles=0 (invalid) — must bail, not crash.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-NOT: hacc.grid_num_tiles
// CHECK-LABEL: func.func @tile_chunk_grid_hint_zero
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 0 : i32} {
  tt.func public @tile_chunk_grid_hint_zero(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// grid_num_tiles negative (invalid) — must bail, not crash.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-NOT: hacc.grid_num_tiles
// CHECK-LABEL: func.func @tile_chunk_grid_hint_negative
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = -1 : i32} {
  tt.func public @tile_chunk_grid_hint_negative(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
