// RUN: triton-opt %s --tritonhcugpu-block-pingpong="num-stages=3" --verify-each | FileCheck %s
//
// Derived from valid IR immediately before BlockPingpong. Preserve the real
// stage3/W8 loop, its two async producers, carried local-load descriptors,
// tail wait and the B reshape/trans/reshape chain. There are no private module
// hints, debug locations or unrelated C epilogue operations.
//
// The pass must move the view dependencies together with the dot. Moving only
// the dot leaves its B operand defined later in the block (invalid SSA).
// Keep both asymmetric conditional barriers around the scheduled loop.
//
// CHECK-LABEL: @block_pingpong_view_dependencies(
// CHECK: %[[GROUP:[a-zA-Z0-9_]+]] = arith.divsi
// CHECK: %[[LOW:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[GROUP]],
// CHECK: %[[HIGH:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[GROUP]],
// CHECK: hcug.cond_barrier %[[HIGH]]
// CHECK: scf.for
// CHECK: %[[A:[a-zA-Z0-9_]+]] = ttg.local_load {{.*}} -> tensor<256x16xbf16,
// CHECK: %[[B:[a-zA-Z0-9_]+]] = ttg.local_load {{.*}} -> tensor<16x256xbf16,
// CHECK: %[[R0:[a-zA-Z0-9_]+]] = tt.reshape %[[B]] : tensor<16x256xbf16, {{.*}}> -> tensor<16x8x16x2xbf16,
// CHECK: %[[T:[a-zA-Z0-9_]+]] = tt.trans %[[R0]] {order = array<i32: 0, 1, 3, 2>}
// CHECK: %[[R1:[a-zA-Z0-9_]+]] = tt.reshape %[[T]] : tensor<16x8x2x16xbf16, {{.*}}> -> tensor<16x256xbf16,
// CHECK: %[[DOT:[a-zA-Z0-9_]+]] = tt.dot %[[A]], %[[R1]],
// CHECK: scf.yield %[[DOT]],
// CHECK: hcug.cond_barrier %[[LOW]]
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear2 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 32], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 2, 0, 0]], lane = [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 8, 0], [4, 0, 0, 0], [8, 0, 0, 0]], warp = [[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 2, 0, 0]], lane = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [4, 0, 0, 0], [8, 0, 0, 0]], warp = [[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1, tilesPerWarp = [4, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func @block_pingpong_view_dependencies(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %acc_0 = arith.constant 3 : i32
    %acc_1 = arith.constant 62 : i32
    %a = arith.constant dense<16> : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %a_2 = arith.constant dense<16> : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_3 = arith.constant dense<1024> : tensor<256x1xi64, #blocked1>
    %cst_4 = arith.constant dense<2048> : tensor<16x1xi64, #blocked>
    %row0 = tt.get_program_id x : i32
    %row0_5 = arith.muli %row0, %c256_i32 : i32
    %tile_n = tt.get_program_id y : i32
    %tile_n_6 = arith.remui %tile_n, %c8_i32 : i32
    %col0 = arith.muli %tile_n_6, %c256_i32 : i32
    %ap = arith.extsi %row0_5 : i32 to i64
    %bp = arith.extsi %col0 : i32 to i64
    %a_7 = tt.splat %A : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked1>
    %a_8 = tt.splat %ap : i64 -> tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %a_9 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %a_10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_11 = arith.extsi %a_9 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %a_12 = arith.extsi %a_10 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_13 = arith.addi %a_8, %a_11 : tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %a_14 = tt.expand_dims %a_13 {axis = 1 : i32} : tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi64, #blocked1>
    %a_15 = arith.muli %a_14, %cst_3 : tensor<256x1xi64, #blocked1>
    %a_16 = tt.broadcast %a_15 : tensor<256x1xi64, #blocked1> -> tensor<256x16xi64, #blocked1>
    %a_17 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_18 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %a_19 = arith.extsi %a_17 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_20 = arith.extsi %a_18 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %b = tt.splat %B : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>, #blocked>
    %b_21 = tt.splat %bp : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %b_22 = arith.addi %b_21, %a_12 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %b_23 = tt.expand_dims %b_22 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi64, #blocked>
    %b_24 = tt.broadcast %b_23 : tensor<1x256xi64, #blocked> -> tensor<16x256xi64, #blocked>
    %a_25 = ttg.local_alloc : () -> !ttg.memdesc<3x256x16xbf16, #shared, #smem, mutable>
    %b_26 = ttg.local_alloc : () -> !ttg.memdesc<3x16x256xbf16, #shared1, #smem, mutable>
    %a_27 = tt.expand_dims %a_19 {axis = 0 : i32} : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi64, #blocked1>
    %a_28 = tt.broadcast %a_27 : tensor<1x16xi64, #blocked1> -> tensor<256x16xi64, #blocked1>
    %a_29 = arith.addi %a_16, %a_28 : tensor<256x16xi64, #blocked1>
    %a_30 = tt.addptr %a_7, %a_29 : tensor<256x16x!tt.ptr<bf16>, #blocked1>, tensor<256x16xi64, #blocked1>
    %a_31 = ttg.memdesc_index %a_25[%c0_i32] : !ttg.memdesc<3x256x16xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    %a_32 = ttg.async_copy_global_to_local %a_30, %a_31 {contiguity = 2 : i32} : tensor<256x16x!tt.ptr<bf16>, #blocked1> -> <256x16xbf16, #shared, #smem, mutable>
    %a_33 = ttg.async_commit_group tokens %a_32
    %b_34 = tt.expand_dims %a_20 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %b_35 = arith.muli %b_34, %cst_4 : tensor<16x1xi64, #blocked>
    %b_36 = tt.broadcast %b_35 : tensor<16x1xi64, #blocked> -> tensor<16x256xi64, #blocked>
    %b_37 = arith.addi %b_36, %b_24 : tensor<16x256xi64, #blocked>
    %b_38 = tt.addptr %b, %b_37 : tensor<16x256x!tt.ptr<bf16>, #blocked>, tensor<16x256xi64, #blocked>
    %b_39 = ttg.memdesc_index %b_26[%c0_i32] : !ttg.memdesc<3x16x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>
    %b_40 = ttg.async_copy_global_to_local %b_38, %b_39 {contiguity = 2 : i32} : tensor<16x256x!tt.ptr<bf16>, #blocked> -> <16x256xbf16, #shared1, #smem, mutable>
    %b_41 = ttg.async_commit_group tokens %b_40
    %a_42 = arith.addi %a_19, %a_2 : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_43 = arith.addi %a_20, %a : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %a_44 = tt.expand_dims %a_42 {axis = 0 : i32} : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi64, #blocked1>
    %a_45 = tt.broadcast %a_44 : tensor<1x16xi64, #blocked1> -> tensor<256x16xi64, #blocked1>
    %a_46 = arith.addi %a_16, %a_45 : tensor<256x16xi64, #blocked1>
    %a_47 = tt.addptr %a_7, %a_46 : tensor<256x16x!tt.ptr<bf16>, #blocked1>, tensor<256x16xi64, #blocked1>
    %a_48 = ttg.memdesc_index %a_25[%c1_i32] : !ttg.memdesc<3x256x16xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    %a_49 = ttg.async_copy_global_to_local %a_47, %a_48 {contiguity = 2 : i32} : tensor<256x16x!tt.ptr<bf16>, #blocked1> -> <256x16xbf16, #shared, #smem, mutable>
    %a_50 = ttg.async_commit_group tokens %a_49
    %b_51 = tt.expand_dims %a_43 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %b_52 = arith.muli %b_51, %cst_4 : tensor<16x1xi64, #blocked>
    %b_53 = tt.broadcast %b_52 : tensor<16x1xi64, #blocked> -> tensor<16x256xi64, #blocked>
    %b_54 = arith.addi %b_53, %b_24 : tensor<16x256xi64, #blocked>
    %b_55 = tt.addptr %b, %b_54 : tensor<16x256x!tt.ptr<bf16>, #blocked>, tensor<16x256xi64, #blocked>
    %b_56 = ttg.memdesc_index %b_26[%c1_i32] : !ttg.memdesc<3x16x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>
    %b_57 = ttg.async_copy_global_to_local %b_55, %b_56 {contiguity = 2 : i32} : tensor<16x256x!tt.ptr<bf16>, #blocked> -> <16x256xbf16, #shared1, #smem, mutable>
    %b_58 = ttg.async_commit_group tokens %b_57
    %a_59 = ttg.async_wait %a_33, %b_41 {num = 0 : i32}
    %acc_60:9 = scf.for %acc_86 = %c0_i32 to %acc_1 step %c1_i32 iter_args(%acc_87 = %acc, %arg5 = %c1_i32, %a_88 = %a_31, %a_89 = %a_48, %a_90 = %a_59, %b_91 = %b_39, %b_92 = %b_56, %a_93 = %a_50, %b_94 = %b_58) -> (tensor<256x256xf32, #mma>, i32, !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token)  : i32 {
      %acc_95 = arith.addi %arg5, %c1_i32 : i32
      %acc_96 = arith.cmpi slt, %acc_95, %acc_0 : i32
      %acc_97 = arith.select %acc_96, %acc_95, %c0_i32 : i32
      %acc_98 = arith.addi %acc_86, %c2_i32 : i32
      %k0 = arith.muli %acc_98, %c16_i32 : i32
      %ap_99 = arith.extsi %k0 : i32 to i64
      %a_100 = tt.splat %ap_99 : i64 -> tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %a_101 = tt.splat %ap_99 : i64 -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %a_102 = arith.addi %a_100, %a_19 : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %a_103 = arith.addi %a_101, %a_20 : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %a_104 = tt.expand_dims %a_102 {axis = 0 : i32} : tensor<16xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi64, #blocked1>
      %a_105 = tt.broadcast %a_104 : tensor<1x16xi64, #blocked1> -> tensor<256x16xi64, #blocked1>
      %a_106 = arith.addi %a_16, %a_105 : tensor<256x16xi64, #blocked1>
      %a_107 = tt.addptr %a_7, %a_106 : tensor<256x16x!tt.ptr<bf16>, #blocked1>, tensor<256x16xi64, #blocked1>
      %a_108 = ttg.memdesc_index %a_25[%acc_97] : !ttg.memdesc<3x256x16xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
      %a_109 = ttg.async_copy_global_to_local %a_107, %a_108 {contiguity = 2 : i32} : tensor<256x16x!tt.ptr<bf16>, #blocked1> -> <256x16xbf16, #shared, #smem, mutable>
      %a_110 = ttg.async_commit_group tokens %a_109
      %a_111 = ttg.local_load %a_88 token %a_90 : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %b_112 = tt.expand_dims %a_103 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %b_113 = arith.muli %b_112, %cst_4 : tensor<16x1xi64, #blocked>
      %b_114 = tt.broadcast %b_113 : tensor<16x1xi64, #blocked> -> tensor<16x256xi64, #blocked>
      %b_115 = arith.addi %b_114, %b_24 : tensor<16x256xi64, #blocked>
      %b_116 = tt.addptr %b, %b_115 : tensor<16x256x!tt.ptr<bf16>, #blocked>, tensor<16x256xi64, #blocked>
      %b_117 = ttg.memdesc_index %b_26[%acc_97] : !ttg.memdesc<3x16x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>
      %b_118 = ttg.async_copy_global_to_local %b_116, %b_117 {contiguity = 2 : i32} : tensor<16x256x!tt.ptr<bf16>, #blocked> -> <16x256xbf16, #shared1, #smem, mutable>
      %b_119 = ttg.async_commit_group tokens %b_118
      %b_120 = ttg.local_load %b_91 token %a_90 : !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable> -> tensor<16x256xbf16, #linear2>
      %b_121 = tt.reshape %b_120 : tensor<16x256xbf16, #linear2> -> tensor<16x8x16x2xbf16, #linear3>
      %b_122 = tt.trans %b_121 {order = array<i32: 0, 1, 3, 2>} : tensor<16x8x16x2xbf16, #linear3> -> tensor<16x8x2x16xbf16, #linear4>
      %b_123 = tt.reshape %b_122 : tensor<16x8x2x16xbf16, #linear4> -> tensor<16x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %acc_124 = tt.dot %a_111, %b_123, %acc_87 : tensor<256x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %a_125 = ttg.async_wait %a_93, %b_94 {num = 0 : i32}
      scf.yield %acc_124, %acc_97, %a_89, %a_108, %a_125, %b_92, %b_117, %a_110, %b_119 : tensor<256x256xf32, #mma>, i32, !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<16x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}
