// RUN: triton-opt --analyze-data-flow --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} { // expected-error
  func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c16384_i32 = arith.constant {ssbuffer.block_id = 16 : i32} 16384 : i32
    %cst = arith.constant {ssbuffer.block_id = 12 : i32} dense<[2, 4, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 12 : i32} dense<[64, 2, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : f32
    %c0_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 0 : i32
    %c128_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : i64
    %c1_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 1 : i32
    %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 15 : i32} 64 : i32
    %c32_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 32 : i32
    %c64 = arith.constant {DataUse, ssbuffer.block_id = 15 : i32} 64 : index
    %c0 = arith.constant {DataUse, ssbuffer.block_id = 15 : i32} 0 : index
    %c63_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 63 : i32
    %c16384_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 16384 : i64
    %c524288_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 524288 : i64
    %c128 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : index
    %c32 = arith.constant {ssbuffer.block_id = 15 : i32} 32 : index
    %c4096 = arith.constant {ssbuffer.block_id = 15 : i32} 4096 : index
    %cst_2 = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : bf16
    %c8192 = arith.constant {ssbuffer.block_id = 15 : i32} 8192 : index
    scope.scope : () -> () {
      %0 = arith.divsi %arg14, %c32_i32 {ssbuffer.block_id = 9 : i32} : i32
      %1 = arith.remsi %arg14, %c32_i32 {ssbuffer.block_id = 9 : i32} : i32
      %2 = arith.muli %0, %arg9 {ssbuffer.block_id = 9 : i32} : i32
      %3 = arith.muli %2, %c32_i32 {ssbuffer.block_id = 9 : i32} : i32
      %4 = arith.addi %3, %1 {ssbuffer.block_id = 9 : i32} : i32
      %5 = arith.extsi %4 {ssbuffer.block_id = 9 : i32} : i32 to i64
      %6 = arith.muli %5, %c128_i64 {ssbuffer.block_id = 9 : i32} : i64
      %7 = arith.index_cast %6 {ssbuffer.block_id = 9 : i32} : i64 to index
      %8 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
      %9 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_1 : f32) outs(%8 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %10 = arith.divsi %arg14, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %11 = arith.remsi %arg14, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %12 = arith.muli %10, %arg9 {ssbuffer.block_id = 15 : i32} : i32
      %13 = arith.addi %arg9, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
      %14 = arith.divsi %13, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %15 = arith.muli %12, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %16 = arith.addi %15, %11 {ssbuffer.block_id = 15 : i32} : i32
      %17 = arith.extsi %16 {ssbuffer.block_id = 15 : i32} : i32 to i64
      %18 = arith.muli %17, %c128_i64 {ssbuffer.block_id = 15 : i32} : i64
      %19 = arith.index_cast %18 {ssbuffer.block_id = 15 : i32} : i64 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
      hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
