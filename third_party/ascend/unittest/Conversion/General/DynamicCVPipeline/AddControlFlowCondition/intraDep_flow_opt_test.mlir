// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Unit Tests for AddControlFlowCondition Pass - Cross Core Dependencies Test
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // ============================================
  // Test Case 2: @test_data_flow_with_intra_dependent
  // ============================================
  // C1 -> V1
  // V1 -> V2
  // V2 -> C2
  // C2 -> V3
  // flowopt condition will be added on C3 block
  func.func @test_data_flow_with_intra_dependent(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    scope.scope : () -> () {
      // Vector scope only process each vector core's ssbuffer ptr
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_5a = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5a {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {

        hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_13a = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : tensor<8x8x16x16xf16>
        %reshape_14a = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : tensor<8x8x16x16xf16>
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 4 : i32}

        %reshape_13 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        %reshape_14 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32], ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        hivm.hir.copy ins(%reshape_14 : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32], ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 5 : i32}

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 6 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {

        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 0 : i32}

        // CHECK:       memref.load
        // CHECK:       arith.cmpi sgt
        // CHECK:       memref.load
        // CHECK:       arith.cmpi sgt
        // CHECK:       memref.load
        // CHECK:       arith.cmpi slt
        // CHECK:       memref.load
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi sge
        // CHECK:       arith.cmpi sge
        // CHECK:       arith.ori
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %32 = hivm.hir.convert_layout %alloc_4 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %321 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %40 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 1 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
