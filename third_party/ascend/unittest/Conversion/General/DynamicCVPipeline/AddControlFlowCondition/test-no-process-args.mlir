// RUN: triton-opt --process-args %s --allow-unregistered-dialect | FileCheck %s --implicit-check-not=ssbuffer.arg
// CHECK: func.func @test_no_process_args

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @triton_indirect_load(memref<?xi8>, tensor<128x64xi64>, tensor<128x64xi1>, tensor<128x64xi8>) -> tensor<128x64xi8>
  func.func private @triton_indirect_load_0(memref<?xi8>, tensor<64x256xi64>, tensor<64x256xi1>, tensor<64x256xi8>) -> tensor<64x256xi8>
  func.func @test_no_process_args(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "mix_simd_simt"} {
    %c2048_i32 = arith.constant {DataUse, ssbuffer.block_id = 8 : i32} 2048 : i32
    %c64_i32 = arith.constant {DataUse, ssbuffer.block_id = 8 : i32} 64 : i32
    %c8_i32 = arith.constant {MixUse, ssbuffer.block_id = 8 : i32} 8 : i32
    %c256_i32 = arith.constant {MixUse, ssbuffer.block_id = 8 : i32} 256 : i32
    %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 8 : i32} 128 : i32
    %c2_i32 = arith.constant {DataUse, ssbuffer.block_id = 8 : i32} 2 : i32
    %c3_i32 = arith.constant {DataUse, ssbuffer.block_id = 8 : i32} 3 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    %c127_i32 = arith.constant {MixUse, ssbuffer.block_id = 8 : i32} 127 : i32
    %c255_i32 = arith.constant {MixUse, ssbuffer.block_id = 8 : i32} 255 : i32
    %c4_i32 = arith.constant {Undefined, ssbuffer.block_id = 8 : i32} 4 : i32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 8 : i32} 1 : i32
    %c0_i8 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i8
    %c1_i8 = arith.constant {ssbuffer.block_id = 8 : i32} 1 : i8
    %c128 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : index
    %c256 = arith.constant {ssbuffer.block_id = 8 : i32} 256 : index
    %cst = arith.constant {ssbuffer.block_id = 2 : i32} dense<[8, 4, 16, 32]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 2 : i32} dense<[64, 8, 32]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 2 : i32} dense<[2, 8, 16, 32]> : tensor<4xi64>
    %cst_2 = arith.constant {ssbuffer.block_id = 2 : i32} dense<[128, 2, 32]> : tensor<3xi64>
    scope.scope : () -> () {
      %0 = arith.addi %arg5, %c127_i32 {MixUse, ssbuffer.block_id = 8 : i32} : i32
      %1 = arith.divsi %0, %c128_i32 {MixUse, ssbuffer.block_id = 8 : i32} : i32
      %2 = arith.addi %arg6, %c255_i32 {MixUse, ssbuffer.block_id = 8 : i32} : i32
      %3 = arith.divsi %2, %c256_i32 {MixUse, ssbuffer.block_id = 8 : i32} : i32
      %4 = arith.muli %1, %3 {Undefined, ssbuffer.block_id = 8 : i32} : i32
      scf.for %arg17 = %arg14 to %4 step %arg10  : i32 {
        scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x32xi8, #hivm.address_space<cbuf>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x32xi8, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          %alloc_3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x32xi8, #hivm.address_space<cbuf>>
          annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x32xi8, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          %alloc_4 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x256xi32, #hivm.address_space<ub>>
          annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x256xi32, #hivm.address_space<ub>>
          scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            %5 = hivm.hir.convert_layout %alloc_3 output_shape [64, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : (memref<8x4x16x32xi8, #hivm.address_space<cbuf>>) -> memref<64x256xi8, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %5 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x256xi8, #hivm.address_space<cbuf>> to memref<64x256xi8>
            %6 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x256xi8> to tensor<64x256xi8>
            hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            %7 = hivm.hir.convert_layout %alloc output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<2x8x16x32xi8, #hivm.address_space<cbuf>>) -> memref<128x64xi8, #hivm.address_space<cbuf>>
            %memspacecast_5 = memref.memory_space_cast %7 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xi8, #hivm.address_space<cbuf>> to memref<128x64xi8>
            %8 = bufferization.to_tensor %memspacecast_5 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xi8> to tensor<128x64xi8>
            %9 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x256xi32>
            %10 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%c0_i32 : i32) outs(%9 : tensor<128x256xi32>) -> tensor<128x256xi32>
            %11 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%8, %6 : tensor<128x64xi8>, tensor<64x256xi8>) outs(%10 : tensor<128x256xi32>) -> tensor<128x256xi32>
            hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
            hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
            hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} ins(%11 : tensor<128x256xi32>) outs(%alloc_4 : memref<128x256xi32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          } {DataUse, ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i32}
          hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        } {DataUse, ssbuffer.block_id = 10 : i32}
      } {Undefined, ssbuffer.block_id = 11 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
