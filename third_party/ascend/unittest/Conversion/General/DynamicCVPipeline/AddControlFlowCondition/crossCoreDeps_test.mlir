// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Unit Tests for AddControlFlowCondition Pass - Cross Core Dependencies Test
//
// This test verifies:
// 1. ssbuffer initialization at scope start (constants 0, 0, 1024, 4, 1028, 8, 1032)
// 2. if condition calculation before scf.if (llvm.load, arith.cmpi)
// 3. update operations at end of wrapped compute ops (llvm.load, arith.subi/arith.addi, llvm.store)

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    // ssbuffer init
    // CHECK:       %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:       %[[ADDR0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:       %[[ADDR1:.*]] = llvm.mlir.constant(1024 : i64) : i64
    // CHECK:       %[[PTR0:.*]] = llvm.inttoptr %[[ADDR0]] : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR1:.*]] = llvm.inttoptr %[[ADDR1]] : i64 to !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR0]] : i32, !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR1]] : i32, !llvm.ptr<11>
    // CHECK:       %[[ADDR2:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK:       %[[ADDR3:.*]] = llvm.mlir.constant(1028 : i64) : i64
    // CHECK:       %[[PTR2:.*]] = llvm.inttoptr %[[ADDR2]] : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR3:.*]] = llvm.inttoptr %[[ADDR3]] : i64 to !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR2]] : i32, !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR3]] : i32, !llvm.ptr<11>
    // CHECK:       %[[ADDR4:.*]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK:       %[[ADDR5:.*]] = llvm.mlir.constant(1032 : i64) : i64
    // CHECK:       %[[PTR4:.*]] = llvm.inttoptr %[[ADDR4]] : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR5:.*]] = llvm.inttoptr %[[ADDR5]] : i64 to !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR4]] : i32, !llvm.ptr<11>
    // CHECK:       llvm.store %[[ZERO]], %[[PTR5]] : i32, !llvm.ptr<11>
    scope.scope : () -> () {
      // Vector scope only process each vector core's ssbuffer ptr
      // CHECK:       arith.constant 1024
      // CHECK:       hivm.hir.get_sub_block_idx
      // CHECK:       arith.muli
      // CHECK:       arith.addi
      // CHECK:       llvm.inttoptr
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // This block contain an input and a output
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi sgt
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_13 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}
        // CHECK:       llvm.load
        // CHECK:       arith.subi
        // CHECK:       llvm.store
        // CHECK:       llvm.load
        // CHECK:       arith.addi
        // CHECK:       llvm.store
        // CHECK:       } {ssbuffer.if = 5 : i32}

        // This block only contain an input
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi sgt
        // CHECK:       scf.if
        %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %48 = bufferization.to_tensor %memspacecast_16 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32> to tensor<128x128xf32>
        // CHECK:       llvm.load
        // CHECK:       arith.subi
        // CHECK:       llvm.store
        // CHECK:       } {ssbuffer.if = 6 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossDeps = [2 : i32, 1 : i32]} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // This block contain a output
        // in CUBE scope, it need to process two ssbuffer ptr
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.addi
        // CHECK:       arith.addi
        // CHECK:       llvm.store
        // CHECK:       llvm.store
        // CHECK:       } {ssbuffer.if = 0 : i32}

        // This block contain an input and a output
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi sgt
        // CHECK:       arith.cmpi sgt
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        %32 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %40 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} ins(%40 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.subi
        // CHECK:       arith.subi
        // CHECK:       llvm.store
        // CHECK:       llvm.store
        // CHECK:       llvm.load
        // CHECK:       llvm.load
        // CHECK:       arith.addi
        // CHECK:       arith.addi
        // CHECK:       llvm.store
        // CHECK:       llvm.store
        // CHECK:       } {ssbuffer.if = 1 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
