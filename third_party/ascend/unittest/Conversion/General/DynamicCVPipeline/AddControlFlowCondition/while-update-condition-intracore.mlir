// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// UpdateConditionInfo on scf.while: intra-core (核内) conditions.
// VECTOR while has producer (intraDeps role=1) then consumer (role=0).
// Actual shape (with optional cross gate on the same if):
//   producer: cmpi slt control_var, 1 ; then addi control_var, 1
//   consumer: cmpi sgt latest, 0      ; then subi latest, 1

// CHECK-LABEL: func.func @while_update_condition_intracore

// CHECK: scf.while

// Producer if=5: skip cross memref.load/slt, then intra control_var < 1.
// CHECK: memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
// CHECK: %[[LIM:.*]] = arith.constant 1 : i32
// CHECK: %[[SLT:.*]] = arith.cmpi slt, %{{.*}}, %[[LIM]] : i32
// CHECK: %[[AND_P0:.*]] = arith.andi %{{.*}}, %[[SLT]]
// CHECK: %[[COND5:.*]] = arith.andi %[[AND_P0]], %{{.*}}
// CHECK: scf.if %[[COND5]]
// CHECK: %[[ONE_P:.*]] = arith.constant 1 : i32
// CHECK: %[[INC:.*]] = arith.addi %{{.*}}, %[[ONE_P]] : i32
// CHECK: scf.yield %{{.*}}, %[[INC]]
// CHECK: } {{.*}}ssbuffer.if = 5

// Consumer if=7: uses producer if result (latest), sgt 0 then -1.
// CHECK: %[[SGT:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
// CHECK: %[[COND7:.*]] = arith.andi %[[SGT]], %{{.*}}
// CHECK: scf.if %[[COND7]]
// CHECK: %[[ONE_C:.*]] = arith.constant 1 : i32
// CHECK: %[[DEC:.*]] = arith.subi %{{.*}}, %[[ONE_C]] : i32
// CHECK: } {{.*}}ssbuffer.if = 7

// CHECK: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @while_update_condition_intracore(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %bound = arith.extsi %arg7 : i32 to i64

    scope.scope : () -> () {
      %alloc_ub0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem0 = memref.memory_space_cast %alloc_ub0 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc_ub1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem1 = memref.memory_space_cast %alloc_ub1 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc_cbuf = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>

      %0 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %1 = arith.extsi %arg14 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        // Producer block (intraDeps role = 1).
        %t_prod = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t_prod : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}
        %cbuf_t = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x4x16x16xf16>
        hivm.hir.copy ins(%cbuf_t : tensor<8x4x16x16xf16>) outs(%alloc_cbuf : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        %next5 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 5 : i32} : i32

        // Consumer block (intraDeps role = 0).
        %t_cons = bufferization.to_tensor %mem0 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32> to tensor<128xf32>
        %next7 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 7 : i32} : i32
        scf.yield %next7 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_cbuf = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>

      %0 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %1 = arith.extsi %arg14 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_cbuf output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
        %next2 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 2 : i32} : i32
        scf.yield %next2 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
