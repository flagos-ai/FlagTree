// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_cross_buffer_ctov() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_set {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.for %j = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
        %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
        %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
        scf.yield
      } {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32}
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
}

// Pass runs without crash, nested-loop case handled gracefully
// CHECK: func.func @tc_cross_buffer_ctov
// CHECK: return
