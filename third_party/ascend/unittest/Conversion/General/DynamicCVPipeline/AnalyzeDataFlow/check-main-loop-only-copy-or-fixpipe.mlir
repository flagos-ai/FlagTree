// RUN: triton-opt --analyze-scope --verify-diagnostics

// Unit test for AnalyzeScopePass: when every main_loop id lacks either
// hivm.hir.copy or hivm.hir.fixpipe, the new isMainLoopOnlyCopyOrFixpipe
// check must trigger setFallbackAttr so the dynamic CV pipeline falls
// back to the original workflow.
//
// In this test the VECTOR main_loop (id 0) has only hivm.hir.copy with
// transfer_id (to satisfy checkVecScopeMainLoop) and no fixpipe ops; the
// CUBE main_loop (id 0) has only sync_block ops and no copy or fixpipe.
// For id 0: countCopy > 0 and countFixpipe == 0, so the rule
// (countCopy == 0 || countFixpipe == 0) is satisfied for the only id and
// isMainLoopOnlyCopyOrFixpipe returns true -- fallback is set.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_main_loop_only_copy(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 14 : i32} 1 : i32
    %c32_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 32 : i32
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.000000e+00 : bf16
    %reshape = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<4x4x16x16xbf16>
    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape : tensor<4x4x16x16xbf16>) outs(%alloc : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      } {ssbuffer.block_id = 22 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 22 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 22 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      } {ssbuffer.block_id = 22 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
