// RUN: triton-opt --clone-ops --verify-each %s | FileCheck %s
// RUN: triton-opt --clone-ops --verify-each %s | FileCheck %s --check-prefix=CLONED

// Inter-core multi-buffering wraps the readiness notification in an scf.if
// picking the flag for this iteration's buffer half. CloneOps replays block 1
// into block 3 to resolve the memref dependency on %cast, which is what it is
// there for, but the wrapper must not be replayed with it: one produced tile
// would then be announced twice, from two pipeline stages whose iteration
// counters are their own.
//
// test-clone-ops-if-only-sync.mlir covers a cloned wrapper the exec-after rule
// erases. Here the consumer reads the same buffer, so the memory dependences
// keep the clone alive and only matching the wrapped form by type erases it.
//
// Exactly one notification survives: the original scf.if, whose two branches
// account for both occurrences below.

// CHECK-LABEL: func.func @cloned_wrapped_sync
// CHECK-COUNT-2: hivm.hir.sync_block_set
// CHECK-NOT: hivm.hir.sync_block_set
// CHECK: return

// The dependency itself must still be resolved by duplication, or the check
// above passes for the wrong reason.
// CLONED-LABEL: func.func @cloned_wrapped_sync
// CLONED: ssbuffer.clone

func.func @cloned_wrapped_sync(%out: memref<16x16xf16>, %upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  scope.scope : () -> () {
    %buffer = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %iv = %c0 to %upper step %c1 {
      linalg.fill {ssbuffer.block_id = 1 : i32} ins(%zero : f16) outs(%buffer : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      %layout = hivm.hir.convert_layout %buffer output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) -> memref<16x16xf16, #hivm.address_space<cbuf>>
      %cast = memref.memory_space_cast %layout {ssbuffer.block_id = 1 : i32} : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %condition = arith.cmpi eq, %iv, %c0 {ssbuffer.block_id = 1 : i32} : index
      scf.if %condition {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } else {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      } {ssbuffer.block_id = 1 : i32}
      %tensor = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 3 : i32} : memref<16x16xf16> to tensor<16x16xf16>
      hivm.hir.copy ins(%tensor : tensor<16x16xf16>) outs(%out : memref<16x16xf16>) {ssbuffer.block_id = 3 : i32}
    } {ssbuffer.main_loop = 0 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
