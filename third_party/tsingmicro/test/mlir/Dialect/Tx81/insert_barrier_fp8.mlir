//===----------------------------------------------------------------------===//
// Test File for Tx81 barrier insertion: CPU-executed FP8 conversions
//
// FP8/FP4 conversions lower to scalar CRT loops executed by the KCORE, while
// tx.gemm / tx.addvv are async instructions. When both touch the same SPM
// buffer inside a loop, the next iteration's conversion could overwrite data
// still being read, so InsertBarrierPass must treat the conversion as CPU side
// and emit a barrier at the end of the loop body.
//===----------------------------------------------------------------------===//

// RUN: tsingmicro-opt --tx81-insert-barrier %s | FileCheck %s

// Conversion writes %dst on the KCORE, async addvv reads/writes the same
// buffer: one entry barrier plus one per-iteration barrier at the loop end.
// CHECK-LABEL: func.func @fp8_convert_loop
// CHECK:         tx.barrier
// CHECK:         scf.for
// CHECK-NOT:     tx.barrier
// CHECK:           "tx.fp8E4M3FN_bf16"
// CHECK:           "tx.addvv"
// CHECK:           tx.barrier

func.func @fp8_convert_loop() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %src = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %dst = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %src_ptr = memref.extract_aligned_pointer_as_index %src : memref<64xf8E4M3FN> -> index
  %src_addr = arith.index_cast %src_ptr : index to i64
  %dst_ptr = memref.extract_aligned_pointer_as_index %dst : memref<64xbf16> -> index
  %dst_addr = arith.index_cast %dst_ptr : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %conv = "tx.fp8E4M3FN_bf16"(%src_addr, %dst_addr, %n) : (i64, i64, index) -> i64
    %sum = "tx.addvv"(%dst_addr, %dst_addr, %dst_addr, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----

// Async-only loop: no CPU-side SPM access, so no barrier is needed at all.
// CHECK-LABEL: func.func @async_only_loop
// CHECK-NOT:     tx.barrier

func.func @async_only_loop() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %a_ptr = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %a_addr = arith.index_cast %a_ptr : index to i64
  %b_ptr = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %b_addr = arith.index_cast %b_ptr : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %sum = "tx.addvv"(%a_addr, %b_addr, %a_addr, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----

// Loop-carried SPM buffer reached through an `scf.for` iter_arg: the CPU
// conversion touches the buffer through the block argument while the async op
// uses the underlying alloc. The membar analysis sees the loop-carried hazard
// through the allocation and inserts a barrier at the top of the body; it is
// hoisted only because a per-iteration barrier before scf.yield exists.
// CHECK-LABEL: func.func @fp8_convert_loop_carried_arg
// CHECK:         tx.barrier
// CHECK:         scf.for
// CHECK-NOT:     tx.barrier
// CHECK:           "tx.fp8E4M3FN_bf16"
// CHECK:           "tx.addvv"
// CHECK:           tx.barrier
// CHECK:         scf.yield

func.func @fp8_convert_loop_carried_arg() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %dst = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %dst_ptr = memref.extract_aligned_pointer_as_index %dst : memref<64xbf16> -> index
  %dst_addr = arith.index_cast %dst_ptr : index to i64
  scf.for %i = %c0 to %c4 step %c1 iter_args(%it = %dst) -> (memref<64xbf16>) {
    %it_ptr = memref.extract_aligned_pointer_as_index %it : memref<64xbf16> -> index
    %it_addr = arith.index_cast %it_ptr : index to i64
    %conv = "tx.fp8E4M3FN_bf16"(%it_addr, %it_addr, %n) : (i64, i64, index) -> i64
    %sum = "tx.addvv"(%dst_addr, %dst_addr, %dst_addr, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    scf.yield %it : memref<64xbf16>
  }
  return
}

// -----

// WDMA (async) writes a DDR region, then a CPU-executed FP8 conversion reads
// the same DDR region: a barrier is required before the conversion even though
// it is a tx-dialect op.
// CHECK-LABEL: func.func @fp8_convert_reads_wdma_ddr
// CHECK:         "tx.wdma1d"
// CHECK-NEXT:    tx.barrier
// CHECK-NEXT:    {{.*}}"tx.fp8E5M2_bf16"

func.func @fp8_convert_reads_wdma_ddr(%ddr: memref<64xf8E5M2>, %out: memref<64xbf16>) {
  %c64 = arith.constant 64 : index
  %c64_i32 = arith.constant 64 : i32
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %d = memref.extract_aligned_pointer_as_index %ddr : memref<64xf8E5M2> -> index
  %d_i64 = arith.index_cast %d : index to i64
  %o = memref.extract_aligned_pointer_as_index %out : memref<64xbf16> -> index
  %o_i64 = arith.index_cast %o : index to i64
  %s = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %s_i64 = arith.index_cast %s : index to i64
  "tx.wdma1d"(%d_i64, %s_i64, %c64_i32, %d_i64, %d_i64) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  %conv = "tx.fp8E5M2_bf16"(%d_i64, %o_i64, %c64) : (i64, i64, index) -> i64
  return
}

// -----

// Async producer outside the loop and an unrelated async op inside: one
// barrier before the loop is enough, no per-iteration barrier is required.
// CHECK-LABEL: func.func @outer_producer_inner_consumer
// CHECK:         "tx.addvv"
// CHECK-NEXT:    tx.barrier
// CHECK-NEXT:    scf.for
// CHECK-NOT:     tx.barrier
// CHECK:         return

func.func @outer_producer_inner_consumer() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %c = memref.alloc() {alignment = 256 : i64, allocation.offset = 512 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %cp = memref.extract_aligned_pointer_as_index %c : memref<64xbf16> -> index
  %ci = arith.index_cast %cp : index to i64
  %w = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  scf.for %i = %c0 to %c4 step %c1 {
    %conv = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %sum = "tx.addvv"(%ci, %ci, %ci, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}
