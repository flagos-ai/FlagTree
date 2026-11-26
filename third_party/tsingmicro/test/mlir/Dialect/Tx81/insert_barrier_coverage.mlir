//===----------------------------------------------------------------------===//
// InsertBarrierPass coverage tests.
//
// Covers: SPM tx->CPU hazards and dedup, CPU->tx ordering, DDR WDMA->RDMA and
// WDMA->CPU RAW, loop-carried SPM sync (direct and through scf.for iter_args),
// barrier hoisting, kernel-entry sync, address-form resolution (subview / i64
// arithmetic) and the known gather-scatter gap.
//
// RUN: tsingmicro-opt --tx81-insert-barrier --split-input-file %s | FileCheck %s
//===----------------------------------------------------------------------===//


// -----
// tx<->tx on SPM is ordered by NCC: no barrier.
// CHECK-LABEL: func.func @c01_async_only
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c01_async_only() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %x = "tx.addvv"(%ai, %bi, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %y = "tx.addvv"(%bi, %ai, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  return
}

// -----
// async SPM write -> CPU read needs one barrier.
// CHECK-LABEL: func.func @c02_tx_then_cpu_load
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c02_tx_then_cpu_load() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// one barrier covers the whole CPU chain.
// CHECK-LABEL: func.func @c03_two_cpu_loads_dedup
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c03_two_cpu_loads_dedup() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %a[%c0] : memref<64xbf16>
  %w = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// CPU work before an async op is ordered by the KCORE; the leading barrier only suppresses the kernel-entry rule.
// CHECK-LABEL: func.func @c04_cpu_before_tx
// CHECK: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NEXT: "tx.addvv"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c04_cpu_before_tx() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  tx.barrier
  %v = memref.load %a[%c0] : memref<64xbf16>
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  return
}

// -----
// async write to a, CPU read of b: no hazard (leading barrier suppresses the entry rule).
// CHECK-LABEL: func.func @c05_tx_cpu_disjoint
// CHECK: tx.barrier
// CHECK-NEXT: "tx.addvv"
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c05_tx_cpu_disjoint() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  tx.barrier
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %b[%c0] : memref<64xbf16>
  return
}

// -----
// an explicit barrier between producer and consumer is not duplicated.
// CHECK-LABEL: func.func @c06_existing_barrier_dedup
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c06_existing_barrier_dedup() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  tx.barrier
  %v = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// one barrier per tx->CPU chain.
// CHECK-LABEL: func.func @c07_two_chains
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NEXT: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c07_two_chains() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %a[%c0] : memref<64xbf16>
  %y = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %w = memref.load %b[%c0] : memref<64xbf16>
  return
}

// -----
// CPU-executed FP8 conversion is a CPU consumer: barrier before it.
// CHECK-LABEL: func.func @c08_conv_consumer
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}"tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c08_conv_consumer() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %c = "tx.fp8E4M3FN_bf16"(%bi, %ai, %n) : (i64, i64, index) -> i64
  return
}

// -----
// CPU-executed conversion before an async op: program order, no barrier.
// CHECK-LABEL: func.func @c09_conv_producer
// CHECK: tx.barrier
// CHECK-NEXT: {{.*}}"tx.fp8E4M3FN_bf16"
// CHECK-NEXT: "tx.addvv"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c09_conv_producer() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  tx.barrier
  %c = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
  %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  return
}

// -----
// WDMA->RDMA on the same DDR origin needs a barrier.
// CHECK-LABEL: func.func @c10_ddr_war
// CHECK: "tx.wdma1d"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: "tx.rdma1d"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c10_ddr_war(%d: memref<64xf8E5M2>) {
  %c64_i32 = arith.constant 64 : i32
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %dp = memref.extract_aligned_pointer_as_index %d : memref<64xf8E5M2> -> index
  %di = arith.index_cast %dp : index to i64
  %sp = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %si = arith.index_cast %sp : index to i64
  "tx.wdma1d"(%di, %si, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  "tx.rdma1d"(%si, %di, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  return
}

// -----
// WDMA/RDMA on different DDR origins: no barrier.
// CHECK-LABEL: func.func @c11_ddr_disjoint
// CHECK: "tx.wdma1d"
// CHECK-NEXT: "tx.rdma1d"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c11_ddr_disjoint(%d1: memref<64xf8E5M2>, %d2: memref<64xf8E5M2>) {
  %c64_i32 = arith.constant 64 : i32
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %d1p = memref.extract_aligned_pointer_as_index %d1 : memref<64xf8E5M2> -> index
  %d1i = arith.index_cast %d1p : index to i64
  %d2p = memref.extract_aligned_pointer_as_index %d2 : memref<64xf8E5M2> -> index
  %d2i = arith.index_cast %d2p : index to i64
  %sp = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %si = arith.index_cast %sp : index to i64
  "tx.wdma1d"(%d1i, %si, %c64_i32, %d1i, %d1i) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  "tx.rdma1d"(%si, %d2i, %c64_i32, %d2i, %d2i) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  return
}

// -----
// WDMA->WDMA is left to the DMA queue.
// CHECK-LABEL: func.func @c12_ddr_ww
// CHECK: "tx.wdma1d"
// CHECK-NEXT: "tx.wdma1d"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c12_ddr_ww(%d: memref<64xf8E5M2>) {
  %c64_i32 = arith.constant 64 : i32
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %dp = memref.extract_aligned_pointer_as_index %d : memref<64xf8E5M2> -> index
  %di = arith.index_cast %dp : index to i64
  %sp = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %si = arith.index_cast %sp : index to i64
  "tx.wdma1d"(%di, %si, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  "tx.wdma1d"(%di, %si, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  return
}

// -----
// CPU read of DDR written by a pending WDMA needs a barrier.
// CHECK-LABEL: func.func @c13_ddr_cpu_read
// CHECK: "tx.wdma1d"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c13_ddr_cpu_read(%d: memref<64xf8E5M2>) {
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %dp = memref.extract_aligned_pointer_as_index %d : memref<64xf8E5M2> -> index
  %di = arith.index_cast %dp : index to i64
  %sp = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %si = arith.index_cast %sp : index to i64
  "tx.wdma1d"(%di, %si, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  %v = memref.load %d[%c0] : memref<64xf8E5M2>
  return
}

// -----
// an explicit barrier clears the pending-DDR set; no duplicate.
// CHECK-LABEL: func.func @c14_ddr_pending_cleared
// CHECK: "tx.wdma1d"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: "tx.rdma1d"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c14_ddr_pending_cleared(%d: memref<64xf8E5M2>) {
  %c64_i32 = arith.constant 64 : i32
  %spm = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E5M2>
  %dp = memref.extract_aligned_pointer_as_index %d : memref<64xf8E5M2> -> index
  %di = arith.index_cast %dp : index to i64
  %sp = memref.extract_aligned_pointer_as_index %spm : memref<64xf8E5M2> -> index
  %si = arith.index_cast %sp : index to i64
  "tx.wdma1d"(%di, %si, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  tx.barrier
  "tx.rdma1d"(%si, %di, %c64_i32, %di, %di) {fmt = 3 : i32} : (i64, i64, i32, i64, i64) -> ()
  return
}

// -----
// loop-carried hazard: one barrier before the loop (hoisted) plus one before scf.yield.
// CHECK-LABEL: func.func @c15_loop_direct
// CHECK: tx.barrier
// CHECK: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }

func.func @c15_loop_direct() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %c = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----
// same, with the buffer reached through an scf.for iter_arg.
// CHECK-LABEL: func.func @c16_loop_iterarg
// CHECK: tx.barrier
// CHECK: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: scf.yield

func.func @c16_loop_iterarg() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 iter_args(%it = %b) -> (memref<64xbf16>) {
    %itp = memref.extract_aligned_pointer_as_index %it : memref<64xbf16> -> index
    %iti = arith.index_cast %itp : index to i64
    %c = "tx.fp8E4M3FN_bf16"(%iti, %iti, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    scf.yield %it : memref<64xbf16>
  }
  return
}

// -----
// conversion and async op use different buffers: only the kernel-entry barrier.
// CHECK-LABEL: func.func @c17_loop_disjoint
// CHECK: tx.barrier
// CHECK-NEXT: scf.for
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c17_loop_disjoint() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %c = memref.alloc() {alignment = 256 : i64, allocation.offset = 512 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %cp = memref.extract_aligned_pointer_as_index %c : memref<64xbf16> -> index
  %ci = arith.index_cast %cp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %k = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%ci, %ci, %ci, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----
// explicit yield barrier: the body-top barrier is hoisted, nothing is duplicated.
// CHECK-LABEL: func.func @c18_loop_explicit_yield
// CHECK: tx.barrier
// CHECK: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }

func.func @c18_loop_explicit_yield() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %c = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    tx.barrier
  }
  return
}

// -----
// nested loop: inner and outer per-iteration barriers, plus the pre-loop barrier.
// CHECK-LABEL: func.func @c19_nested
// CHECK: tx.barrier
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }

func.func @c19_nested() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %k = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
      %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    }
  }
  return
}

// -----
// kernel starts with a CPU SPM access: one entry barrier.
// CHECK-LABEL: func.func @c20_entry_cpu_load
// CHECK: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c20_entry_cpu_load() {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %v = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// kernel starts with a CPU-executed conversion: one entry barrier.
// CHECK-LABEL: func.func @c21_entry_convert
// CHECK: tx.barrier
// CHECK-NEXT: {{.*}}"tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c21_entry_convert() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %c = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
  return
}

// -----
// async op first: the barrier is placed before the CPU consumer, not at entry.
// CHECK-LABEL: func.func @c22_async_then_cpu
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c22_async_then_cpu() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// CPU access through memref.subview resolves to the base buffer.
// CHECK-LABEL: func.func @c23_subview
// CHECK: "tx.addvv"
// CHECK: memref.subview
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c23_subview() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %sub = memref.subview %a[0] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1]>>
  %v = memref.load %sub[%c0] : memref<32xbf16, strided<[1]>>
  return
}

// -----
// i64 address arithmetic (addi) resolves to the base buffer.
// CHECK-LABEL: func.func @c24_address_offset
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c24_address_offset() {
  %n = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : i64
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %aoff = arith.addi %ai, %c8 : i64
  %x = "tx.addvv"(%aoff, %aoff, %aoff, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  %v = memref.load %a[%c0] : memref<64xbf16>
  return
}

// -----
// memref.copy is a CPU access and takes part in the hazard.
// CHECK-LABEL: func.func @c25_memref_copy
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.copy
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c25_memref_copy() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  memref.copy %a, %b : memref<64xbf16> to memref<64xbf16>
  return
}

// -----
// KNOWN GAP: gather/scatter -> data op gets no barrier (tx<->tx suppressed). Safe only with the synchronous __GatherScatter entry (default); TRITON_GATHER_SCATTER_ASYNC=1 hits the documented K64 hang.
// CHECK-LABEL: func.func @c26_gatherscatter_then_data
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c26_gatherscatter_then_data() {
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  %gs = "tx.gatherscatter"(%ai, %bi) {bytes = 128 : i32, src_strideN = 0 : i32, src_strideH = 0 : i32, src_strideW = 1 : i32, src_iterN = 1 : i32, src_iterH = 1 : i32, src_iterW = 64 : i32, dst_strideN = 0 : i32, dst_strideH = 0 : i32, dst_strideW = 1 : i32, dst_iterN = 1 : i32, dst_iterH = 1 : i32, dst_iterW = 64 : i32} : (i64, i64) -> i64
  %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  return
}

// -----
// tx write followed by a CPU read in the body: the barrier stays inside the loop (per-iteration sync), nothing is hoisted.
// CHECK-LABEL: func.func @c27_loop_tx_then_cpu
// CHECK-NOT: tx.barrier
// CHECK: scf.for
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c27_loop_tx_then_cpu() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xbf16> -> index
  %ai = arith.index_cast %ap : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %x = "tx.addvv"(%ai, %ai, %ai, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    %v = memref.load %a[%i] : memref<64xbf16>
  }
  return
}

// -----
// each hazard loop gets its own pre-loop barrier plus a yield barrier.
// CHECK-LABEL: func.func @c28_two_loops
// CHECK: tx.barrier
// CHECK-NEXT: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: scf.for
// CHECK-NOT: tx.barrier
// CHECK: "tx.fp8E4M3FN_bf16"
// CHECK-NOT: tx.barrier
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: }
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c28_two_loops() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %c = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  scf.for %j = %c0 to %c4 step %c1 {
    %d = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %y = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----
// in-body barrier (async write before the conversion) already drains every iteration: the redundant scf.yield barrier is suppressed.
// CHECK-LABEL: func.func @c29_loop_in_body_barrier
// CHECK-NOT: tx.barrier
// CHECK: scf.for
// CHECK: "tx.addvv"
// CHECK-NEXT: tx.barrier
// CHECK-NEXT: {{.*}}"tx.fp8E4M3FN_bf16"
// CHECK-NEXT: "tx.addvv"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c29_loop_in_body_barrier() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    %w = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    %cv = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}

// -----
// the tx op guarding the in-body barrier may be nested in an scf.if; it still keeps the barrier from being hoisted, so no yield barrier is added.
// CHECK-LABEL: func.func @c30_nested_tx_before_guard
// CHECK-NOT: tx.barrier
// CHECK: scf.for
// CHECK: scf.if
// CHECK: "tx.addvv"
// CHECK: tx.barrier
// CHECK-NEXT: {{.*}}"tx.fp8E4M3FN_bf16"
// CHECK-NEXT: "tx.addvv"
// CHECK-NOT: tx.barrier
// CHECK: return

func.func @c30_nested_tx_before_guard() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  %true = arith.constant true
  %a = memref.alloc() {alignment = 256 : i64, allocation.offset = 0 : i32} : memref<64xf8E4M3FN>
  %b = memref.alloc() {alignment = 256 : i64, allocation.offset = 256 : i32} : memref<64xbf16>
  %ap = memref.extract_aligned_pointer_as_index %a : memref<64xf8E4M3FN> -> index
  %ai = arith.index_cast %ap : index to i64
  %bp = memref.extract_aligned_pointer_as_index %b : memref<64xbf16> -> index
  %bi = arith.index_cast %bp : index to i64
  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %true {
      %w = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
    }
    %cv = "tx.fp8E4M3FN_bf16"(%ai, %bi, %n) : (i64, i64, index) -> i64
    %x = "tx.addvv"(%bi, %bi, %bi, %n) <{rnd_mode = 0 : i16, fmt = 3 : i16}> : (i64, i64, i64, index) -> i64
  }
  return
}
