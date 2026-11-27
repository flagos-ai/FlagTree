// RUN: triton-opt %s -convert-scf-to-cf -allocate-amdgpu-shared-memory -convert-triton-amdgpu-to-llvm=arch=gfx1100 > %t
// RUN: FileCheck %s --check-prefix=SYNC < %t
// RUN: FileCheck %s --check-prefix=MEMBAR < %t

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#ring = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @entry(%out: !tt.ptr<i32>, %limit: i32) {
    %payload = ttg.local_alloc : () -> !ttg.memdesc<2x32xi32, #ring, #smem, mutable>
    %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %full, 32 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %empty, 32 {init_phase = 1 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.call @writer(%payload, %full, %empty, %limit) : (!ttg.memdesc<2x32xi32, #ring, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32) -> ()
    tt.call @reader(%payload, %full, %empty, %out) : (!ttg.memdesc<2x32xi32, #ring, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !tt.ptr<i32>) -> ()
    tt.return
  }

  // The pipe's payload and its two counters are different LDS allocations.
  // A workgroup rendezvous in either helper would also block the peer group.
  // MEMBAR-LABEL: llvm.func internal @writer
  // MEMBAR-NOT: rocdl.barrier
  // MEMBAR: llvm.return
  // SYNC-LABEL: llvm.func internal @writer
  // SYNC: llvm.load {{.*}} atomic syncscope("workgroup") acquire
  // SYNC: llvm.cmpxchg {{.*}} syncscope("workgroup") acq_rel
  tt.func private @writer(%payload: !ttg.memdesc<2x32xi32, #ring, #smem, mutable>, %full: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %empty: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %limit: i32) attributes {noinline = true} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %values = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    scf.for %i = %c0 to %limit step %c1 : i32 {
      %stage = arith.andi %i, %c1 : i32
      %slot = ttg.memdesc_index %payload[%stage] : !ttg.memdesc<2x32xi32, #ring, #smem, mutable> -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
      amdg.wait_barrier %empty, %stage : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.local_store %values, %slot : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
      %phase = amdg.arrive_barrier %full, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    }
    tt.return
  }

  // MEMBAR-LABEL: llvm.func internal @reader
  // MEMBAR-NOT: rocdl.barrier
  // MEMBAR: llvm.return
  // SYNC-LABEL: llvm.func internal @reader
  // SYNC: llvm.load {{.*}} atomic syncscope("workgroup") acquire
  // SYNC: llvm.cmpxchg {{.*}} syncscope("workgroup") acq_rel
  tt.func private @reader(%payload: !ttg.memdesc<2x32xi32, #ring, #smem, mutable>, %full: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %empty: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %out: !tt.ptr<i32>) attributes {noinline = true} {
    %c0 = arith.constant 0 : i32
    %slot = ttg.memdesc_index %payload[%c0] : !ttg.memdesc<2x32xi32, #ring, #smem, mutable> -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    amdg.wait_barrier %full, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %values = ttg.local_load %slot : !ttg.memdesc<32xi32, #shared, #smem, mutable> -> tensor<32xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %ptrs, %values : tensor<32x!tt.ptr<i32>, #blocked>
    %phase = amdg.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    tt.return
  }

  tt.func public @alias_entry(%value: tensor<32xi32, #blocked>) {
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    tt.call @alias(%buffer, %buffer, %value) : (!ttg.memdesc<32xi32, #shared, #smem, mutable>, !ttg.memdesc<32xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> ()
    tt.return
  }

  // Passing the same allocation to two formals still requires a barrier.
  // MEMBAR-LABEL: llvm.func internal @alias
  // MEMBAR: rocdl.barrier
  tt.func private @alias(%lhs: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %rhs: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %value: tensor<32xi32, #blocked>) attributes {noinline = true} {
    ttg.local_store %value, %lhs : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    ttg.local_store %value, %rhs : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    tt.return
  }
}
