// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' | FileCheck %s

// A callee receives the grid-wide frame base. Only ordinary scratch accesses
// add the physical CTA's stride, including in nested calls.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func internal @scratch_helper(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<3>, %[[BASE:arg[0-9]+]]: !llvm.ptr<1>,
  tt.func private @scratch_helper() attributes {noinline = true} {
    // CHECK: llvm.getelementptr %[[BASE]]
    %scratch = ttg.global_scratch_alloc {alignment = 4 : i32, nbytes = 4 : i32} : !tt.ptr<i32>
    %id = tt.get_program_id x : i32
    tt.store %scratch, %id : !tt.ptr<i32>
    // CHECK: llvm.getelementptr %[[BASE]]
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    tt.return
  }
  // CHECK-LABEL: llvm.func internal @scratch_middle(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<3>, %[[MID:arg[0-9]+]]: !llvm.ptr<1>,
  tt.func private @scratch_middle() attributes {noinline = true} {
    // CHECK-NOT: nvvm.read.ptx.sreg.ctaid
    // CHECK: llvm.call @scratch_helper
    tt.call @scratch_helper() : () -> ()
    tt.return
  }
  // CHECK-LABEL: llvm.func @scratch_kernel(
  // CHECK-SAME: %[[ROOT:arg[0-9]+]]: !llvm.ptr<1>,
  tt.func public @scratch_kernel() {
    // CHECK: llvm.call @scratch_middle
    tt.call @scratch_middle() : () -> ()
    tt.return
  }
}
