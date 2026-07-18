// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @commit_group_parse
  // LLVM-LABEL: llvm.func @commit_group_parse
  tt.func @commit_group_parse() {
    // CHECK: tle.tma_store.commit_group
    // LLVM: nvvm.cp.async.bulk.commit.group
    tle.tma_store.commit_group
    tt.return
  }
}
